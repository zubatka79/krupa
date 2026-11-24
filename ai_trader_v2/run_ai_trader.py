"""
Autonomous 24/7 15m 12-School TA + AI Judge Trading Bot (`ai_trader_v2/run_ai_trader.py`).

Completely isolated from live bots:
  - Own config:  ai_trader_v2/config_ai_trader.json
  - Own cache:   ai_trader_v2/data/cache/
  - Own SQLite:  ai_trader_v2/data/ai_trades.db
  - Own AI log:  ai_trader_v2/data/ai_usage.json

Usage:
  .venv\\Scripts\\python.exe ai_trader_v2\\run_ai_trader.py --once          # 1 live scan + AI judge (no loop)
  .venv\\Scripts\\python.exe ai_trader_v2\\run_ai_trader.py --paper         # 24/7 paper mode + Telegram + SQLite
  .venv\\Scripts\\python.exe ai_trader_v2\\run_ai_trader.py --testnet       # 24/7 Bybit Testnet real orders
  .venv\\Scripts\\python.exe ai_trader_v2\\run_ai_trader.py --status        # Print portfolio & AI statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trader_v2.ai_judge import AITradeJudge, redact_secrets
from ai_trader_v2.executor import AITradeExecutor, TelegramNotifier
from ai_trader_v2.journal import AITradeJournal
from ai_trader_v2.ta_schools import CandidateSetup, build_btc_context, evaluate_symbol_12_schools
from ai_trader_v2.telegram_ui import AITraderTelegramUI, is_symbol_paused, load_bot_state
from bot.bybit_client import BybitClient
from bot.news_guard import NewsGuard

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("AI_TRADER_DATA_DIR", str(ROOT / "data" / "ai_trader_v2")))
LOG_DIR = Path(os.getenv("AI_TRADER_LOG_DIR", str(ROOT / "logs")))
CONFIG_PATH = APP_DIR / "config_ai_trader.json"
CACHE_DIR = DATA_DIR / "cache"
MACRO_STATE_PATH = DATA_DIR / "news_state.json"
MACRO_EVENTS_PATH = DATA_DIR / "macro_events.json"


class RedactingLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        if record.args:
            record.args = tuple(redact_secrets(a) if isinstance(a, str) else a for a in record.args)
        return True


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ai_trader_v2")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "ai_trader_v2.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.addFilter(RedactingLogFilter())
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.addFilter(RedactingLogFilter())
    logger.addHandler(fh)
    logger.addHandler(sh)
    # Also route module loggers to same handlers
    for mod in ("ai_trader_v2.ta_schools", "ai_trader_v2.ai_judge", "ai_trader_v2.executor", "ai_trader_v2.journal"):
        ml = logging.getLogger(mod)
        ml.setLevel(logging.INFO)
    return logger


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # Validate mandatory numeric bounds (trading-core.md)
    assert 3 <= int(cfg["leverage"]) <= 10, "leverage must be in [3, 10]"
    assert 1.0 <= float(cfg["margin_pct_per_trade"]) <= 15.0, "margin_pct_per_trade must be in [1, 15]"
    assert 0.5 <= float(cfg["max_risk_pct_per_trade"]) <= 2.5, "max_risk_pct_per_trade must be <= 2.5"
    assert 1.2 <= float(cfg["sl_min_pct"]) <= float(cfg["sl_max_pct"]) <= 1.6, "SL must be inside [1.2%, 1.6%]"
    assert float(cfg["daily_loss_limit_pct"]) <= 6.0, "daily_loss_limit_pct must be <= 6.0%"
    return cfg


def current_closed_15m_ts() -> int:
    now_ms = int(time.time() * 1000)
    return ((now_ms // 900_000) - 1) * 900_000


def run_scan_cycle(
    client: BybitClient,
    judge: AITradeJudge,
    executor: AITradeExecutor,
    journal: AITradeJournal,
    news_guard: NewsGuard,
    config: Dict[str, Any],
    logger: logging.Logger,
    allow_execute: bool = True,
) -> Dict[str, Any]:
    """Run one complete closed-15m scan across all 29 coins & all 12 TA schools + AI Judge."""
    t0 = time.time()
    symbols: List[str] = list(config["symbols"])
    stats: Dict[str, Any] = {
        "scanned": 0,
        "candidates": 0,
        "ai_called": 0,
        "entered": 0,
        "skipped": 0,
        "btc_trend": "UNKNOWN",
        "macro_blocked": False,
        "details": [],
    }

    # 1. Manage active positions & daily -6% drawdown limit first
    executor.manage_active_positions()
    bot_state = load_bot_state()
    if bool(bot_state.get("global_paused", False)):
        logger.info("Scan skipped: global pause active via Telegram UI.")
        stats["halted"] = "global_paused"
        return stats
    ok_dd, dd_reason = executor.check_daily_loss_limit()
    if not ok_dd:
        logger.warning("Scan halted by daily risk guard: %s", dd_reason)
        stats["halted"] = dd_reason
        return stats

    # 2. Macro News Shield check (US CPI / FOMC / NFP)
    if bool(config.get("macro_shield_enabled", True)):
        try:
            shield = news_guard.evaluate_macro_shield(allow_network_refresh=True)
            if not shield.get("entry_allowed", True):
                logger.warning("Macro shield blocking entries: %s", shield.get("reason"))
                stats["macro_blocked"] = True
                stats["macro_reason"] = shield.get("reason")
                return stats
        except Exception as exc:
            logger.warning("Macro shield check error (continuing cautiously): %s", exc)

    # 3. BTCUSDT 15m/1H/4H macro gate
    btc_ctx = build_btc_context(client, shock_limit_pct=float(config.get("btc_shock_pause_pct", 0.9)))
    if btc_ctx is None:
        logger.error("Failed to fetch BTCUSDT context — aborting scan cycle.")
        return stats
    stats["btc_trend"] = f"1H:{btc_ctx.h1_trend} / 4H:{btc_ctx.h4_trend} (15m move {btc_ctx.bar_move_15m_pct:.2f}%)"
    if btc_ctx.shock_active:
        logger.warning("BTC shock active (%.2f%%) — pausing new entries this bar.", btc_ctx.bar_move_15m_pct)
        return stats

    # 4. Scan all 29 liquid coins across ALL 12 TA schools
    candidates: List[CandidateSetup] = []
    for sym in symbols:
        if is_symbol_paused(sym, bot_state):
            continue
        stats["scanned"] += 1
        try:
            cand = evaluate_symbol_12_schools(client, sym, btc_ctx, config)
            if cand is not None:
                candidates.append(cand)
        except Exception as exc:
            logger.warning("Error scanning %s: %s", sym, exc)

    candidates.sort(key=lambda c: c.priority_score, reverse=True)
    stats["candidates"] = len(candidates)
    logger.info(
        "Scan complete in %.1fs | BTC %s | Scanned %d coins -> %d candidates passed 1H+4H+12-schools+SL+OB gates.",
        time.time() - t0,
        stats["btc_trend"],
        stats["scanned"],
        len(candidates),
    )

    if not candidates:
        return stats

    # 5. Send top candidates to AI Judge (sees full 15m/1H/4H/BTC candle picture!)
    max_ai = int(config.get("max_ai_candidates_per_bar", 3))
    for cand in candidates[:max_ai]:
        # Skip if already judged on this exact closed 15m bar
        if journal.has_decided_bar(cand.symbol, cand.bar_ts):
            logger.info("Already judged %s on bar %d — skipping duplicate AI call.", cand.symbol, cand.bar_ts)
            continue

        ok_slot, slot_reason = executor.can_open_symbol(cand.symbol, cand.direction)
        if not ok_slot:
            logger.info("Candidate %s (%s) skipped before AI: %s", cand.symbol, cand.direction, slot_reason)
            continue

        portfolio = executor.get_portfolio_state()
        decision = judge.evaluate_candidate(cand, portfolio, config)
        stats["ai_called"] += 1

        journal.log_ai_decision(
            bar_ts=cand.bar_ts,
            symbol=cand.symbol,
            direction=cand.direction,
            decision=decision.decision,
            confidence=decision.confidence,
            primary_school=decision.selected_school,
            fired_schools=cand.fired_schools,
            price=cand.close_price,
            sl_pct=decision.sl_pct,
            tp1_r=decision.tp1_r,
            tp2_r=decision.tp2_r,
            regime=cand.regime_label,
            provider=decision.provider,
            reason_ru=decision.reason_ru,
            warnings=decision.risk_warnings,
            guardrail_notes=decision.guardrail_notes,
        )

        detail = {
            "symbol": cand.symbol,
            "direction": cand.direction,
            "schools": cand.fired_schools,
            "decision": decision.decision,
            "confidence": decision.confidence,
            "sl_pct": decision.sl_pct,
            "tp1_r": decision.tp1_r,
            "tp2_r": decision.tp2_r,
            "reason_ru": decision.reason_ru,
            "guardrails": decision.guardrail_notes,
        }
        stats["details"].append(detail)
        logger.info(
            "AI VERDICT %s %s -> %s (%d/100) | Schools=%s | SL=%.2f%% TP1=%.1fR TP2=%.1fR | %s",
            cand.symbol,
            cand.direction,
            decision.decision,
            decision.confidence,
            ",".join(cand.fired_schools),
            decision.sl_pct,
            decision.tp1_r,
            decision.tp2_r,
            decision.reason_ru,
        )

        if decision.decision == "ENTER":
            if allow_execute:
                tid = executor.open_trade(cand, decision)
                if tid is not None:
                    stats["entered"] += 1
        else:
            stats["skipped"] += 1
            send_skips = bool(bot_state.get("telegram_send_skips", config.get("telegram_send_skips", False)))
            if send_skips:
                executor.notifier.send(
                    f"⏭️ [AI SKIP] {cand.symbol} ({cand.direction}) | Уверенность: {decision.confidence}/100\n"
                    f"Школы: {', '.join(cand.fired_schools)}\nПричина: {decision.reason_ru}"
                )

    return stats


def print_status(journal: AITradeJournal, judge: AITradeJudge, config: Dict[str, Any]) -> None:
    rep = journal.get_summary_report(float(config.get("paper_start_equity_usdt", 2000.0)))
    bud = judge.get_budget_status()
    print("\n================ AI TRADER v2 STATUS ================")
    print(f"AI Provider      : {judge.active_provider_label}")
    print(f"AI Budget ({bud['month']}): ${bud['spent_usd']:.4f} / ${bud['limit_usd']:.2f} ({bud['requests']} reqs)")
    print(f"Paper Equity     : ${rep['paper_equity_usdt']:.2f}")
    print(f"Closed Trades    : {rep['closed_trades']} (Wins: {rep['wins']} | Losses: {rep['losses']} | WR: {rep['win_rate_pct']}%)")
    print(f"Net PnL / PF     : ${rep['net_pnl_usdt']:+.2f} | PF {rep['profit_factor']:.2f}")
    print(f"AI Decisions     : ENTER={rep['ai_enter_count']} | SKIP={rep['ai_skip_count']}")
    print(f"Active Positions : {rep['active_trades']} / {config['max_open_positions']}")
    for p in rep["active_positions"]:
        print(
            f"  #{p['id']} [{p['status']}] {p['symbol']} {p['direction']} @ {p['entry_price']} | "
            f"SL {p['sl_price']} | TP1 {p['tp1_price']} | TP2 {p['tp2_price']} | AI {p['ai_confidence']}/100"
        )
    print("=====================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous 15m 12-School TA + AI Judge Trading Bot")
    parser.add_argument("--once", action="store_true", help="Run a single live closed-15m scan cycle and exit")
    parser.add_argument("--paper", action="store_true", help="Run in live Paper mode (real candles + Telegram + SQLite)")
    parser.add_argument("--testnet", action="store_true", help="Run with REAL orders on Bybit Testnet")
    parser.add_argument("--demo", action="store_true", help="Run with REAL orders on Bybit Demo Trading")
    parser.add_argument("--no-execute", action="store_true", help="Scan & ask AI Judge only without opening trades")
    parser.add_argument("--status", action="store_true", help="Print SQLite journal & AI budget status and exit")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config()
    journal = AITradeJournal()
    judge = AITradeJudge(monthly_limit_usd=float(config.get("ai_monthly_limit_usd", 50.0)))

    if args.status:
        print_status(journal, judge, config)
        return

    # Strict execution mode selection (NEVER mainnet autonomously!)
    if args.testnet:
        mode = "testnet"
        paper_mode = False
    elif args.demo:
        mode = "demo"
        paper_mode = False
    elif args.paper:
        mode = "testnet"
        paper_mode = True
    else:
        mode = str(config.get("mode", "testnet")).lower().strip()
        if mode == "mainnet":
            raise SystemExit("SAFETY LOCK: mainnet cannot be started autonomously. Use --paper, --testnet, or --demo.")
        paper_mode = bool(config.get("dry_run", True))

    client = BybitClient(mode=mode, cache_dir=CACHE_DIR, public_mainnet=True)
    notifier = TelegramNotifier(enabled=bool(config.get("telegram_enabled", True)) and (not args.no_execute))
    executor = AITradeExecutor(client=client, journal=journal, config=config, notifier=notifier, paper_mode=paper_mode)
    news_guard = NewsGuard(
        config=config,
        bybit_client=client,
        state_path=MACRO_STATE_PATH,
        macro_events_path=MACRO_EVENTS_PATH,
    )

    logger.info(
        "Starting AI-Trader v2 | Exec=%s | AI=%s | Coins=%d | 12 Schools | Lev=%dx Margin=%.0f%% | MaxPos=%d (SameDir=%d)",
        "PAPER" if executor.paper_mode else client.mode.upper(),
        judge.active_provider_label,
        len(config["symbols"]),
        config["leverage"],
        config["margin_pct_per_trade"],
        config["max_open_positions"],
        config["max_same_direction"],
    )

    if args.once:
        res = run_scan_cycle(
            client, judge, executor, journal, news_guard, config, logger, allow_execute=(not args.no_execute)
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print_status(journal, judge, config)
        return

    # Start completely redesigned AI-Trader v2 Telegram Control Center (buttons + slash commands)
    tg_ui = AITraderTelegramUI(
        client=client,
        executor=executor,
        journal=journal,
        judge=judge,
        news_guard=news_guard,
        config=config,
        scan_callback=lambda: run_scan_cycle(
            client, judge, executor, journal, news_guard, config, logger, allow_execute=True
        ),
    )
    tg_ui.start_background()
    tg_ui.send(tg_ui.build_welcome_text(), reply_markup=tg_ui.main_menu_keyboard())

    last_scanned_bar = -1
    last_briefing_key = ""
    while True:
        try:
            executor.manage_active_positions()

            now_dt = datetime.now(timezone.utc)
            now_s = time.time()
            closed_ts = current_closed_15m_ts()
            sec_since_bar_close = now_s - ((closed_ts + 900_000) / 1000.0)

            # Run full 29-coin x 12-school scan strictly after 15m bar closes (+8s buffer)
            if closed_ts > last_scanned_bar and sec_since_bar_close >= 8.0:
                last_scanned_bar = closed_ts
                run_scan_cycle(client, judge, executor, journal, news_guard, config, logger, allow_execute=True)

            # Automated 07:00 UTC (morning) & 19:00 UTC (evening) Telegram briefings
            b_key = f"{now_dt.strftime('%Y-%m-%d')}_{now_dt.hour}"
            if now_dt.hour in (7, 19) and now_dt.minute < 5 and b_key != last_briefing_key:
                last_briefing_key = b_key
                tg_ui.send(
                    tg_ui.build_period_report_text(1) + "\n\n" + tg_ui.build_positions_text(),
                    reply_markup=tg_ui.main_menu_keyboard(),
                )

            time.sleep(20)
        except KeyboardInterrupt:
            logger.info("AI-Trader v2 stopped gracefully by user.")
            tg_ui.stop()
            break
        except Exception as exc:
            logger.error("Main loop error: %s", redact_secrets(exc))
            try:
                client.reset_sessions()
            except Exception:
                pass
            time.sleep(15)


if __name__ == "__main__":
    main()
