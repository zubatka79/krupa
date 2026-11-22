"""Main autonomous trading engine for Bybit USDT Perpetuals (`fibo_trend_v2` + Two-Layer NewsGuard)."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

from bot.bybit_client import BybitClient, INTERVAL_TO_MS, round_step_down, round_tick
from bot.executor import TradeExecutor
from bot.journal import SignalJournal
from bot.news_guard import NewsGuard
from bot.risk_manager import RiskManager
from bot.scanner import MarketScanner

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_bot")


def parse_id_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip() and "your_" not in x.lower() and x.strip().lstrip("-").isdigit()]


class TelegramNotifier:
    """Lightweight Telegram Bot API notifier (silently disabled if env vars are unset)."""

    def __init__(self) -> None:
        self._reload()

    def _reload(self) -> None:
        load_dotenv(override=True)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_ids = parse_id_list(os.getenv("TELEGRAM_CHAT_ID", "")) or parse_id_list(
            os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
        )
        self.enabled = bool(self.token and self.chat_ids and "your_telegram" not in self.token.lower())

    def _redact(self, text: str) -> str:
        return text.replace(self.token, "***") if self.token else text

    def send(self, text: str) -> None:
        if not self.enabled:
            self._reload()
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        for chat_id in self.chat_ids:
            try:
                resp = requests.post(
                    url,
                    json={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True},
                    timeout=8,
                )
                if resp.status_code != 200:
                    logger.warning("Telegram notification to %s failed: HTTP %s", chat_id, resp.status_code)
            except Exception as exc:
                logger.warning("Telegram notification failed: %s", self._redact(str(exc)))


REQUIRED_CONFIG_KEYS = ("mode", "strategy", "whitelist", "leverage", "margin_pct_per_trade", "sl_pct", "entry_tf")


def load_config(path: str | Path = "config.json") -> Dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config root must be an object")
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"missing keys: {missing}")
    lev = int(cfg["leverage"])
    if not (1 <= lev <= int(cfg.get("max_leverage", lev)) and lev <= 25):
        raise ValueError(f"leverage {lev} out of bounds")
    if not (0 < float(cfg["margin_pct_per_trade"]) <= 50):
        raise ValueError("margin_pct_per_trade must be in (0, 50]")
    if not (0 < float(cfg["sl_pct"]) <= 10):
        raise ValueError("sl_pct must be in (0, 10]")
    if str(cfg["entry_tf"]) not in INTERVAL_TO_MS:
        raise ValueError(f"unsupported entry_tf {cfg['entry_tf']}")
    if str(cfg["mode"]).lower() not in ("demo", "testnet", "live", "mainnet"):
        raise ValueError(f"unknown mode {cfg['mode']}")
    return cfg


class _RedactTokens(logging.Filter):
    """Strip bot tokens and API keys that requests/urllib3 put into exception texts."""

    _patterns = (re.compile(r"bot\d+:[A-Za-z0-9_-]{20,}"), re.compile(r"key=[A-Za-z0-9_-]{20,}"))

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        clean = msg
        for pat in self._patterns:
            clean = pat.sub("***", clean)
        if clean != msg:
            record.msg, record.args = clean, None
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Bybit USDT Perpetual Trading Bot (fibo_trend_v2)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--dry-run", action="store_true", help="Run on live data & live calendar without sending real orders")
    parser.add_argument("--once", action="store_true", help="Run one complete NewsGuard + sync + scan cycle and exit")
    parser.add_argument("--poll-sec", type=int, default=15, help="Position sync interval in seconds (default: 15)")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    mode = str(config.get("mode", "testnet")).lower()
    db_path = str(config.get("db_path", "data/trades.db"))
    log_path = str(config.get("log_path", "logs/bot.log"))
    news_state_path = str(config.get("news_state_path", "data/news_state.json"))
    macro_events_path = str(config.get("macro_events_path", "data/macro_events.json"))

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler)):
            root_logger.removeHandler(h)
            h.close()
    fh = logging.handlers.RotatingFileHandler(log_path, maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(fh)
    for h in root_logger.handlers:
        h.addFilter(_RedactTokens())

    notifier = TelegramNotifier()

    # Always read real live public market klines/tickers from Bybit mainnet; private orders obey `mode` ("demo" / "testnet")
    client = BybitClient(mode=mode, cache_dir="data/cache", public_mainnet=True)
    risk_manager = RiskManager(config)

    # Create NewsGuard first so it is wired as a HARD core dependency in both Scanner and Executor
    news_guard = NewsGuard(
        config=config,
        bybit_client=client,
        state_path=news_state_path,
        macro_events_path=macro_events_path,
    )
    scanner = MarketScanner(
        client=client,
        risk_manager=risk_manager,
        config=config,
        news_guard=news_guard,
        news_state_path=news_state_path,
    )
    executor = TradeExecutor(
        client=client,
        risk_manager=risk_manager,
        config=config,
        db_path=db_path,
        dry_run=args.dry_run,
        notifier=notifier.send,
        news_guard=news_guard,
        trend_provider=scanner.get_latest_1h_trend,
        news_state_path=news_state_path,
    )
    journal = SignalJournal(db_path, strategy=str(config.get("strategy", "")))

    def _on_news_pause(layer: str, pause_until_utc: str, reason: str) -> None:
        executor.log_news_pause(layer, pause_until_utc, reason)
        notifier.send(f"[MACRO SHIELD] New entries blocked until {pause_until_utc} | [{layer}] {reason}")

    news_guard.on_pause_callback = _on_news_pause

    running = True

    def _handle_stop(signum: int, _frame: Any) -> None:
        nonlocal running
        logger.info("Received signal %s, shutting down gracefully...", signum)
        running = False

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    eff_mode = f"DRY-RUN ({mode.upper()})" if executor.dry_run else mode.upper()
    max_notional_cfg = float(config.get("max_notional_usdt", 0.0) or 0.0)
    cap_note = f" | MaxNotionalCap=${max_notional_cfg:,.0f}" if max_notional_cfg > 0 else ""
    start_msg = (
        f"Bybit Autonomous Bot started | Config: {args.config} | Mode: {eff_mode} | Endpoint: {client.private_base_url} | "
        f"Strategy: {config.get('strategy')} | Symbols: {config.get('whitelist')} | "
        f"DB: {db_path} | Log: {log_path} | NewsState: {news_state_path} | "
        f"Isolated {config.get('leverage')}x ({config.get('margin_pct_per_trade')}% margin{cap_note}) | "
        f"SL {config.get('sl_pct')}% | BTC1H Veto>={config.get('btc_1h_veto_pct')}% | "
        f"US Macro Shield (-{config.get('macro_pre_window_min')}m..+{config.get('macro_post_window_min')}m, "
        f"uPnL<+{config.get('macro_protect_upnl_pct')}%) | Max {config.get('max_daily_stops')} SL/day"
    )
    logger.info(start_msg)
    notifier.send(start_msg)

    # Pre-flight connection, real USDT wallet balance & lot-sizing check
    preview_equity = 1000.0
    if client.has_credentials:
        try:
            bal = client.get_wallet_balance("USDT")
            preview_equity = float(bal.get("equity", 0.0)) or 1000.0
            logger.info(
                "[PRE-FLIGHT %s OK] Connected to %s | Real USDT Wallet Equity: $%.2f | Available: $%.2f",
                mode.upper(),
                client.private_base_url,
                float(bal.get("equity", 0.0)),
                float(bal.get("available", 0.0)),
            )
        except Exception as exc:
            logger.warning("[PRE-FLIGHT %s] Failed to fetch live USDT wallet balance from %s: %s", mode.upper(), client.private_base_url, exc)
    else:
        logger.info(
            "[PRE-FLIGHT %s] API keys not set in .env yet (using paper preview equity $1,000.00 USDT for %s).",
            mode.upper(),
            client.private_base_url,
        )

    try:
        wl_syms = [str(s).upper().strip() for s in config.get("whitelist", []) if str(s).strip()]
        specs_map = client.get_instrument_specs_for_symbols(wl_syms)
        lev = int(config.get("leverage", 15))
        mpct = float(config.get("margin_pct_per_trade", 15.0))
        sl_p = float(config.get("sl_pct", 2.50))
        for sym in wl_syms:
            sp = specs_map.get(sym)
            if sp is None or sp.last_price <= 0:
                continue
            raw_margin = preview_equity * (mpct / 100.0)
            raw_notional = raw_margin * lev
            if max_notional_cfg > 0 and raw_notional > max_notional_cfg:
                raw_notional = max_notional_cfg
            qty = round_step_down(raw_notional / sp.last_price, sp.qty_step)
            act_notional = qty * sp.last_price
            act_margin = act_notional / lev
            sl_long = round_tick(sp.last_price * (1.0 - sl_p / 100.0), sp.tick_size)
            logger.info(
                "[LOT PREVIEW %s] LivePrice=%s (tick=%s, qtyStep=%s, maxLev=%.0fx) | Equity=$%.2f -> "
                "Isolated %dx: Qty=%s | Margin=$%.2f USDT | Notional=$%.2f USDT%s | Sample LONG SL(-%.2f%%)=%s",
                sym,
                sp.last_price,
                sp.tick_size,
                sp.qty_step,
                sp.max_leverage,
                preview_equity,
                lev,
                qty,
                act_margin,
                act_notional,
                f" (Cap=${max_notional_cfg:,.0f})" if max_notional_cfg > 0 else "",
                sl_p,
                sl_long,
            )
    except Exception as exc:
        logger.warning("[LOT PREVIEW] Could not compute lot preview: %s", exc)

    entry_tf = str(config.get("entry_tf", "15"))
    entry_ms = INTERVAL_TO_MS[entry_tf]
    last_scanned_bar_ts = -1

    config_error: str | None = None
    while running:
        try:
            # Hot-reload the config; a broken file keeps the last good one so positions stay managed
            try:
                new_config = load_config(args.config)
                if str(new_config.get("mode", "")).lower() != mode:
                    raise ValueError("mode cannot be changed without a restart")
                config = new_config
                scanner.update_config(config)
                executor.update_config(config)
                news_guard.config = config
                if config_error is not None:
                    logger.info("Config %s is valid again.", args.config)
                    notifier.send(f"[CONFIG] {args.config} снова читается, применён.")
                config_error = None
            except Exception as exc:
                if str(exc) != config_error:
                    logger.error("Config reload failed (%s), keeping previous config.", exc)
                    notifier.send(f"[CONFIG ERROR] {args.config}: {exc}. Работаю на прежнем конфиге, новые входы приостановлены.")
                config_error = str(exc)

            # 1. Fail-CLOSED US Macro Shield update (auto-refreshes data/macro_events.json on schedule)
            try:
                news_state = news_guard.update(force_refresh=(args.once or last_scanned_bar_ts == -1))
            except Exception as exc:
                logger.exception("Macro shield update failed: %s", exc)
                news_state = {"entry_allowed": False, "upcoming_core_events": []}
            if last_scanned_bar_ts == -1:
                upcoming_core = news_state.get("upcoming_core_events", [])
                logger.info(
                    "US Macro Shield status: entry_allowed=%s | fail_safe=%s | cancel_limits=%s | protect_open=%s | "
                    "Upcoming core US events (FOMC/CPI/NFP/PPI): %d",
                    news_state.get("entry_allowed"),
                    news_state.get("fail_safe_active"),
                    news_state.get("cancel_pending_limits"),
                    news_state.get("protect_open_positions"),
                    len(upcoming_core),
                )
                for ev in upcoming_core[:10]:
                    logger.info("  [US MACRO CORE] %s | %s (forecast=%s, prev=%s)", ev["time_utc"], ev["title"], ev["forecast"], ev["previous"])

            # 2. Sync open & pending Maker positions (Macro Shield cancel/close, TTL, 2-Step BE, H1 Flip)
            executor.sync_positions()

            # 3. Scan strictly after each closed 15M candle (wait 4s for exchange bar finalization)
            now_ms = int(time.time() * 1000)
            latest_closed_open_ms = ((now_ms // entry_ms) - 1) * entry_ms
            ms_since_close = now_ms - (latest_closed_open_ms + entry_ms)

            should_scan = args.once or (latest_closed_open_ms > last_scanned_bar_ts and ms_since_close >= 4000)
            if should_scan and config_error is not None:
                logger.warning("Scan skipped: config %s is invalid (%s).", args.config, config_error)
                last_scanned_bar_ts = latest_closed_open_ms
                should_scan = False
            if should_scan:
                equity = executor.get_current_equity()
                bar_utc = datetime.fromtimestamp(latest_closed_open_ms / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                logger.info("Scanning closed %sM candle (%s) | Equity: $%.2f", entry_tf, bar_utc, equity)

                signals = scanner.scan_once(equity=equity)
                logger.info("Scan complete: %d qualifying %s signal(s).", len(signals), config.get("strategy"))

                if getattr(scanner, "last_scan_incomplete", False) and not args.once and ms_since_close < 120_000:
                    logger.info("Latest closed candle not yet finalized on Bybit; will retry within 2m window.")
                else:
                    last_scanned_bar_ts = latest_closed_open_ms
                for ev in getattr(scanner, "journal_events", []):
                    journal.record(**ev)
                for sig in signals:
                    if journal.entry_exists(sig.timestamp, sig.symbol):
                        logger.info("Skip %s %s: entry for this bar was already placed before restart.", sig.symbol, sig.direction)
                        continue
                    trade_id = None
                    outcome, reason = "REJECTED", ""
                    try:
                        trade_id = executor.open_position(sig, current_equity=equity)
                        if trade_id:
                            outcome = "OPENED"
                        else:
                            reason = executor.last_reject_reason or "unknown"
                    except Exception as exc:
                        outcome, reason = "ERROR", f"{type(exc).__name__}: {exc}"[:300]
                        logger.exception("Entry failed for %s %s: %s", sig.symbol, sig.direction, exc)
                        notifier.send(f"[ERROR] Не удалось выставить вход {sig.symbol} {sig.direction}: {exc}")
                    journal.record(
                        bar_ts_ms=sig.timestamp,
                        symbol=sig.symbol,
                        direction=sig.direction,
                        outcome=outcome,
                        reason=reason,
                        entry_price=sig.entry_price,
                        score=sig.score,
                        trade_id=trade_id,
                        details=sig.breakdown,
                    )
                    equity = executor.get_current_equity()

            if args.once:
                break

            time.sleep(max(5, args.poll_sec))

        except Exception as exc:
            logger.exception("Main loop error (resetting HTTP sessions and reconnecting): %s", exc)
            try:
                client.reset_sessions()
            except Exception:
                pass
            notifier.send(f"[ERROR] Bot loop exception: {exc}")
            if args.once:
                break
            time.sleep(15)

    logger.info("Bot stopped cleanly.")


if __name__ == "__main__":
    main()
