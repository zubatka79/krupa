"""Main autonomous trading engine for Bybit USDT Perpetuals (`fibo_trend_v2` + Two-Layer NewsGuard)."""

from __future__ import annotations

import argparse
import json
import logging
import os
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


class TelegramNotifier:
    """Lightweight Telegram Bot API notifier (silently disabled if env vars are unset)."""

    def __init__(self) -> None:
        self._reload()

    def _reload(self) -> None:
        load_dotenv(override=True)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(
            self.token
            and self.chat_id
            and "your_telegram" not in self.token.lower()
            and "your_telegram" not in self.chat_id.lower()
        )

    def send(self, text: str) -> None:
        if not self.enabled:
            self._reload()
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
                timeout=8,
            )
        except Exception as exc:
            logger.warning("Telegram notification failed: %s", exc)


def load_config(path: str | Path = "config.json") -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
            h.close()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger.addHandler(fh)

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

    while running:
        try:
            # Hot-reload config.json without restarting
            config = load_config(args.config)
            scanner.update_config(config)
            executor.update_config(config)
            news_guard.config = config

            # 1. Fail-CLOSED US Macro Shield update (auto-refreshes data/macro_events.json on schedule)
            news_state = news_guard.update(force_refresh=(args.once or last_scanned_bar_ts == -1))
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
            if should_scan:
                equity = executor.get_current_equity()
                bar_utc = datetime.fromtimestamp(latest_closed_open_ms / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
                logger.info("Scanning closed %sM candle (%s) | Equity: $%.2f", entry_tf, bar_utc, equity)

                signals = scanner.scan_once(equity=equity)
                logger.info("Scan complete: %d qualifying %s signal(s).", len(signals), config.get("strategy"))

                for sig in signals:
                    executor.open_position(sig, current_equity=equity)
                    equity = executor.get_current_equity()
                last_scanned_bar_ts = latest_closed_open_ms

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
