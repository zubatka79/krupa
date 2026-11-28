"""
Standalone LONG Runner for L2_NEWS_LONG (All-Thick-18 LONG, 15x Isolated, 5x12%, DCA + BE +0.30%,
BTC 4H + closed 1D EMA20/ATR, Retail L/S Ratio, US Macro Blackout).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime, evaluate_btc_l2_regime, fetch_retail_ls_snapshot
from ai_trader_v2.liveness import RunState, StallGuard, git_version
from ai_trader_v2.macro_calendar import MacroCalendarGuard, MacroCalendarStatus
from ai_trader_v2.run_ai_trader import fetch_recent_klines, setup_logging
from ai_trader_v2_long.long_engine import BtcRegimeState, LongCoinEvaluation, M15_MS, evaluate_long_coin
from ai_trader_v2_long.long_executor import VikaLongExecutor
from ai_trader_v2_long.long_telegram_ui import VikaLongTelegramUI
from bot.bybit_client import BybitClient, round_step_down, round_tick

logger = logging.getLogger("vika_long.runner")


def _resolve_long_api_keys(mode: str, dry_run: bool, allow_shared: bool) -> Tuple[str, str]:
    if dry_run or mode == "paper":
        return "", ""
    if mode == "demo":
        k = os.getenv("LONG_BYBIT_DEMO_API_KEY", "").strip() or os.getenv("LONG_BYBIT_API_KEY", "").strip()
        s = os.getenv("LONG_BYBIT_DEMO_API_SECRET", "").strip() or os.getenv("LONG_BYBIT_API_SECRET", "").strip()
        if k and s:
            return k, s
        if allow_shared:
            return (
                os.getenv("BYBIT_DEMO_API_KEY", "").strip() or os.getenv("BYBIT_API_KEY", "").strip(),
                os.getenv("BYBIT_DEMO_API_SECRET", "").strip() or os.getenv("BYBIT_API_SECRET", "").strip(),
            )
        return "", ""
    k = os.getenv("LONG_BYBIT_API_KEY", "").strip()
    s = os.getenv("LONG_BYBIT_API_SECRET", "").strip()
    if k and s:
        return k, s
    if allow_shared:
        return os.getenv("BYBIT_API_KEY", "").strip(), os.getenv("BYBIT_API_SECRET", "").strip()
    return "", ""


class VikaLongRunner:
    def __init__(self, cfg: Dict[str, Any], dry_run_override: bool = False, allow_shared_account: bool = False) -> None:
        self.cfg = cfg
        if dry_run_override:
            self.cfg["dry_run"] = True

        self.mode = str(cfg.get("mode", "demo")).lower()
        self.sync_interval = max(10, int(cfg.get("sync_interval_sec", 20)))
        self.filters_cfg = dict(cfg.get("filters", {}))
        self.storm_atr_thresh = float(cfg.get("storm_atr1d_threshold_pct", 4.0))
        self.dry_run = bool(cfg.get("dry_run", False)) or (self.mode == "paper")

        api_key, api_secret = _resolve_long_api_keys(self.mode, self.dry_run, allow_shared_account)
        if (not self.dry_run) and (not api_key or not api_secret):
            logger.warning("Dedicated LONG_BYBIT_DEMO_API_KEY/SECRET missing — failing closed to DRY-RUN.")
            self.cfg["dry_run"] = True
            self.dry_run = True

        cal_rel = str(cfg.get("macro_calendar_path", "config/macro_events_utc.json"))
        cal_path = Path(cal_rel) if Path(cal_rel).is_absolute() else (ROOT / cal_rel)
        self.macro_guard = MacroCalendarGuard(
            cal_path,
            pre_minutes=int(cfg.get("macro_pre_minutes", 60)),
            post_minutes=int(cfg.get("macro_post_minutes", 45)),
            horizon_warn_days=float(cfg.get("macro_horizon_warn_days", 3.0)),
        )

        self.client = BybitClient(
            mode=self.mode,
            cache_dir=ROOT / cfg.get("cache_dir", "data/ai_trader_v2_long/cache"),
            public_mainnet=True,
            api_key=api_key,
            api_secret=api_secret,
        )
        self.journal = VikaJournal(ROOT / cfg.get("db_path", "data/ai_trader_v2_long/vika_long_trades.db"))

        self._lock = threading.Lock()
        self._latest_evals: Dict[str, LongCoinEvaluation] = {}
        self._latest_btc: Optional[BtcL2Regime] = None
        self._stop_evt = threading.Event()
        self.version = "unknown"
        self.run_state: Optional[RunState] = None

        self.tg: Optional[VikaLongTelegramUI] = None
        self.executor = VikaLongExecutor(
            cfg=self.cfg,
            client=self.client,
            journal=self.journal,
            notifier=self._notify,
            macro_guard=self.macro_guard,
        )
        self.tg = VikaLongTelegramUI(
            cfg=self.cfg,
            executor=self.executor,
            journal=self.journal,
            get_latest_evals=self.get_latest_evals,
            get_btc_regime=self.get_btc_regime,
            get_macro_status=self.get_macro_status,
        )

    def _notify(self, text: str) -> None:
        if self.tg is not None:
            self.tg.send_message(text)

    def get_latest_evals(self) -> Dict[str, LongCoinEvaluation]:
        with self._lock:
            return dict(self._latest_evals)

    def get_btc_regime(self) -> Optional[BtcL2Regime]:
        with self._lock:
            return self._latest_btc

    def get_macro_status(self) -> MacroCalendarStatus:
        return self.macro_guard.evaluate()

    def scan_all_coins(self) -> Tuple[BtcL2Regime, Dict[str, LongCoinEvaluation]]:
        t0 = time.monotonic()
        now_ms = int(time.time() * 1000)

        df_btc_4h = fetch_recent_klines(self.client, "BTCUSDT", "240", limit=240)
        btc_regime = evaluate_btc_l2_regime(df_btc_4h, now_ms, storm_atr_pct=self.storm_atr_thresh)
        self.executor.last_btc_regime = btc_regime
        btc_4h_compat = BtcRegimeState(
            bar_ts_4h=btc_regime.bar_ts_4h,
            close_4h=btc_regime.close_4h,
            ema20_4h=btc_regime.ema20_4h,
            ema50_4h=btc_regime.ema50_4h,
            is_4h_up=btc_regime.v1_long,
            dist_ema50_pct=btc_regime.dist_ema50_4h_pct,
        )

        tickers = self.client.get_tickers()
        ls_enabled = bool(self.filters_cfg.get("retail_ls_enabled", True))
        evals: Dict[str, LongCoinEvaluation] = {}

        def _eval_one(coin: Dict[str, Any]) -> Tuple[str, Optional[LongCoinEvaluation]]:
            sym = coin["symbol"]
            key = coin["key"]
            try:
                df15 = fetch_recent_klines(self.client, sym, "15", limit=240)
                df1h = fetch_recent_klines(self.client, sym, "60", limit=160)
                df4h = fetch_recent_klines(self.client, sym, "240", limit=120)
                fr = float(tickers.get(sym, {}).get("funding_rate", 0.0))
                ls_snap = fetch_retail_ls_snapshot(self.client, sym, now_ms) if ls_enabled else None
                ls_z = ls_snap.z_score if (ls_snap is not None and ls_snap.available) else None
                ev = evaluate_long_coin(
                    symbol=sym,
                    key=key,
                    df15_raw=df15,
                    df1h_raw=df1h,
                    df4h_raw=df4h,
                    btc_regime=btc_4h_compat,
                    funding_rate=fr,
                    now_ms=now_ms,
                    filters_cfg=self.filters_cfg,
                    retail_ls_z=ls_z,
                    btc_1d_up=(btc_regime.d1_dir == 1),
                )
                return sym, ev
            except Exception as exc:
                logger.error("Failed to evaluate LONG %s: %s", sym, exc)
                return sym, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, max(1, len(self.cfg["coins"])))) as pool:
            for sym, ev in pool.map(_eval_one, self.cfg["coins"]):
                if ev is not None:
                    evals[sym] = ev

        logger.info("Scanned %d/%d Vika Hybrid LONG coins in %.2fs | %s", len(evals), len(self.cfg["coins"]), time.monotonic() - t0, btc_regime.mode_label)
        with self._lock:
            self._latest_btc = btc_regime
            self._latest_evals = evals
        return btc_regime, evals

    def run_dry_run_check(self) -> int:
        n_coins = len(self.cfg["coins"])
        specs = self.executor.load_instruments()
        lev_status = self.executor.ensure_margin_and_leverage()
        eq, avail = self.executor.refresh_balance()
        btc_regime, evals = self.scan_all_coins()
        all_ok = len(evals) == n_coins
        ok_lev = sum(1 for v in lev_status.values() if v.startswith("OK") or v.startswith("DRY_RUN"))
        print(f"Vika Hybrid LONG | {btc_regime.mode_label} | Wallet=${eq:,.2f} (avail=${avail:,.2f}) | specs={len(specs)}/{n_coins} lev15x={ok_lev}/{n_coins}")
        return 0 if (all_ok and ok_lev == n_coins) else 1

    def run_forever(self) -> None:
        logger.info("Starting Vika Hybrid LONG runner (mode=%s, dry_run=%s, version=%s)", self.mode, self.executor.dry_run, self.version)
        self.executor.load_instruments()
        lev_status = self.executor.ensure_margin_and_leverage()
        self.executor.cleanup_stale_entry_orders()
        self.executor.check_preexisting_positions()
        self.executor.sync_with_exchange()

        btc_regime, _ = self.scan_all_coins()
        self.executor.check_macro_calendar(notify_events=False)

        if self.tg is not None and self.tg.enabled:
            self.tg.register_commands()
            welcome = self.tg.format_startup_welcome(lev_status) + f"\nВерсия: <code>{self.version}</code>"
            self.tg.send_message(welcome)
            self.tg.start_polling()

        last_scanned_bucket = int(time.time() * 1000) // M15_MS
        last_sync_ts = time.monotonic()
        stall_guard = StallGuard(float(self.cfg.get("stall_restart_sec", 600)))
        stall_guard.start()

        while not self._stop_evt.is_set():
            stall_guard.beat()
            now_mono = time.monotonic()
            now_ms = int(time.time() * 1000)

            if (now_mono - last_sync_ts) >= self.sync_interval:
                try:
                    self.executor.sync_with_exchange()
                    self.executor.check_macro_calendar(notify_events=True)
                except Exception as exc:
                    logger.error("LONG sync exception: %s", exc)
                last_sync_ts = time.monotonic()

            cur_bucket = now_ms // M15_MS
            ms_into_bar = now_ms % M15_MS
            if cur_bucket > last_scanned_bucket and ms_into_bar >= 2500:
                last_scanned_bucket = cur_bucket
                try:
                    self.executor.sync_with_exchange()
                    btc_regime, evals = self.scan_all_coins()
                    for sym, ev in evals.items():
                        self.executor.check_breakeven_on_closed_bar(sym, ev.bar_ts, ev.close_price)
                    candidates = [ev for ev in evals.values() if ev.final_signal == "LONG"]
                    candidates.sort(key=lambda e: e.rvol, reverse=True)
                    for ev in candidates:
                        try:
                            self.executor.open_signal(ev, btc_regime=btc_regime)
                        except Exception as exc:
                            logger.error("Failed opening LONG %s: %s", ev.symbol, exc)
                except Exception as exc:
                    logger.error("LONG 15m scan error: %s", exc)
                    self.client.reset_sessions()

            self._stop_evt.wait(timeout=1.0)

        stall_guard.stop()
        if self.tg is not None:
            self.tg.stop()

    def stop(self) -> None:
        self._stop_evt.set()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="L2_NEWS_LONG Autonomous Trader")
    p.add_argument("--config", default="ai_trader_v2_long/config_long_trader.json", help="Path to LONG JSON config")
    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument("--demo", action="store_true")
    mode_grp.add_argument("--testnet", action="store_true")
    mode_grp.add_argument("--paper", action="store_true")
    p.add_argument("--dry-run-check", action="store_true")
    p.add_argument("--allow-shared-account", action="store_true")
    return p.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.demo:
        cfg["mode"] = "demo"
        cfg["dry_run"] = False
    elif args.testnet:
        cfg["mode"] = "testnet"
        cfg["dry_run"] = False
    elif args.paper:
        cfg["mode"] = "paper"
        cfg["dry_run"] = True

    setup_logging(cfg.get("log_path", "logs/ai_trader_v2_long.log"))
    if args.dry_run_check:
        return VikaLongRunner(cfg, dry_run_override=True, allow_shared_account=args.allow_shared_account).run_dry_run_check()

    version = git_version(ROOT)
    db_dir = (ROOT / cfg.get("db_path", "data/ai_trader_v2_long/vika_long_trades.db")).parent
    run_state = RunState(db_dir / "run_state.json")
    run_state.start(version)

    runner = VikaLongRunner(cfg, allow_shared_account=args.allow_shared_account)
    runner.version = version
    runner.run_state = run_state

    def _sig_handler(signum: int, frame: Any) -> None:
        runner.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
    runner.run_forever()
    run_state.mark_clean_exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
