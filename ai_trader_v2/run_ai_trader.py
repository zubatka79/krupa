"""
Production Runner for VIKA_N2_N7 («ВИКА (N2+N7)») on Bybit USDT Linear Perpetuals.

Modes:
  --demo          Bybit Demo Trading (api-demo.bybit.com) — default from config_ai_trader.json
  --testnet       Bybit Testnet (api-testnet.bybit.com)
  --paper         Paper simulation (no real/demo orders placed; public mainnet data)
  --dry-run-check Verify live Bybit data, 10x Cross setup, indicators, filters & order sizing, then exit.

Note: Mainnet requires explicit config modification by the user.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.bybit_client import BybitClient, KLINE_COLUMNS, round_step_down, round_tick
from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.liveness import RunState, StallGuard, git_version, send_telegram_plain
from ai_trader_v2.telegram_ui import VikaTelegramUI
from ai_trader_v2.vika_engine import CoinEvaluation, M15_MS, evaluate_coin

logger = logging.getLogger("vika.runner")


def setup_logging(log_path: str) -> None:
    p = ROOT / log_path
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    root_log.handlers.clear()

    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(fmt)
    root_log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root_log.addHandler(sh)


_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "bybit-autonomous-engine/1.0"})
        _thread_local.session = sess
    return sess


def fetch_recent_klines(client: BybitClient, symbol: str, interval: str, limit: int = 300) -> pd.DataFrame:
    """Thread-safe single-request fetch of the latest `limit` klines (forming bar stripped downstream)."""
    url = f"{client.public_base_url}/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": str(interval), "limit": int(limit)}
    sess = _get_thread_session()
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = sess.get(url, params=params, timeout=(6, 12))
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("retCode") == 0:
                rows = payload.get("result", {}).get("list", [])
                if not rows:
                    return pd.DataFrame(columns=KLINE_COLUMNS)
                df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
                df["timestamp"] = df["timestamp"].astype("int64")
                for col in ("open", "high", "low", "close", "volume", "turnover"):
                    df[col] = df[col].astype("float64")
                return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            last_err = RuntimeError(f"Bybit kline retCode={payload.get('retCode')}: {payload.get('retMsg')}")
        except Exception as exc:
            last_err = exc
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Failed fetching klines {symbol} {interval}m: {last_err}")


class VikaRunner:
    def __init__(self, cfg: Dict[str, Any], dry_run_override: bool = False) -> None:
        self.cfg = cfg
        if dry_run_override:
            self.cfg["dry_run"] = True

        self.mode = str(cfg.get("mode", "demo")).lower()
        self.sync_interval = max(10, int(cfg.get("sync_interval_sec", 20)))
        self.filters_cfg = dict(cfg.get("filters", {}))
        self.filters_cfg.setdefault("allowed_directions", str(cfg.get("allowed_directions", "long_short")).strip().lower())

        self.client = BybitClient(
            mode=self.mode,
            cache_dir=ROOT / cfg.get("cache_dir", "data/ai_trader_v2/cache"),
            public_mainnet=True,
        )
        self.journal = VikaJournal(ROOT / cfg.get("db_path", "data/ai_trader_v2/vika_trades.db"))

        self._evals_lock = threading.Lock()
        self._latest_evals: Dict[str, CoinEvaluation] = {}
        self._stop_evt = threading.Event()
        self.version = "unknown"
        self.run_state: Optional[RunState] = None

        # Wire executor <-> telegram notifier
        self.tg: Optional[VikaTelegramUI] = None
        self.executor = VikaExecutor(
            cfg=cfg,
            client=self.client,
            journal=self.journal,
            notifier=self._notify,
        )
        self.tg = VikaTelegramUI(
            cfg=cfg,
            executor=self.executor,
            journal=self.journal,
            get_latest_evals=self.get_latest_evals,
        )

    def _notify(self, text: str) -> None:
        if self.tg is not None:
            self.tg.send_message(text)

    def get_latest_evals(self) -> Dict[str, CoinEvaluation]:
        with self._evals_lock:
            return dict(self._latest_evals)

    def scan_all_coins(self) -> Dict[str, CoinEvaluation]:
        """Parallel (~2-3s) fetch of closed 15m/1H/4H candles & live Funding Rates across all configured coins."""
        t0 = time.monotonic()
        now_ms = int(time.time() * 1000)
        tickers = self.client.get_tickers()
        evals: Dict[str, CoinEvaluation] = {}

        def _eval_one(coin: Dict[str, Any]) -> Tuple[str, Optional[CoinEvaluation]]:
            sym = coin["symbol"]
            key = coin["key"]
            try:
                df15 = fetch_recent_klines(self.client, sym, "15", limit=320)
                df1h = fetch_recent_klines(self.client, sym, "60", limit=240)
                df4h = fetch_recent_klines(self.client, sym, "240", limit=240)
                fr = float(tickers.get(sym, {}).get("funding_rate", 0.0))
                ev = evaluate_coin(
                    symbol=sym,
                    key=key,
                    df15_raw=df15,
                    df1h_raw=df1h,
                    df4h_raw=df4h,
                    funding_rate=fr,
                    now_ms=now_ms,
                    filters_cfg=self.filters_cfg,
                )
                return sym, ev
            except Exception as exc:
                logger.error("Failed to evaluate %s: %s", sym, exc)
                return sym, None

        max_workers = min(10, max(1, len(self.cfg["coins"])))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for sym, ev in pool.map(_eval_one, self.cfg["coins"]):
                if ev is not None:
                    evals[sym] = ev

        logger.info("Scanned %d/%d coins in %.2fs", len(evals), len(self.cfg["coins"]), time.monotonic() - t0)
        with self._evals_lock:
            self._latest_evals = evals
        return evals

    # ------------------------------------------------------------------
    # --dry-run-check Mode
    # ------------------------------------------------------------------
    def run_dry_run_check(self) -> int:
        logger.info("=== VIKA_N2_N7 DRY-RUN CHECK (mode=%s, dry_run=%s) ===", self.mode, self.executor.dry_run)
        n_coins = len(self.cfg["coins"])
        specs = self.executor.load_instruments()
        logger.info("Loaded %d/%d Bybit Linear USDT instruments.", len(specs), n_coins)

        lev_status = self.executor.ensure_cross_margin_and_leverage()
        for sym, st in lev_status.items():
            logger.info("  Leverage/Margin %-13s -> %s", sym, st)

        stale = self.executor.check_preexisting_positions()
        if stale:
            logger.warning("Pre-existing untracked items: %s", stale)
        else:
            logger.info("Pre-existing positions/orders check: CLEAN (0 untracked).")

        eq, avail = self.executor.refresh_balance()
        logger.info("Wallet balance: equity=$%.2f | available=$%.2f", eq, avail)

        evals = self.scan_all_coins()
        logger.info("Evaluated %d/%d coins on latest closed 15m candle:", len(evals), n_coins)
        print("\n" + "=" * 134)
        print(
            f"{'SYMBOL':<13} {'KEY':<17} {'CLOSE':>10} {'QTY(10%x10x)':>13} {'NOTION':>8} "
            f"{'ADX15':>6} {'BBW%':>6} {'ATR%':>6} {'FR_bps':>7} {'SLP4H%':>7} {'4H_DIV':<24} {'RAW':<6} {'FINAL':<6} {'BLOCK'}"
        )
        print("-" * 134)
        all_ok = len(evals) == n_coins
        margin_usd = eq * (float(self.cfg.get("margin_pct_per_trade", 10.0)) / 100.0)
        lev = int(self.cfg.get("leverage", 10))
        for coin in self.cfg["coins"]:
            sym = coin["symbol"]
            ev = evals.get(sym)
            if ev is None:
                print(f"{sym:<13} ERROR")
                all_ok = False
                continue
            spec = specs[sym]
            px = round_tick(ev.close_price, spec.tick_size)
            qty = round_step_down((margin_usd * lev) / px, spec.qty_step)
            notional = qty * px
            if qty < spec.min_order_qty or notional < max(spec.min_notional, 5.0):
                all_ok = False
            print(
                f"{sym:<13} {ev.key:<17} {px:>10.5g} {qty:>13.6g} {notional:>8.1f} "
                f"{ev.adx15:>6.1f} {ev.bbw15:>6.2f} {ev.atr15_pct:>6.2f} {ev.funding_bps:>+7.2f} {ev.slope4h_pct:>+6.2f}% "
                f"{ev.div4h_label:<24} {(ev.raw_signal or '-'):<6} {(ev.final_signal or '-'):<6} {ev.block_reason or '-'}"
            )
        print("=" * 134)

        ok_lev = sum(1 for v in lev_status.values() if v.startswith("OK") or v.startswith("DRY_RUN"))
        print(f"Summary: instruments={len(specs)}/{n_coins} | leverage_10x_cross={ok_lev}/{n_coins} | evals={len(evals)}/{n_coins} | equity=${eq:,.2f}")
        return 0 if (all_ok and ok_lev == n_coins) else 1

    # ------------------------------------------------------------------
    # Main Production Loop
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        logger.info(
            "Starting VIKA_N2_N7 runner (mode=%s, dry_run=%s, version=%s)", self.mode, self.executor.dry_run, self.version
        )

        self.executor.load_instruments()
        lev_status = self.executor.ensure_cross_margin_and_leverage()
        for sym, st in lev_status.items():
            logger.info("Margin/Leverage %-13s: %s", sym, st)

        cancelled_stale = self.executor.cleanup_stale_entry_orders()
        if cancelled_stale:
            logger.warning("Startup stale limit orders cancelled (>45m): %s", cancelled_stale)
        else:
            logger.info("Startup stale limit orders check (>45m): CLEAN (0 stale entry orders).")
        self.executor.check_preexisting_positions()
        self.executor.sync_with_exchange()

        # Initial snapshot of all coins before startup welcome message
        init_evals = self.scan_all_coins()
        logger.info("Initial scan complete: %d/%d coins evaluated.", len(init_evals), len(self.cfg["coins"]))

        if self.tg is not None:
            cmds_ok = self.tg.register_commands()
            logger.info("Telegram setMyCommands updated: %s", cmds_ok)
            welcome = self.tg.format_startup_welcome(lev_status) + f"\nВерсия: <code>{self.version}</code>"
            note = self.run_state.startup_note() if self.run_state else ""
            if note:
                welcome += f"\n{note}"
            sent = self.tg.send_message(welcome)
            logger.info("Telegram startup welcome sent: %s", sent)
            self.tg.start_polling()

        # Track the current 15m bucket so we scan immediately after each newly closed 15m candle
        last_scanned_bucket = int(time.time() * 1000) // M15_MS
        last_sync_ts = time.monotonic()

        logger.info(
            "Main loop active: 15m closed-candle scans (%d coins, parallel) + %ds exchange sync | "
            "dir=%s entry=%s tp=%s sl=Market | R3 max_same_key=%d | T4 slope4h>%.2f%% | K7_color=%s.",
            len(self.cfg["coins"]),
            self.sync_interval,
            self.executor.allowed_directions,
            self.executor.entry_order_type,
            self.executor.tp_order_type,
            self.executor.max_same_key,
            float(self.filters_cfg.get("t4_min_4h_ema50_slope_pct", 0.15)),
            bool(self.filters_cfg.get("k7_require_candle_dir", True)),
        )
        stall_sec = float(self.cfg.get("stall_restart_sec", 600))
        stall_guard = StallGuard(
            stall_sec,
            on_stall=lambda age: send_telegram_plain(
                f"🚨 Vika зависла: главный цикл стоит {age / 60:.0f} мин. Перезапускаю процесс (версия {self.version})."
            ),
        )
        stall_guard.start()

        while not self._stop_evt.is_set():
            stall_guard.beat()
            now_mono = time.monotonic()
            now_ms = int(time.time() * 1000)

            # 1. Exchange sync every sync_interval_sec (default 20s)
            if (now_mono - last_sync_ts) >= self.sync_interval:
                try:
                    self.executor.sync_with_exchange()
                except Exception as exc:
                    logger.error("Sync loop exception: %s", exc)
                last_sync_ts = time.monotonic()

            # 2. Closed 15m candle boundary check (+2.5s buffer for Bybit candle finalization)
            cur_bucket = now_ms // M15_MS
            ms_into_bar = now_ms % M15_MS
            if cur_bucket > last_scanned_bucket and ms_into_bar >= 2500:
                last_scanned_bucket = cur_bucket
                bar_dt = datetime.fromtimestamp((cur_bucket * M15_MS) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                logger.info("--- Closed 15m bar scan @ %s ---", bar_dt)
                try:
                    self.executor.sync_with_exchange()
                    evals = self.scan_all_coins()

                    # First check BE triggers on closed 15m bar for all open positions
                    for sym, ev in evals.items():
                        self.executor.check_breakeven_on_closed_bar(sym, ev.bar_ts, ev.close_price)

                    # Collect passing signals sorted by RVOL descending (matching backtest slot priority)
                    candidates: List[CoinEvaluation] = [
                        ev for ev in evals.values() if ev.final_signal in ("LONG", "SHORT")
                    ]
                    candidates.sort(key=lambda e: e.rvol, reverse=True)

                    blocked_log = [
                        f"{ev.symbol}({ev.raw_signal} blocked:{ev.block_reason})"
                        for ev in evals.values()
                        if ev.raw_signal is not None and ev.final_signal is None
                    ]
                    if blocked_log:
                        logger.info("Filtered raw signals: %s", ", ".join(blocked_log))

                    for ev in candidates:
                        try:
                            self.executor.open_signal(ev)
                        except Exception as exc:
                            logger.error("Failed opening %s %s: %s", ev.symbol, ev.final_signal, exc)
                except Exception as exc:
                    logger.error("15m bar scan error: %s", exc)
                    self.client.reset_sessions()

            self._stop_evt.wait(timeout=1.0)

        stall_guard.stop()
        if self.tg is not None:
            self.tg.stop()
        logger.info("VIKA_N2_N7 runner stopped cleanly.")

    def stop(self) -> None:
        self._stop_evt.set()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VIKA_N2_N7 Bybit Autonomous Trader")
    p.add_argument("--config", default="ai_trader_v2/config_ai_trader.json", help="Path to JSON config")
    mode_grp = p.add_mutually_exclusive_group()
    mode_grp.add_argument("--demo", action="store_true", help="Run on Bybit Demo Trading")
    mode_grp.add_argument("--testnet", action="store_true", help="Run on Bybit Testnet")
    mode_grp.add_argument("--paper", action="store_true", help="Run in offline/paper execution mode")
    p.add_argument("--dry-run-check", action="store_true", help="Run live verification check and exit")
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

    setup_logging(cfg.get("log_path", "logs/ai_trader_v2.log"))
    if args.dry_run_check:
        return VikaRunner(cfg).run_dry_run_check()

    version = git_version(ROOT)
    db_dir = (ROOT / cfg.get("db_path", "data/ai_trader_v2/vika_trades.db")).parent
    run_state = RunState(db_dir / "run_state.json")
    run_state.start(version)
    try:
        runner = VikaRunner(cfg)
    except Exception as exc:
        logger.exception("Runner init failed: %s", exc)
        if run_state.fatal_alert_allowed():
            send_telegram_plain(f"❌ Vika не запустилась (версия {version}): {type(exc).__name__}: {exc}"[:600])
        return 1
    runner.version = version
    runner.run_state = run_state

    def _sig_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %s, shutting down...", signum)
        runner.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        runner.run_forever()
    except Exception as exc:
        logger.exception("Runner crashed: %s", exc)
        if run_state.fatal_alert_allowed():
            send_telegram_plain(f"❌ Vika упала (версия {version}): {type(exc).__name__}: {exc}"[:600])
        return 1
    run_state.mark_clean_exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
