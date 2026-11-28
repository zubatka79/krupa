"""
Production Runner for L2_news Hybrid (All-Thick-18: 18 LONG + 18 SHORT on one Bybit Demo account).

Certified Step 17 Configuration:
  - 5x12% margin, 15x Isolated, cap $5,000 ($2,500 in VolStorm 1D ATR > 4%), Bybit reserve <= 85%
  - BTC 4H V1 + strictly closed previous-UTC-day 1D EMA20 & 1D ATR(14)
  - Live Retail L/S Ratio filter (|z| < 1.0) via Bybit V5 /v5/market/account-ratio?period=1h
  - AntiSalvo (<=2 LONG/bar, <=2 SHORT/bar, <=5 same direction, <=2 same key)
  - 2-Step Limit DCA (35/65 normal, 30/70 storm) + BE +0.30% (cancels pending DCA limit, never triggers AC/Probation)
  - US Macro Calendar blackout [T-60m .. T+45m] UTC from config/macro_events_utc.json

Modes:
  --demo          Bybit Demo Trading (api-demo.bybit.com) — default from config_ai_trader.json
  --testnet       Bybit Testnet (api-testnet.bybit.com)
  --paper         Paper simulation (no orders sent; live public mainnet data)
  --dry-run-check Verify live Bybit data, BTC 4H+1D regime, Retail L/S, macro calendar & 15x sizing, then exit.
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

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime, evaluate_btc_l2_regime, fetch_retail_ls_snapshot
from ai_trader_v2.liveness import RunState, StallGuard, git_version, send_telegram_plain
from ai_trader_v2.macro_calendar import MacroCalendarGuard, MacroCalendarStatus
from ai_trader_v2.telegram_ui import VikaTelegramUI
from ai_trader_v2.vika_engine import CoinEvaluation, M15_MS, evaluate_coin
from ai_trader_v2_long.long_engine import BtcRegimeState, LongCoinEvaluation, evaluate_long_coin
from bot.bybit_client import BybitClient, KLINE_COLUMNS, round_step_down, round_tick
import update_macro_calendar

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
        sess.headers.update({"User-Agent": "bybit-l2-news-engine/1.0"})
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
        self.storm_atr_thresh = float(cfg.get("storm_atr1d_threshold_pct", 4.0))

        self.long_coins: List[Dict[str, Any]] = list(cfg.get("long_coins", []))
        self.short_coins: List[Dict[str, Any]] = list(cfg.get("short_coins", cfg.get("coins", [])))

        cal_rel = str(cfg.get("macro_calendar_path", "config/macro_events_utc.json"))
        self.macro_cal_path = Path(cal_rel) if Path(cal_rel).is_absolute() else (ROOT / cal_rel)
        db_dir = (ROOT / cfg.get("db_path", "data/ai_trader_v2/vika_trades.db")).parent
        self.macro_runtime_path = db_dir / "macro_events_utc.json"
        self.macro_refresh_days = float(cfg.get("macro_refresh_interval_days", 7.0))
        self.macro_guard = MacroCalendarGuard(
            self.macro_cal_path,
            runtime_override_path=self.macro_runtime_path,
            pre_minutes=int(cfg.get("macro_pre_minutes", 60)),
            post_minutes=int(cfg.get("macro_post_minutes", 45)),
            horizon_warn_days=float(cfg.get("macro_horizon_warn_days", 3.0)),
        )
        self._macro_refresh_lock = threading.Lock()
        self._macro_refresh_running = False
        self._last_macro_attempt_mono = 0.0

        self.client = BybitClient(
            mode=self.mode,
            cache_dir=ROOT / cfg.get("cache_dir", "data/ai_trader_v2/cache"),
            public_mainnet=True,
        )
        self.journal = VikaJournal(ROOT / cfg.get("db_path", "data/ai_trader_v2/vika_trades.db"))

        self._evals_lock = threading.Lock()
        self._latest_long_evals: Dict[str, LongCoinEvaluation] = {}
        self._latest_short_evals: Dict[str, CoinEvaluation] = {}
        self._latest_btc_regime: Optional[BtcL2Regime] = None
        self._stop_evt = threading.Event()
        self.version = "unknown"
        self.run_state: Optional[RunState] = None

        self.tg: Optional[VikaTelegramUI] = None
        self.executor = VikaExecutor(
            cfg=cfg,
            client=self.client,
            journal=self.journal,
            notifier=self._notify,
            macro_guard=self.macro_guard,
        )
        self.tg = VikaTelegramUI(
            cfg=cfg,
            executor=self.executor,
            journal=self.journal,
            get_latest_evals=self.get_latest_evals,
            get_latest_long_evals=self.get_latest_long_evals,
            get_btc_regime=self.get_btc_regime,
            get_macro_status=self.get_macro_status,
        )

    def _notify(self, text: str) -> None:
        if self.tg is not None:
            self.tg.send_message(text)

    def get_latest_evals(self) -> Dict[str, CoinEvaluation]:
        with self._evals_lock:
            return dict(self._latest_short_evals)

    def get_latest_long_evals(self) -> Dict[str, LongCoinEvaluation]:
        with self._evals_lock:
            return dict(self._latest_long_evals)

    def get_btc_regime(self) -> Optional[BtcL2Regime]:
        with self._evals_lock:
            return self._latest_btc_regime

    def get_macro_status(self) -> MacroCalendarStatus:
        return self.macro_guard.evaluate()

    def _run_macro_refresh_worker(self) -> None:
        """Background weekly refresh of macro_events_utc.json via Claude API (zero disruption on failure)."""
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not api_key or "your_anthropic" in api_key.lower():
                logger.info("Macro calendar auto-refresh skipped: ANTHROPIC_API_KEY not configured in .env.")
                return
            model = (os.getenv("ANTHROPIC_MODEL", "") or "claude-sonnet-4-5").strip()
            now_utc = datetime.now(timezone.utc)
            existing: Optional[Dict[str, Any]] = None
            for fp in (self.macro_runtime_path, self.macro_cal_path):
                if fp.exists():
                    try:
                        existing = json.loads(fp.read_text(encoding="utf-8"))
                        break
                    except Exception:
                        continue
            raw_fresh = update_macro_calendar.fetch_events_from_claude(
                api_key=api_key, model=model, now_utc=now_utc, horizon_days=60,
            )
            fresh_valid = update_macro_calendar.validate_calendar_payload(raw_fresh, now_utc=now_utc, min_future_events=1)
            final_payload = update_macro_calendar.merge_and_prune_events(existing, fresh_valid, now_utc)

            # Persist in volume-backed runtime path (survives Docker redeploys) + update config/ if writable
            update_macro_calendar.atomic_write_json(self.macro_runtime_path, final_payload)
            try:
                update_macro_calendar.atomic_write_json(self.macro_cal_path, final_payload)
            except OSError as exc:
                logger.debug("Config macro path not writable (%s); runtime volume updated cleanly.", exc)

            self.journal.set_kv("macro_cal_last_refresh_ms", str(int(now_utc.timestamp() * 1000)))
            self.macro_guard.evaluate(now_utc)
            logger.info(
                "In-bot weekly macro calendar refresh succeeded: %d events (latest=%s).",
                len(final_payload["events"]), final_payload["events"][-1]["utc"],
            )
        except Exception as exc:
            logger.warning("In-bot macro calendar refresh failed (%s) — keeping existing calendar untouched.", exc)
        finally:
            with self._macro_refresh_lock:
                self._macro_refresh_running = False

    def maybe_refresh_macro_calendar(self, now_ms: Optional[int] = None) -> bool:
        """Trigger non-blocking weekly (7d) or low-horizon (<14d) macro calendar refresh inside the bot loop."""
        cur_ms = int(now_ms if now_ms is not None else time.time() * 1000)
        now_mono = time.monotonic()
        # Rate-limit failed/retry attempts to at most once every 12h
        if self._last_macro_attempt_mono > 0 and (now_mono - self._last_macro_attempt_mono) < 12 * 3600:
            return False

        last_ok_ms = int(self.journal.get_kv("macro_cal_last_refresh_ms", "0") or "0")
        due_weekly = (cur_ms - last_ok_ms) >= int(self.macro_refresh_days * 86_400_000)
        st = self.macro_guard.evaluate()
        due_horizon = st.horizon_days < 14.0
        if not (due_weekly or due_horizon):
            return False

        with self._macro_refresh_lock:
            if self._macro_refresh_running:
                return False
            self._macro_refresh_running = True
            self._last_macro_attempt_mono = now_mono

        t = threading.Thread(target=self._run_macro_refresh_worker, name="macro-cal-refresh", daemon=True)
        t.start()
        return True

    def scan_all_coins(self) -> Tuple[BtcL2Regime, Dict[str, LongCoinEvaluation], Dict[str, CoinEvaluation]]:
        """
        Parallel fetch of BTC 4H+1D regime, closed 15m/1H/4H candles, Funding Rates & Retail L/S Ratio
        across the entire All-Thick-18 pool (18 LONG + 18 SHORT, deduplicating OHLCV fetches per symbol).
        """
        t0 = time.monotonic()
        now_ms = int(time.time() * 1000)

        # 1. BTC 4H V1 + strictly closed previous-UTC-day 1D EMA20 & 1D ATR(14)
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
        long_map = {c["symbol"]: c["key"] for c in self.long_coins}
        short_map = {c["symbol"]: c["key"] for c in self.short_coins}
        unique_syms = sorted(set(long_map.keys()) | set(short_map.keys()))

        long_filters = {
            "btc_4h_up_switch": bool(self.filters_cfg.get("btc_4h_v1_switch", True)),
            "btc_1d_ema20_lock": bool(self.filters_cfg.get("btc_1d_ema20_lock", True)),
            "retail_ls_enabled": ls_enabled,
            "retail_ls_max_abs_z": float(self.filters_cfg.get("retail_ls_max_abs_z", 1.0)),
            "skip_utc_hours": self.filters_cfg.get("skip_utc_hours_long", [8, 9, 10, 11]),
            "max_coin_dist4h_ema50_pct": float(self.filters_cfg.get("max_coin_dist4h_ema50_pct", 10.0)),
            "min_atr15_pct": float(self.filters_cfg.get("min_atr15_pct_long", 0.10)),
            "max_atr15_pct": float(self.filters_cfg.get("max_atr15_pct_long", 1.80)),
        }
        short_filters = dict(self.filters_cfg)
        short_filters["allowed_directions"] = "only_short"

        def _eval_symbol(sym: str) -> Tuple[str, Optional[LongCoinEvaluation], Optional[CoinEvaluation]]:
            try:
                df15 = fetch_recent_klines(self.client, sym, "15", limit=320)
                df1h = fetch_recent_klines(self.client, sym, "60", limit=240)
                df4h = fetch_recent_klines(self.client, sym, "240", limit=240)
                fr = float(tickers.get(sym, {}).get("funding_rate", 0.0))
                ls_snap = fetch_retail_ls_snapshot(self.client, sym, now_ms) if ls_enabled else None
                ls_z = ls_snap.z_score if (ls_snap is not None and ls_snap.available) else None

                ev_l: Optional[LongCoinEvaluation] = None
                ev_s: Optional[CoinEvaluation] = None
                if sym in long_map:
                    ev_l = evaluate_long_coin(
                        symbol=sym,
                        key=long_map[sym],
                        df15_raw=df15,
                        df1h_raw=df1h,
                        df4h_raw=df4h,
                        btc_regime=btc_4h_compat,
                        funding_rate=fr,
                        now_ms=now_ms,
                        filters_cfg=long_filters,
                        retail_ls_z=ls_z,
                        btc_1d_up=(btc_regime.d1_dir == 1),
                    )
                if sym in short_map:
                    ev_s = evaluate_coin(
                        symbol=sym,
                        key=short_map[sym],
                        df15_raw=df15,
                        df1h_raw=df1h,
                        df4h_raw=df4h,
                        funding_rate=fr,
                        now_ms=now_ms,
                        filters_cfg=short_filters,
                        retail_ls_z=ls_z,
                        btc_allow_short=btc_regime.allow_short,
                        btc_allow_long=False,
                    )
                return sym, ev_l, ev_s
            except Exception as exc:
                logger.error("Failed evaluating %s: %s", sym, exc)
                return sym, None, None

        long_evals: Dict[str, LongCoinEvaluation] = {}
        short_evals: Dict[str, CoinEvaluation] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            for sym, ev_l, ev_s in pool.map(_eval_symbol, unique_syms):
                if ev_l is not None:
                    long_evals[sym] = ev_l
                if ev_s is not None:
                    short_evals[sym] = ev_s

        logger.info(
            "Scanned Vika Hybrid pool (%dL / %dS, %d unique) in %.2fs | %s",
            len(long_evals), len(short_evals), len(unique_syms), time.monotonic() - t0, btc_regime.mode_label,
        )
        with self._evals_lock:
            self._latest_btc_regime = btc_regime
            self._latest_long_evals = long_evals
            self._latest_short_evals = short_evals
        return btc_regime, long_evals, short_evals

    # ------------------------------------------------------------------
    # --dry-run-check Mode
    # ------------------------------------------------------------------
    def run_dry_run_check(self) -> int:
        logger.info("=== VIKA HYBRID DRY-RUN CHECK (mode=%s, dry_run=%s) ===", self.mode, self.executor.dry_run)
        n_total = len(self.executor.coin_cfg_map)
        specs = self.executor.load_instruments()
        lev_status = self.executor.ensure_margin_and_leverage()
        eq, avail = self.executor.refresh_balance()
        macro_st = self.executor.check_macro_calendar(notify_events=False)
        btc_regime, long_evals, short_evals = self.scan_all_coins()

        print("\n" + "=" * 124)
        print(
            f"VIKA HYBRID | {btc_regime.mode_label} | Wallet=${eq:,.2f} (avail=${avail:,.2f}) | "
            f"15x Isolated 5x12% Cap=${self.executor.max_margin_usd:,.0f}/${self.executor.storm_max_margin_usd:,.0f}"
        )
        print(f"Macro Calendar: {macro_st.short_label if macro_st else 'disabled'} (horizon={macro_st.horizon_days if macro_st else 0:.1f}d)")
        print("-" * 124)
        print(f"{'SIDE':<6} {'SYMBOL':<13} {'KEY':<20} {'CLOSE':>10} {'LEG1_QTY':>12} {'LS_Z':>7} {'RAW':<6} {'FINAL':<6} {'BLOCK'}")
        print("-" * 124)

        all_ok = (len(long_evals) == len(self.long_coins)) and (len(short_evals) == len(self.short_coins))
        eff_cap = self.executor.storm_max_margin_usd if btc_regime.in_storm else self.executor.max_margin_usd
        s_mult = self.executor.storm_size_mult if btc_regime.in_storm else 1.0
        leg1_frac = self.executor.storm_leg1_pct if btc_regime.in_storm else self.executor.dca_leg1_pct
        m1 = min(eq * self.executor.margin_pct * s_mult, eff_cap) * leg1_frac

        for side_lbl, clist, evmap in (("LONG", self.long_coins, long_evals), ("SHORT", self.short_coins, short_evals)):
            for c in clist:
                sym = c["symbol"]
                ev = evmap.get(sym)
                if ev is None:
                    print(f"{side_lbl:<6} {sym:<13} ERROR")
                    all_ok = False
                    continue
                spec = specs[sym]
                px = round_tick(float(ev.close_price), spec.tick_size)
                qty1 = round_step_down((m1 * self.executor.leverage) / px, spec.qty_step)
                ls_s = f"{ev.retail_ls_z:+.2f}" if ev.retail_ls_available else "n/a"
                print(
                    f"{side_lbl:<6} {sym:<13} {ev.key:<20} {px:>10.5g} {qty1:>12.6g} {ls_s:>7} "
                    f"{(ev.raw_signal or '-'):<6} {(ev.final_signal or '-'):<6} {ev.block_reason or '-'}"
                )
        print("=" * 124)
        ok_lev = sum(1 for v in lev_status.values() if v.startswith("OK") or v.startswith("DRY_RUN"))
        print(
            f"Summary: instruments={len(specs)}/{n_total} | lev_15x={ok_lev}/{n_total} | "
            f"LONG={len(long_evals)}/{len(self.long_coins)} | SHORT={len(short_evals)}/{len(self.short_coins)}"
        )
        return 0 if (all_ok and ok_lev == n_total) else 1

    # ------------------------------------------------------------------
    # Main Production Loop
    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        logger.info("Starting Vika Hybrid runner (mode=%s, dry_run=%s, version=%s)", self.mode, self.executor.dry_run, self.version)
        self.executor.load_instruments()
        lev_status = self.executor.ensure_margin_and_leverage()
        self.executor.cleanup_stale_entry_orders()
        self.executor.check_preexisting_positions()
        self.executor.sync_with_exchange()

        btc_regime, l_evals, s_evals = self.scan_all_coins()
        self.executor.check_macro_calendar(notify_events=False)
        logger.info("Initial Vika Hybrid scan complete: %dL/%dS | %s", len(l_evals), len(s_evals), btc_regime.mode_label)

        if self.tg is not None:
            self.tg.register_commands()
            welcome = self.tg.format_startup_welcome(lev_status) + f"\nВерсия: <code>{self.version}</code>"
            note = self.run_state.startup_note() if self.run_state else ""
            if note:
                welcome += f"\n{note}"
            self.tg.send_message(welcome)
            self.tg.start_polling()

        last_scanned_bucket = int(time.time() * 1000) // M15_MS
        last_sync_ts = time.monotonic()

        stall_guard = StallGuard(
            float(self.cfg.get("stall_restart_sec", 600)),
            on_stall=lambda age: send_telegram_plain(
                f"🚨 Vika Hybrid завис: цикл стоит {age / 60:.0f} мин. Перезапускаю (версия {self.version})."
            ),
        )
        stall_guard.start()

        while not self._stop_evt.is_set():
            stall_guard.beat()
            now_mono = time.monotonic()
            now_ms = int(time.time() * 1000)

            if (now_mono - last_sync_ts) >= self.sync_interval:
                try:
                    self.executor.sync_with_exchange()
                    self.executor.check_macro_calendar(notify_events=True)
                    self.maybe_refresh_macro_calendar(now_ms)
                except Exception as exc:
                    logger.error("Sync loop exception: %s", exc)
                last_sync_ts = time.monotonic()

            cur_bucket = now_ms // M15_MS
            ms_into_bar = now_ms % M15_MS
            if cur_bucket > last_scanned_bucket and ms_into_bar >= 2500:
                last_scanned_bucket = cur_bucket
                bar_dt = datetime.fromtimestamp((cur_bucket * M15_MS) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                logger.info("--- Closed 15m Vika Hybrid scan @ %s ---", bar_dt)
                try:
                    self.executor.sync_with_exchange()
                    btc_regime, long_evals, short_evals = self.scan_all_coins()

                    # 1. Check BE (+1.00% -> +0.30% + synchronous Leg-2 DCA cancel) on ALL open positions
                    price_by_sym: Dict[str, Tuple[int, float]] = {}
                    for sym, ev in short_evals.items():
                        price_by_sym[sym] = (ev.bar_ts, float(ev.close_price))
                    for sym, ev in long_evals.items():
                        price_by_sym.setdefault(sym, (ev.bar_ts, float(ev.close_price)))
                    for sym, (bts, cpx) in price_by_sym.items():
                        self.executor.check_breakeven_on_closed_bar(sym, bts, cpx)

                    # 2. Rank passing LONG + SHORT candidates by RVOL descending
                    candidates: List[Any] = []
                    for ev in long_evals.values():
                        if ev.final_signal == "LONG":
                            candidates.append(ev)
                    for ev in short_evals.values():
                        if ev.final_signal == "SHORT":
                            candidates.append(ev)
                    candidates.sort(key=lambda e: float(e.rvol), reverse=True)

                    for ev in candidates:
                        try:
                            self.executor.open_signal(ev, btc_regime=btc_regime)
                        except Exception as exc:
                            logger.error("Failed opening %s %s: %s", ev.symbol, ev.final_signal, exc)
                except Exception as exc:
                    logger.error("15m Vika Hybrid scan error: %s", exc)
                    self.client.reset_sessions()

            self._stop_evt.wait(timeout=1.0)

        stall_guard.stop()
        if self.tg is not None:
            self.tg.stop()
        logger.info("Vika Hybrid runner stopped cleanly.")

    def stop(self) -> None:
        self._stop_evt.set()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vika Hybrid Bybit Autonomous Trader (15x, 5x12%, All-Thick-18)")
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
        return VikaRunner(cfg, dry_run_override=True).run_dry_run_check()

    version = git_version(ROOT)
    db_dir = (ROOT / cfg.get("db_path", "data/ai_trader_v2/vika_trades.db")).parent
    run_state = RunState(db_dir / "run_state.json")
    run_state.start(version)

    runner = VikaRunner(cfg)
    runner.version = version
    runner.run_state = run_state

    def _sig_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %s, stopping Vika Hybrid...", signum)
        runner.stop()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    runner.run_forever()
    run_state.mark_clean_exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
