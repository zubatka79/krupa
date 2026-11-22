"""Grid Backtester & 10,000-Run Monte Carlo evaluator for Variant 1 (`trend_pullback_v2`)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.bybit_client import BybitClient, INTERVAL_TO_MS, InstrumentSpec
from bot.scanner import load_blacklist
from bot.strategy_v2 import build_v2_position_plan, enrich_v2_dataframe
from bot.time_machine import _simulate_paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest_v2")


@dataclass
class GridParams:
    tp_pct: float
    sl_pct: float
    entry_type: str  # "maker" or "taker"
    smart_be: bool
    maker_pullback_pct: float = 0.075  # midpoint of 0.05%-0.10%
    maker_ttl_bars: int = 2
    be_trigger_pct: float = 0.85
    be_offset_pct: float = 0.15

    @property
    def label(self) -> str:
        be_str = "BE(0.85->0.15)" if self.smart_be else "NoBE"
        return f"TP={self.tp_pct:.2f}% SL={self.sl_pct:.1f}% {self.entry_type.upper()} {be_str}"


@dataclass
class PendingOrder:
    symbol: str
    direction: str
    limit_price: float
    placed_idx: int
    expires_idx: int
    quality: float


@dataclass
class OpenPosV2:
    symbol: str
    direction: str
    period: str  # "IS" or "OOS"
    entry_ts: int
    entry_idx: int
    entry_price: float
    sl_price: float
    tp_price: float
    be_trigger_price: Optional[float]
    be_stop_price: Optional[float]
    qty: float
    leverage: int
    notional_usdt: float
    equity_at_entry: float
    entry_fee_usdt: float
    entry_is_maker: bool
    be_active: bool = False


@dataclass
class ClosedTradeV2:
    symbol: str
    direction: str
    period: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    leverage: int
    exit_reason: str  # "TP", "BE", "SL", "EOD"
    gross_pnl_usdt: float
    fees_usdt: float
    net_pnl_usdt: float
    equity_return_pct: float
    fees_pct_of_equity: float


def compute_metrics_v2(trades: List[ClosedTradeV2], days: int, n_sims: int, seed: int) -> Dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {
            "days": days,
            "trades": 0,
            "trades_per_day": 0.0,
            "win_rate_pct": 0.0,
            "tp_count": 0,
            "be_count": 0,
            "sl_count": 0,
            "fees_pct": 0.0,
            "net_pnl_pct": 0.0,
            "gross_pnl_pct": 0.0,
            "profit_factor": 0.0,
            "max_dd_pct": 0.0,
            "mc_p95_dd_pct": 0.0,
            "mc_prob_profit_pct": 0.0,
        }

    tp_c = sum(1 for t in trades if t.exit_reason == "TP")
    be_c = sum(1 for t in trades if t.exit_reason == "BE")
    sl_c = sum(1 for t in trades if t.exit_reason == "SL")
    wins = sum(1 for t in trades if t.net_pnl_usdt > 0)

    eq = 100.0
    peak = 100.0
    max_dd = 0.0
    g_win = 0.0
    g_loss = 0.0
    total_fees_pct = 0.0
    total_gross_pct = 0.0

    ret_arr = np.empty(n, dtype=np.float64)
    for i, t in enumerate(trades):
        r = t.equity_return_pct
        ret_arr[i] = r
        eq *= 1.0 + r / 100.0
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
        if r > 0:
            g_win += r
        elif r < 0:
            g_loss += abs(r)
        total_fees_pct += t.fees_pct_of_equity
        total_gross_pct += r + t.fees_pct_of_equity

    pf = (g_win / g_loss) if g_loss > 0 else (99.99 if g_win > 0 else 0.0)

    rng = np.random.default_rng(seed)
    mc = _simulate_paths(ret_arr, n_sims=n_sims, rng=rng)

    return {
        "days": days,
        "trades": n,
        "trades_per_day": round(n / max(days, 1), 2),
        "win_rate_pct": round((wins / n) * 100.0, 2),
        "tp_count": tp_c,
        "be_count": be_c,
        "sl_count": sl_c,
        "fees_pct": round(total_fees_pct, 2),
        "gross_pnl_pct": round(total_gross_pct, 2),
        "net_pnl_pct": round(eq - 100.0, 2),
        "profit_factor": round(pf, 2),
        "max_dd_pct": round(max_dd, 2),
        "mc_p95_dd_pct": mc["p95_max_dd_pct"],
        "mc_prob_profit_pct": mc["prob_profit_pct"],
    }


def compute_coin_stats_v2(trades: List[ClosedTradeV2], min_trades: int = 15) -> Dict[str, Any]:
    by_sym: Dict[str, List[ClosedTradeV2]] = {}
    for t in trades:
        by_sym.setdefault(t.symbol, []).append(t)

    rows: List[Dict[str, Any]] = []
    for sym, c_trades in by_sym.items():
        n = len(c_trades)
        wins = sum(1 for t in c_trades if t.net_pnl_usdt > 0)
        tp_c = sum(1 for t in c_trades if t.exit_reason == "TP")
        be_c = sum(1 for t in c_trades if t.exit_reason == "BE")
        sl_c = sum(1 for t in c_trades if t.exit_reason == "SL")

        eq = 100.0
        g_win = 0.0
        g_loss = 0.0
        for t in c_trades:
            eq *= 1.0 + t.equity_return_pct / 100.0
            if t.equity_return_pct > 0:
                g_win += t.equity_return_pct
            elif t.equity_return_pct < 0:
                g_loss += abs(t.equity_return_pct)

        pf = (g_win / g_loss) if g_loss > 0 else (99.99 if g_win > 0 else 0.0)
        rows.append(
            {
                "symbol": sym,
                "trades": n,
                "win_rate_pct": round((wins / n) * 100.0, 2),
                "tp": tp_c,
                "be": be_c,
                "sl": sl_c,
                "net_pnl_pct": round(eq - 100.0, 2),
                "profit_factor": round(pf, 2),
            }
        )

    qualified = [r for r in rows if r["trades"] >= min_trades]
    top_10 = sorted(qualified, key=lambda r: (r["net_pnl_pct"], r["win_rate_pct"]), reverse=True)[:10]
    worst_5 = sorted(rows, key=lambda r: (r["net_pnl_pct"], r["win_rate_pct"]))[:5]
    return {
        "min_trades": min_trades,
        "qualified_count": len(qualified),
        "top_10_best": top_10,
        "worst_5": worst_5,
        "all_coins": sorted(rows, key=lambda r: r["net_pnl_pct"], reverse=True),
    }


def simulate_grid_combo(
    params: GridParams,
    timeline: np.ndarray,
    split_ts: int,
    sym_data: Dict[str, Dict[str, np.ndarray]],
    spec_by_sym: Dict[str, InstrumentSpec],
    initial_equity: float = 1000.0,
    maker_fee_pct: float = 0.02,
    taker_fee_pct: float = 0.055,
    max_risk_pct: float = 2.5,
    max_open_positions: int = 3,
    max_same_direction: int = 2,
    daily_loss_limit_pct: float = 6.0,
    cooldown_bars: int = 3,
) -> List[ClosedTradeV2]:
    """Fast, strictly causal bar-by-bar portfolio simulator for one parameter combination."""
    maker_rate = maker_fee_pct / 100.0
    taker_rate = taker_fee_pct / 100.0

    equity = float(initial_equity)
    open_positions: List[OpenPosV2] = []
    pending_orders: List[PendingOrder] = []
    closed_trades: List[ClosedTradeV2] = []
    cooldown_until: Dict[str, int] = {}

    symbols = list(sym_data.keys())
    n_bars = len(timeline)

    current_day = -1
    day_start_equity = equity
    daily_halted = False

    def _close_pos(pos: OpenPosV2, exit_price: float, reason: str, exit_is_maker: bool, ts: int, bar_i: int) -> None:
        nonlocal equity
        gross = (
            (exit_price - pos.entry_price) * pos.qty
            if pos.direction == "LONG"
            else (pos.entry_price - exit_price) * pos.qty
        )
        exit_fee = abs(pos.qty * exit_price) * (maker_rate if exit_is_maker else taker_rate)
        total_fees = pos.entry_fee_usdt + exit_fee
        net = gross - total_fees
        equity += net
        eq_base = max(pos.equity_at_entry, 1e-9)
        closed_trades.append(
            ClosedTradeV2(
                symbol=pos.symbol,
                direction=pos.direction,
                period=pos.period,
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                leverage=pos.leverage,
                exit_reason=reason,
                gross_pnl_usdt=round(gross, 6),
                fees_usdt=round(total_fees, 6),
                net_pnl_usdt=round(net, 6),
                equity_return_pct=round((net / eq_base) * 100.0, 6),
                fees_pct_of_equity=round((total_fees / eq_base) * 100.0, 6),
            )
        )
        cooldown_until[pos.symbol] = bar_i + cooldown_bars

    for i in range(n_bars):
        ts = int(timeline[i])
        day_key = ts // 86_400_000
        if day_key != current_day:
            current_day = day_key
            day_start_equity = equity
            daily_halted = False

        # ------------------------------------------------------------------
        # 1. Manage existing open positions on bar `i` (conservative fills)
        # ------------------------------------------------------------------
        still_open: List[OpenPosV2] = []
        for pos in open_positions:
            sd = sym_data[pos.symbol]
            hi = sd["high"][i]
            lo = sd["low"][i]
            cl = sd["close"][i]
            if not np.isfinite(cl):
                still_open.append(pos)
                continue

            is_long = pos.direction == "LONG"
            if params.smart_be and pos.be_active:
                # Active BE stop (+0.15%) vs TP (conservative: BE wins on same-bar collision)
                be_p = pos.be_stop_price if pos.be_stop_price is not None else pos.sl_price
                be_hit = (lo <= be_p) if is_long else (hi >= be_p)
                tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                if be_hit:
                    _close_pos(pos, be_p, "BE", exit_is_maker=False, ts=ts, bar_i=i)
                elif tp_hit:
                    _close_pos(pos, pos.tp_price, "TP", exit_is_maker=True, ts=ts, bar_i=i)
                else:
                    still_open.append(pos)
            else:
                sl_hit = (lo <= pos.sl_price) if is_long else (hi >= pos.sl_price)
                tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)

                # Conservative priority: SL always wins if touched on same bar
                if sl_hit:
                    _close_pos(pos, pos.sl_price, "SL", exit_is_maker=False, ts=ts, bar_i=i)
                elif tp_hit:
                    _close_pos(pos, pos.tp_price, "TP", exit_is_maker=True, ts=ts, bar_i=i)
                else:
                    # Check Smart Break-Even activation (+0.85% reached -> move SL to +0.15%)
                    if params.smart_be and pos.be_trigger_price is not None:
                        trig = (hi >= pos.be_trigger_price) if is_long else (lo <= pos.be_trigger_price)
                        if trig:
                            pos.be_active = True
                    still_open.append(pos)

        open_positions = still_open

        # ------------------------------------------------------------------
        # 2. Check Daily Loss Limit (-6.0% realized + unrealized)
        # ------------------------------------------------------------------
        unrealized = 0.0
        for pos in open_positions:
            cl = sym_data[pos.symbol]["close"][i]
            if np.isfinite(cl):
                gross_u = (
                    (cl - pos.entry_price) * pos.qty
                    if pos.direction == "LONG"
                    else (pos.entry_price - cl) * pos.qty
                )
                unrealized += gross_u - pos.entry_fee_usdt

        dd_today = ((day_start_equity - (equity + unrealized)) / max(day_start_equity, 1e-9)) * 100.0
        if dd_today >= daily_loss_limit_pct:
            daily_halted = True

        # ------------------------------------------------------------------
        # 3. Process pending Maker limit orders on bar `i` (if entry_type == maker)
        # ------------------------------------------------------------------
        if params.entry_type == "maker" and pending_orders:
            next_pending: List[PendingOrder] = []
            for order in pending_orders:
                if i > order.expires_idx:
                    continue
                if daily_halted:
                    continue
                if any(p.symbol == order.symbol for p in open_positions):
                    continue
                if i < cooldown_until.get(order.symbol, -1):
                    continue

                sd = sym_data[order.symbol]
                hi = sd["high"][i]
                lo = sd["low"][i]
                if not np.isfinite(lo):
                    next_pending.append(order)
                    continue

                is_long = order.direction == "LONG"
                filled = (lo <= order.limit_price) if is_long else (hi >= order.limit_price)
                if not filled:
                    if i < order.expires_idx:
                        next_pending.append(order)
                    continue

                # Check portfolio caps at fill time
                if len(open_positions) >= max_open_positions:
                    continue
                if sum(1 for p in open_positions if p.direction == order.direction) >= max_same_direction:
                    continue

                plan = build_v2_position_plan(
                    symbol=order.symbol,
                    direction=order.direction,
                    entry_price=order.limit_price,
                    equity=equity,
                    tp_pct=params.tp_pct,
                    sl_pct=params.sl_pct,
                    entry_is_maker=True,
                    smart_be=params.smart_be,
                    be_trigger_pct=params.be_trigger_pct,
                    be_offset_pct=params.be_offset_pct,
                    max_risk_pct=max_risk_pct,
                    maker_fee_pct=maker_fee_pct,
                    taker_fee_pct=taker_fee_pct,
                    spec=spec_by_sym.get(order.symbol),
                )
                if plan is None:
                    continue

                entry_fee = plan.notional_usdt * maker_rate
                pos = OpenPosV2(
                    symbol=plan.symbol,
                    direction=plan.direction,
                    period="IS" if ts < split_ts else "OOS",
                    entry_ts=ts,
                    entry_idx=i,
                    entry_price=plan.entry_price,
                    sl_price=plan.sl_price,
                    tp_price=plan.tp_price,
                    be_trigger_price=plan.be_trigger_price,
                    be_stop_price=plan.be_stop_price,
                    qty=plan.qty,
                    leverage=plan.leverage,
                    notional_usdt=plan.notional_usdt,
                    equity_at_entry=equity,
                    entry_fee_usdt=entry_fee,
                    entry_is_maker=True,
                )

                # Conservative same-bar SL check on fill bar (never optimistic TP on fill bar)
                sl_same_bar = (lo <= pos.sl_price) if is_long else (hi >= pos.sl_price)
                if sl_same_bar:
                    _close_pos(pos, pos.sl_price, "SL", exit_is_maker=False, ts=ts, bar_i=i)
                else:
                    open_positions.append(pos)

            pending_orders = next_pending

        if daily_halted:
            continue

        # ------------------------------------------------------------------
        # 4. Scan closed 15M bar `i` for new `trend_pullback_v2` signals
        # ------------------------------------------------------------------
        if len(open_positions) >= max_open_positions:
            continue

        busy_syms = {p.symbol for p in open_positions} | {o.symbol for o in pending_orders}
        candidates: List[Tuple[float, str, str, float]] = []

        for sym in symbols:
            if sym in busy_syms:
                continue
            if i < cooldown_until.get(sym, -1):
                continue

            sd = sym_data[sym]
            if sd["sig_long"][i]:
                candidates.append((float(sd["q_long"][i]), sym, "LONG", float(sd["close"][i])))
            elif sd["sig_short"][i]:
                candidates.append((float(sd["q_short"][i]), sym, "SHORT", float(sd["close"][i])))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)

        for q, sym, direction, close_p in candidates:
            if len(open_positions) + len(pending_orders) >= max_open_positions + (2 if params.entry_type == "maker" else 0):
                break
            same_dir_open = sum(1 for p in open_positions if p.direction == direction)
            if same_dir_open >= max_same_direction:
                continue

            if params.entry_type == "maker":
                pb = params.maker_pullback_pct / 100.0
                limit_p = close_p * (1.0 - pb) if direction == "LONG" else close_p * (1.0 + pb)
                pending_orders.append(
                    PendingOrder(
                        symbol=sym,
                        direction=direction,
                        limit_price=limit_p,
                        placed_idx=i,
                        expires_idx=i + params.maker_ttl_bars,
                        quality=q,
                    )
                )
            else:
                if len(open_positions) >= max_open_positions:
                    break
                plan = build_v2_position_plan(
                    symbol=sym,
                    direction=direction,
                    entry_price=close_p,
                    equity=equity,
                    tp_pct=params.tp_pct,
                    sl_pct=params.sl_pct,
                    entry_is_maker=False,
                    smart_be=params.smart_be,
                    be_trigger_pct=params.be_trigger_pct,
                    be_offset_pct=params.be_offset_pct,
                    max_risk_pct=max_risk_pct,
                    maker_fee_pct=maker_fee_pct,
                    taker_fee_pct=taker_fee_pct,
                    spec=spec_by_sym.get(sym),
                )
                if plan is None:
                    continue
                entry_fee = plan.notional_usdt * taker_rate
                open_positions.append(
                    OpenPosV2(
                        symbol=plan.symbol,
                        direction=plan.direction,
                        period="IS" if ts < split_ts else "OOS",
                        entry_ts=ts,
                        entry_idx=i,
                        entry_price=plan.entry_price,
                        sl_price=plan.sl_price,
                        tp_price=plan.tp_price,
                        be_trigger_price=plan.be_trigger_price,
                        be_stop_price=plan.be_stop_price,
                        qty=plan.qty,
                        leverage=plan.leverage,
                        notional_usdt=plan.notional_usdt,
                        equity_at_entry=equity,
                        entry_fee_usdt=entry_fee,
                        entry_is_maker=False,
                    )
                )

    # Mark-to-market close remaining positions at dataset end
    if open_positions and n_bars > 0:
        last_ts = int(timeline[-1])
        for pos in open_positions:
            cl = sym_data[pos.symbol]["close"][-1]
            if not np.isfinite(cl):
                cl = pos.entry_price
            _close_pos(pos, float(cl), "EOD", exit_is_maker=False, ts=last_ts, bar_i=n_bars - 1)

    return closed_trades


def _download_symbol(client: BybitClient, sym: str, interval: str, start_ms: int, end_ms: int) -> Tuple[str, pd.DataFrame]:
    df = client.get_klines(sym, interval=interval, start_ms=start_ms, end_ms=end_ms, use_cache=True, closed_only=True)
    return sym, df


def run_grid_backtest(
    days: int = 180,
    top_n: int = 25,
    is_days: int = 120,
    oos_days: int = 60,
    n_sims: int = 10_000,
    output_path: str | Path = "data/backtest_v2_results.json",
) -> Dict[str, Any]:
    t0 = time.time()
    blacklist, _ = load_blacklist("data/blacklist.json")
    client = BybitClient(mode="testnet", cache_dir="data/cache", public_mainnet=True)

    logger.info("Fetching fresh Top-%d PURE CRYPTO Bybit USDT perpetuals (min_order <= 20 USDT)...", top_n)
    raw_top = client.get_top_usdt_perpetuals(top_n=top_n + 10, max_min_order_usdt=20.0, crypto_only=True)
    universe: List[InstrumentSpec] = [s for s in raw_top if s.symbol != "BTCUSDT" and s.symbol not in blacklist][:top_n]
    spec_by_sym = {s.symbol: s for s in universe}

    logger.info("Selected %d pure crypto pairs:", len(universe))
    logger.info("  %s", ", ".join(s.symbol for s in universe))

    entry_ms = INTERVAL_TO_MS["15"]
    now_ms = int(time.time() * 1000)
    sim_end_ms = ((now_ms // entry_ms) - 1) * entry_ms
    sim_start_ms = sim_end_ms - days * 86_400_000
    # 35 days warmup so 4H EMA(200) (200 * 4h = 33.3 days) is 100% warmed up before day 1
    warmup_days = 36
    fetch_start_ms = sim_start_ms - warmup_days * 86_400_000
    split_ms = sim_start_ms + is_days * 86_400_000

    all_symbols = ["BTCUSDT"] + [s.symbol for s in universe]
    raw_frames: Dict[str, pd.DataFrame] = {}

    logger.info("Loading/updating cached 15M klines (180d + %dd warmup) for %d symbols...", warmup_days, len(all_symbols))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {
            pool.submit(_download_symbol, client, sym, "15", fetch_start_ms, sim_end_ms): sym
            for sym in all_symbols
        }
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            done += 1
            try:
                s_name, df_s = fut.result()
                if not df_s.empty:
                    raw_frames[s_name] = df_s
                if done % 5 == 0 or done == len(all_symbols):
                    logger.info("  Ready %d/%d (%s: %d bars)", done, len(all_symbols), sym, len(df_s))
            except Exception as exc:
                logger.warning("  Failed %s: %s", sym, exc)

    df_btc = raw_frames["BTCUSDT"]
    logger.info("Computing `trend_pullback_v2` features (15M/1H/4H EMA50+EMA200+ADX+VWAP+Fib+RSI)...")

    # Master unified timeline across [sim_start_ms, sim_end_ms]
    btc_sim_ts = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]["timestamp"].to_numpy(dtype=np.int64)
    timeline = np.sort(np.unique(btc_sim_ts))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    sym_data: Dict[str, Dict[str, np.ndarray]] = {}
    active_symbols: List[str] = []

    for spec in universe:
        sym = spec.symbol
        df_15m = raw_frames.get(sym)
        if df_15m is None or len(df_15m) < 500:
            logger.warning("Skipping %s: insufficient candles (%d)", sym, 0 if df_15m is None else len(df_15m))
            continue

        df_1h = BybitClient.resample_klines(df_15m, "60")
        df_4h = BybitClient.resample_klines(df_15m, "240")
        enriched = enrich_v2_dataframe(
            df_15m=df_15m,
            df_btc_15m=df_btc,
            df_1h=df_1h,
            df_4h=df_4h,
            btc_shock_pause_pct=0.8,
        )

        # Align onto exact unified timeline for O(1) array indexing
        aligned = timeline_df.merge(
            enriched[["timestamp", "high", "low", "close", "sig_long", "sig_short", "quality_long", "quality_short"]],
            on="timestamp",
            how="left",
        )
        sym_data[sym] = {
            "high": aligned["high"].to_numpy(dtype=np.float64),
            "low": aligned["low"].to_numpy(dtype=np.float64),
            "close": aligned["close"].to_numpy(dtype=np.float64),
            "sig_long": aligned["sig_long"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "sig_short": aligned["sig_short"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "q_long": aligned["quality_long"].fillna(0.0).to_numpy(dtype=np.float64),
            "q_short": aligned["quality_short"].fillna(0.0).to_numpy(dtype=np.float64),
        }
        active_symbols.append(sym)

    # Build full 24-combination grid
    grid: List[GridParams] = []
    for tp in (1.25, 1.0):
        for sl in (2.8, 3.2, 3.5):
            for entry_type in ("maker", "taker"):
                for smart_be in (False, True):
                    grid.append(GridParams(tp_pct=tp, sl_pct=sl, entry_type=entry_type, smart_be=smart_be))

    logger.info("Running %d grid combinations x 10,000 Monte Carlo sims on %d pure crypto pairs...", len(grid), len(active_symbols))

    grid_results: List[Dict[str, Any]] = []
    trades_by_label: Dict[str, List[ClosedTradeV2]] = {}

    for idx_g, gp in enumerate(grid):
        trades = simulate_grid_combo(
            params=gp,
            timeline=timeline,
            split_ts=split_ms,
            sym_data=sym_data,
            spec_by_sym=spec_by_sym,
            initial_equity=1000.0,
            maker_fee_pct=0.02,
            taker_fee_pct=0.055,
            max_risk_pct=2.5,
            max_open_positions=3,
            max_same_direction=2,
            daily_loss_limit_pct=6.0,
            cooldown_bars=3,
        )
        is_trades = [t for t in trades if t.period == "IS"]
        oos_trades = [t for t in trades if t.period == "OOS"]

        m_is = compute_metrics_v2(is_trades, is_days, n_sims=n_sims, seed=100 + idx_g)
        m_oos = compute_metrics_v2(oos_trades, oos_days, n_sims=n_sims, seed=200 + idx_g)
        m_tot = compute_metrics_v2(trades, days, n_sims=n_sims, seed=300 + idx_g)

        trades_by_label[gp.label] = trades
        grid_results.append(
            {
                "label": gp.label,
                "params": asdict(gp),
                "in_sample": m_is,
                "out_of_sample": m_oos,
                "total": m_tot,
            }
        )

    # Sort combinations: prioritize positive OOS & Total Net PnL, Profit Factor, and controlled drawdown
    def _rank_key(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
        tot = item["total"]
        oos = item["out_of_sample"]
        is_m = item["in_sample"]
        both_pos = 1.0 if (is_m["net_pnl_pct"] > 0 and oos["net_pnl_pct"] > 0) else 0.0
        # Composite risk-adjusted score across IS + OOS
        return (
            both_pos,
            tot["net_pnl_pct"],
            oos["net_pnl_pct"],
            -tot["mc_p95_dd_pct"],
        )

    grid_results.sort(key=_rank_key, reverse=True)
    top_5 = grid_results[:5]
    best = top_5[0]
    best_trades = trades_by_label[best["label"]]
    best_coin_stats = compute_coin_stats_v2(best_trades, min_trades=15)

    elapsed = round(time.time() - t0, 2)
    report = {
        "meta": {
            "strategy": "trend_pullback_v2",
            "crypto_only": True,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_utc": datetime.fromtimestamp(sim_start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "split_utc": datetime.fromtimestamp(split_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "end_utc": datetime.fromtimestamp(sim_end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "days": days,
            "is_days": is_days,
            "oos_days": oos_days,
            "mc_sims": n_sims,
            "elapsed_sec": elapsed,
            "universe_25": [
                {
                    "rank": r_i + 1,
                    "symbol": s.symbol,
                    "turnover_24h_musdt": round(s.turnover_24h / 1e6, 2),
                    "min_order_usdt": round(s.min_order_usdt, 2),
                    "bars_loaded": len(raw_frames.get(s.symbol, [])),
                }
                for r_i, s in enumerate(universe)
            ],
        },
        "top_5": top_5,
        "all_grid": grid_results,
        "best_combo_coins": best_coin_stats,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Grid backtest complete in %.2fs -> saved to %s", elapsed, out_p)

    _print_report(report)
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 118)
    print(f"PURE CRYPTO TOP-25 UNIVERSE ({report['meta']['start_utc']} -> {report['meta']['end_utc']})")
    print("=" * 118)
    syms = [u["symbol"] for u in report["meta"]["universe_25"]]
    print("  " + ", ".join(syms))

    print("\n" + "=" * 118)
    print("TOP-5 PARAMETER COMBINATIONS (trend_pullback_v2 | IS=120d, OOS=60d | 10,000 Monte Carlo)")
    print("=" * 118)
    for idx, item in enumerate(report["top_5"], 1):
        m_is = item["in_sample"]
        m_oos = item["out_of_sample"]
        m_tot = item["total"]
        print(f"\n#{idx}: {item['label']}")
        print(
            f"  IS  (120d): {m_is['trades']:>4} tr ({m_is['trades_per_day']:.2f}/d) | WR={m_is['win_rate_pct']:>6.2f}% "
            f"(TP={m_is['tp_count']}, BE={m_is['be_count']}, SL={m_is['sl_count']}) | "
            f"Fees={m_is['fees_pct']:>5.2f}% | Net={m_is['net_pnl_pct']:>+7.2f}% | PF={m_is['profit_factor']:>5.2f} | "
            f"MaxDD={m_is['max_dd_pct']:>5.2f}% | MC95DD={m_is['mc_p95_dd_pct']:>5.2f}%"
        )
        print(
            f"  OOS  (60d): {m_oos['trades']:>4} tr ({m_oos['trades_per_day']:.2f}/d) | WR={m_oos['win_rate_pct']:>6.2f}% "
            f"(TP={m_oos['tp_count']}, BE={m_oos['be_count']}, SL={m_oos['sl_count']}) | "
            f"Fees={m_oos['fees_pct']:>5.2f}% | Net={m_oos['net_pnl_pct']:>+7.2f}% | PF={m_oos['profit_factor']:>5.2f} | "
            f"MaxDD={m_oos['max_dd_pct']:>5.2f}% | MC95DD={m_oos['mc_p95_dd_pct']:>5.2f}%"
        )
        print(
            f"  TOT (180d): {m_tot['trades']:>4} tr ({m_tot['trades_per_day']:.2f}/d) | WR={m_tot['win_rate_pct']:>6.2f}% "
            f"(TP={m_tot['tp_count']}, BE={m_tot['be_count']}, SL={m_tot['sl_count']}) | "
            f"Fees={m_tot['fees_pct']:>5.2f}% | Net={m_tot['net_pnl_pct']:>+7.2f}% | PF={m_tot['profit_factor']:>5.2f} | "
            f"MaxDD={m_tot['max_dd_pct']:>5.2f}% | MC95DD={m_tot['mc_p95_dd_pct']:>5.2f}%"
        )

    best_coins = report["best_combo_coins"]
    print("\n" + "=" * 118)
    print(f"TOP-10 BEST CRYPTO COINS FOR #1 COMBO ({report['top_5'][0]['label']}) [min {best_coins['min_trades']} trades; qualified={best_coins['qualified_count']}]:")
    print("=" * 118)
    print(f"  {'Symbol':<14} {'Trades':>6} {'WinRate':>9} {'TP':>5} {'BE':>5} {'SL':>5} {'Net PnL%':>10} {'PF':>6}")
    for c in best_coins["top_10_best"]:
        print(
            f"  {c['symbol']:<14} {c['trades']:>6} {c['win_rate_pct']:>8.2f}% "
            f"{c['tp']:>5} {c['be']:>5} {c['sl']:>5} {c['net_pnl_pct']:>+9.2f}% {c['profit_factor']:>6.2f}"
        )
    print("\nALL COINS (sorted by Net PnL%):")
    for c in best_coins["all_coins"]:
        print(
            f"  {c['symbol']:<14} {c['trades']:>6} {c['win_rate_pct']:>8.2f}% "
            f"{c['tp']:>5} {c['be']:>5} {c['sl']:>5} {c['net_pnl_pct']:>+9.2f}% {c['profit_factor']:>6.2f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Variant 1 (`trend_pullback_v2`) Grid Backtest + Monte Carlo")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--is-days", type=int, default=120)
    parser.add_argument("--oos-days", type=int, default=60)
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--output", type=str, default="data/backtest_v2_results.json")
    args = parser.parse_args()

    run_grid_backtest(
        days=args.days,
        top_n=args.top,
        is_days=args.is_days,
        oos_days=args.oos_days,
        n_sims=args.sims,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
