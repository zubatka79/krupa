"""Honest bar-by-bar historical backtester on real Bybit 15M/1H/4H data with IS/OOS split."""

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

# Allow running directly via `python bot/backtest.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.bybit_client import BybitClient, INTERVAL_TO_MS, InstrumentSpec
from bot.indicators import enrich_entry_dataframe
from bot.risk_manager import RiskManager
from bot.scanner import load_blacklist, score_closed_bar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backtest")


@dataclass
class SimPosition:
    symbol: str
    direction: str
    entry_ts: int
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    be_price: float
    sl_pct: float
    qty: float
    tp1_qty: float
    tp2_qty: float
    leverage: int
    margin_usdt: float
    notional_usdt: float
    equity_at_entry: float
    score: float
    tp1_hit: bool = False
    realized_gross: float = 0.0
    realized_fees: float = 0.0
    period: str = "IS"  # "IS" or "OOS"


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    period: str  # "IS" or "OOS"
    entry_ts: int
    exit_ts: int
    entry_price: float
    sl_pct: float
    leverage: int
    score: float
    exit_reason: str  # "TP1+TP2", "TP1+BE", "SL", "EOD"
    gross_pnl_usdt: float
    fees_usdt: float
    net_pnl_usdt: float
    equity_return_pct: float  # net_pnl_usdt / equity_at_entry * 100
    tp1_hit: bool


def load_config(path: str | Path = "config.json") -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compute_period_metrics(trades: List[ClosedTrade], days: int) -> Dict[str, Any]:
    """Compute exact compounded Net PnL %, Win Rate breakdown, Profit Factor, and Max Drawdown %."""
    n = len(trades)
    if n == 0:
        return {
            "days": days,
            "trades": 0,
            "trades_per_day": 0.0,
            "win_rate_pct": 0.0,
            "tp1_tp2_count": 0,
            "tp1_tp2_pct": 0.0,
            "tp1_be_count": 0,
            "tp1_be_pct": 0.0,
            "sl_count": 0,
            "sl_pct": 0.0,
            "eod_count": 0,
            "net_pnl_pct": 0.0,
            "gross_pnl_pct": 0.0,
            "fees_impact_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_trade_return_pct": 0.0,
        }

    tp1_tp2 = sum(1 for t in trades if t.exit_reason == "TP1+TP2")
    tp1_be = sum(1 for t in trades if t.exit_reason == "TP1+BE")
    sl = sum(1 for t in trades if t.exit_reason == "SL")
    eod = sum(1 for t in trades if t.exit_reason == "EOD")

    wins = sum(1 for t in trades if t.net_pnl_usdt > 0)
    win_rate = (wins / n) * 100.0

    # Compounded normalized equity curve starting at 100.0 for apples-to-apples IS / OOS / Total %
    eq = 100.0
    peak = 100.0
    max_dd = 0.0
    gross_win_pct = 0.0
    gross_loss_pct = 0.0
    sum_gross_pct = 0.0
    sum_fees_pct = 0.0

    for t in trades:
        r_pct = t.equity_return_pct
        eq *= 1.0 + (r_pct / 100.0)
        if eq > peak:
            peak = eq
        dd = ((peak - eq) / peak) * 100.0 if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

        if r_pct > 0:
            gross_win_pct += r_pct
        elif r_pct < 0:
            gross_loss_pct += abs(r_pct)

        if t.equity_return_pct != 0 and t.net_pnl_usdt != 0:
            scale = t.equity_return_pct / t.net_pnl_usdt
            sum_gross_pct += t.gross_pnl_usdt * scale
            sum_fees_pct += t.fees_usdt * scale

    pf = (gross_win_pct / gross_loss_pct) if gross_loss_pct > 0 else (99.99 if gross_win_pct > 0 else 0.0)
    net_pnl_pct = eq - 100.0

    return {
        "days": days,
        "trades": n,
        "trades_per_day": round(n / max(days, 1), 2),
        "win_rate_pct": round(win_rate, 2),
        "tp1_tp2_count": tp1_tp2,
        "tp1_tp2_pct": round((tp1_tp2 / n) * 100.0, 2),
        "tp1_be_count": tp1_be,
        "tp1_be_pct": round((tp1_be / n) * 100.0, 2),
        "sl_count": sl,
        "sl_pct": round((sl / n) * 100.0, 2),
        "eod_count": eod,
        "net_pnl_pct": round(net_pnl_pct, 2),
        "gross_pnl_pct": round(sum_gross_pct, 2),
        "fees_impact_pct": round(sum_fees_pct, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_trade_return_pct": round(float(np.mean([t.equity_return_pct for t in trades])), 3),
    }


def compute_coin_rankings(trades: List[ClosedTrade], min_trades_for_top: int = 15) -> Dict[str, Any]:
    """Aggregate per-coin Win Rate, Net PnL %, and Profit Factor."""
    by_coin: Dict[str, List[ClosedTrade]] = {}
    for t in trades:
        by_coin.setdefault(t.symbol, []).append(t)

    rows: List[Dict[str, Any]] = []
    for sym, c_trades in by_coin.items():
        n = len(c_trades)
        wins = sum(1 for t in c_trades if t.net_pnl_usdt > 0)
        tp1_tp2 = sum(1 for t in c_trades if t.exit_reason == "TP1+TP2")
        tp1_be = sum(1 for t in c_trades if t.exit_reason == "TP1+BE")
        sl = sum(1 for t in c_trades if t.exit_reason == "SL")

        eq = 100.0
        g_win = 0.0
        g_loss = 0.0
        for t in c_trades:
            eq *= 1.0 + (t.equity_return_pct / 100.0)
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
                "tp1_tp2": tp1_tp2,
                "tp1_be": tp1_be,
                "sl": sl,
                "net_pnl_pct": round(eq - 100.0, 2),
                "sum_return_pct": round(sum(t.equity_return_pct for t in c_trades), 2),
                "profit_factor": round(pf, 2),
            }
        )

    qualified = [r for r in rows if r["trades"] >= min_trades_for_top]
    top_10 = sorted(qualified, key=lambda r: (r["net_pnl_pct"], r["win_rate_pct"]), reverse=True)[:10]
    worst_5 = sorted(rows, key=lambda r: (r["net_pnl_pct"], r["win_rate_pct"]))[:5]
    return {
        "all_coins": sorted(rows, key=lambda r: r["net_pnl_pct"], reverse=True),
        "qualified_count": len(qualified),
        "min_trades_threshold": min_trades_for_top,
        "top_10_best": top_10,
        "worst_5": worst_5,
    }


def _download_symbol_data(
    client: BybitClient,
    symbol: str,
    entry_tf: str,
    start_ms: int,
    end_ms: int,
) -> Tuple[str, pd.DataFrame]:
    df = client.get_klines(
        symbol=symbol,
        interval=entry_tf,
        start_ms=start_ms,
        end_ms=end_ms,
        use_cache=True,
        closed_only=True,
    )
    return symbol, df


def run_backtest(
    days: int = 180,
    top_n: int = 25,
    is_days: int = 120,
    oos_days: int = 60,
    initial_equity: float = 1000.0,
    config_path: str | Path = "config.json",
    output_path: str | Path = "data/backtest_results.json",
    ignore_whitelist: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    config = load_config(config_path)
    risk = RiskManager(config)
    blacklist, _ = load_blacklist("data/blacklist.json")

    entry_tf = str(config.get("entry_tf", "15"))
    context_tf = str(config.get("context_tf", "60"))
    trend_tf = str(config.get("trend_tf", "240"))
    entry_ms = INTERVAL_TO_MS[entry_tf]
    context_ms = INTERVAL_TO_MS[context_tf]
    trend_ms = INTERVAL_TO_MS[trend_tf]

    # Always use Bybit mainnet public endpoints for realistic historical data (no API keys required)
    client = BybitClient(mode=config.get("mode", "testnet"), cache_dir="data/cache", public_mainnet=True)

    whitelist = [] if ignore_whitelist else [str(s).upper().strip() for s in config.get("whitelist", []) if str(s).strip()]
    if whitelist:
        logger.info("Using whitelist from config.json (%d symbols).", len(whitelist))
        specs_map = client.get_instrument_specs_for_symbols(whitelist)
        universe: List[InstrumentSpec] = [specs_map[s] for s in whitelist if s in specs_map and s != "BTCUSDT"]
    else:
        logger.info("Fetching top-%d Bybit USDT perpetuals (min_order <= 20 USDT)...", top_n)
        raw_top = client.get_top_usdt_perpetuals(top_n=top_n + 5, max_min_order_usdt=20.0)
        universe = [s for s in raw_top if s.symbol != "BTCUSDT" and s.symbol not in blacklist][:top_n]

    if not universe:
        raise RuntimeError("No eligible Bybit USDT perpetual symbols found.")

    spec_by_sym: Dict[str, InstrumentSpec] = {s.symbol: s for s in universe}
    logger.info("Selected %d symbols: %s", len(universe), ", ".join(s.symbol for s in universe))

    # Align end time to latest closed 15M bar
    now_ms = int(time.time() * 1000)
    sim_end_ms = ((now_ms // entry_ms) - 1) * entry_ms
    sim_start_ms = sim_end_ms - days * 86_400_000
    warmup_days = 14
    fetch_start_ms = sim_start_ms - warmup_days * 86_400_000

    # In-Sample = first is_days, Out-of-Sample = last oos_days
    if is_days + oos_days != days:
        is_days = int(round(days * (2.0 / 3.0)))
        oos_days = days - is_days
    split_ms = sim_start_ms + is_days * 86_400_000

    logger.info(
        "Window: %s -> %s (%dd total | IS=%dd, OOS=%dd)",
        datetime.fromtimestamp(sim_start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        datetime.fromtimestamp(sim_end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        days,
        is_days,
        oos_days,
    )

    # Fetch BTCUSDT + universe klines in parallel using incremental .csv.gz cache
    all_symbols = ["BTCUSDT"] + [s.symbol for s in universe]
    raw_frames: Dict[str, pd.DataFrame] = {}

    logger.info("Downloading / loading cached 15M klines for %d symbols...", len(all_symbols))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_download_symbol_data, client, sym, entry_tf, fetch_start_ms, sim_end_ms): sym
            for sym in all_symbols
        }
        done_cnt = 0
        for fut in as_completed(futures):
            sym = futures[fut]
            done_cnt += 1
            try:
                s_name, df_sym = fut.result()
                if not df_sym.empty:
                    raw_frames[s_name] = df_sym
                if done_cnt % 5 == 0 or done_cnt == len(all_symbols):
                    logger.info("  Loaded %d/%d symbols (%s: %d bars)", done_cnt, len(all_symbols), sym, len(df_sym))
            except Exception as exc:
                logger.warning("  Failed loading %s: %s", sym, exc)

    if "BTCUSDT" not in raw_frames or raw_frames["BTCUSDT"].empty:
        raise RuntimeError("Failed to download BTCUSDT reference klines.")

    df_btc = raw_frames["BTCUSDT"]

    # Enrich each symbol with causal indicators + precompute bar lookup maps
    logger.info("Computing indicators and multi-timeframe features...")
    enriched_map: Dict[str, pd.DataFrame] = {}
    arrays_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    ts_to_idx: Dict[str, Dict[int, int]] = {}
    all_timestamps_set = set()

    for spec in universe:
        sym = spec.symbol
        df_15m = raw_frames.get(sym)
        if df_15m is None or len(df_15m) < 400:
            logger.warning("Skipping %s: insufficient candles (%d).", sym, 0 if df_15m is None else len(df_15m))
            continue

        df_1h = BybitClient.resample_klines(df_15m, context_tf)
        df_4h = BybitClient.resample_klines(df_15m, trend_tf)
        enriched = enrich_entry_dataframe(
            df_15m=df_15m,
            df_btc_15m=df_btc,
            df_1h=df_1h,
            df_4h=df_4h,
            entry_interval_ms=entry_ms,
            context_interval_ms=context_ms,
            trend_interval_ms=trend_ms,
        )
        enriched_map[sym] = enriched
        arrays_map[sym] = (
            enriched["high"].to_numpy(dtype=np.float64),
            enriched["low"].to_numpy(dtype=np.float64),
            enriched["volume"].to_numpy(dtype=np.float64),
        )
        ts_arr = enriched["timestamp"].to_numpy(dtype=np.int64)
        idx_lookup = {int(ts): i for i, ts in enumerate(ts_arr)}
        ts_to_idx[sym] = idx_lookup
        for ts in ts_arr:
            if sim_start_ms <= int(ts) <= sim_end_ms:
                all_timestamps_set.add(int(ts))

    timeline = sorted(all_timestamps_set)
    logger.info("Simulating %d chronological 15M bars across %d active symbols...", len(timeline), len(enriched_map))

    # Allowed directions from config
    dir_mode = str(config.get("allowed_directions", "long_short")).lower()
    if dir_mode == "paused":
        directions: List[str] = []
    elif dir_mode == "only_long":
        directions = ["LONG"]
    elif dir_mode == "only_short":
        directions = ["SHORT"]
    else:
        directions = ["LONG", "SHORT"]

    # Portfolio simulation state
    equity = float(initial_equity)
    open_positions: List[SimPosition] = []
    closed_trades: List[ClosedTrade] = []

    current_day_key: Optional[int] = None
    day_start_equity = equity
    daily_halted = False
    daily_halt_days = 0

    # Also track standalone signals per symbol (for diagnostic context)
    active_symbols = list(enriched_map.keys())

    for ts in timeline:
        day_key = ts // 86_400_000
        if current_day_key is None or day_key != current_day_key:
            current_day_key = day_key
            day_start_equity = equity
            daily_halted = False

        # ------------------------------------------------------------------
        # 1. Manage existing open positions on bar `ts` (conservative fills)
        # ------------------------------------------------------------------
        still_open: List[SimPosition] = []
        for pos in open_positions:
            idx = ts_to_idx[pos.symbol].get(ts)
            if idx is None:
                still_open.append(pos)
                continue

            bar = enriched_map[pos.symbol].iloc[idx]
            hi = float(bar["high"])
            lo = float(bar["low"])
            cl = float(bar["close"])
            is_long = pos.direction == "LONG"

            closed_reason: Optional[str] = None
            rem_gross = 0.0
            rem_fee = 0.0

            if not pos.tp1_hit:
                sl_hit = (lo <= pos.sl_price) if is_long else (hi >= pos.sl_price)
                tp1_hit = (hi >= pos.tp1_price) if is_long else (lo <= pos.tp1_price)

                # Conservative priority: if BOTH SL and TP1 hit inside same 15M candle -> SL fills first!
                if sl_hit:
                    rem_gross = risk.leg_gross_pnl(pos.direction, pos.entry_price, pos.sl_price, pos.qty)
                    rem_fee = risk.fee_usdt(pos.qty * pos.sl_price, is_maker=False)
                    closed_reason = "SL"
                elif tp1_hit:
                    # Close 55% at TP1 (Maker limit) and move SL to BE (+0.15%)
                    tp1_gross = risk.leg_gross_pnl(pos.direction, pos.entry_price, pos.tp1_price, pos.tp1_qty)
                    tp1_fee = risk.fee_usdt(pos.tp1_qty * pos.tp1_price, is_maker=True)
                    pos.realized_gross += tp1_gross
                    pos.realized_fees += tp1_fee
                    pos.tp1_hit = True
                    pos.sl_price = pos.be_price

                    # Same-bar TP2 check: ONLY count same-bar TP2 if candle cleanly closed at/beyond TP2
                    tp2_clean = (hi >= pos.tp2_price and cl >= pos.tp2_price) if is_long else (
                        lo <= pos.tp2_price and cl <= pos.tp2_price
                    )
                    if tp2_clean:
                        rem_gross = risk.leg_gross_pnl(pos.direction, pos.entry_price, pos.tp2_price, pos.tp2_qty)
                        rem_fee = risk.fee_usdt(pos.tp2_qty * pos.tp2_price, is_maker=True)
                        closed_reason = "TP1+TP2"
            else:
                # TP1 already filled; SL is now at breakeven (+0.15%), waiting for TP2 or BE
                be_hit = (lo <= pos.be_price) if is_long else (hi >= pos.be_price)
                tp2_hit = (hi >= pos.tp2_price) if is_long else (lo <= pos.tp2_price)

                # Conservative priority: if BOTH BE and TP2 touched on same bar -> BE fills first!
                if be_hit:
                    rem_gross = risk.leg_gross_pnl(pos.direction, pos.entry_price, pos.be_price, pos.tp2_qty)
                    rem_fee = risk.fee_usdt(pos.tp2_qty * pos.be_price, is_maker=False)
                    closed_reason = "TP1+BE"
                elif tp2_hit:
                    rem_gross = risk.leg_gross_pnl(pos.direction, pos.entry_price, pos.tp2_price, pos.tp2_qty)
                    rem_fee = risk.fee_usdt(pos.tp2_qty * pos.tp2_price, is_maker=True)
                    closed_reason = "TP1+TP2"

            if closed_reason is not None:
                total_gross = pos.realized_gross + rem_gross
                total_fees = pos.realized_fees + rem_fee
                net_pnl = total_gross - total_fees
                equity += net_pnl
                ret_pct = (net_pnl / max(pos.equity_at_entry, 1e-9)) * 100.0
                closed_trades.append(
                    ClosedTrade(
                        symbol=pos.symbol,
                        direction=pos.direction,
                        period=pos.period,
                        entry_ts=pos.entry_ts,
                        exit_ts=ts,
                        entry_price=pos.entry_price,
                        sl_pct=pos.sl_pct,
                        leverage=pos.leverage,
                        score=pos.score,
                        exit_reason=closed_reason,
                        gross_pnl_usdt=round(total_gross, 6),
                        fees_usdt=round(total_fees, 6),
                        net_pnl_usdt=round(net_pnl, 6),
                        equity_return_pct=round(ret_pct, 6),
                        tp1_hit=pos.tp1_hit,
                    )
                )
            else:
                still_open.append(pos)

        open_positions = still_open

        # ------------------------------------------------------------------
        # 2. Check Daily Loss Circuit-Breaker (-6% from UTC day start equity)
        # ------------------------------------------------------------------
        # Include unrealized PnL of open positions in daily drawdown check
        unrealized = 0.0
        for pos in open_positions:
            idx = ts_to_idx[pos.symbol].get(ts)
            if idx is not None:
                cl = float(enriched_map[pos.symbol].iloc[idx]["close"])
                rem_qty = pos.tp2_qty if pos.tp1_hit else pos.qty
                unrealized += pos.realized_gross - pos.realized_fees + risk.leg_gross_pnl(
                    pos.direction, pos.entry_price, cl, rem_qty
                )

        halted_now, _ = risk.is_daily_loss_halted(day_start_equity, equity + unrealized)
        if halted_now and not daily_halted:
            daily_halted = True
            daily_halt_days += 1

        if daily_halted or not directions or len(open_positions) >= risk.max_open_positions:
            continue

        # ------------------------------------------------------------------
        # 3. Scan closed 15M bars for new entries (highest score first)
        # ------------------------------------------------------------------
        open_syms = {p.symbol for p in open_positions}
        candidates: List[Tuple[float, str, str, float, float]] = []

        for sym in active_symbols:
            if sym in open_syms:
                continue
            idx = ts_to_idx[sym].get(ts)
            if idx is None or idx < 100:
                continue

            df_sym = enriched_map[sym]
            highs, lows, volumes = arrays_map[sym]

            best_dir: Optional[Tuple[float, str, float]] = None
            for direction in directions:
                scored = score_closed_bar(
                    df=df_sym,
                    idx=idx,
                    direction=direction,
                    config=config,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    oi_drop_pct=None,  # 15m OI unavailable on 180d history -> graceful redistribution
                    funding_rate=None,
                )
                if scored is not None:
                    sc, sl_p, _ = scored
                    if best_dir is None or sc > best_dir[0]:
                        best_dir = (sc, direction, sl_p)

            if best_dir is not None:
                sc, direction, sl_p = best_dir
                entry_price = float(df_sym.iloc[idx]["close"])
                candidates.append((sc, sym, direction, sl_p, entry_price))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        period_label = "IS" if ts < split_ms else "OOS"

        for sc, sym, direction, sl_p, entry_price in candidates:
            if len(open_positions) >= risk.max_open_positions:
                break
            same_dir = sum(1 for p in open_positions if p.direction == direction)
            if same_dir >= risk.max_same_direction:
                continue

            plan, _ = risk.build_position_plan(
                symbol=sym,
                direction=direction,
                entry_price=entry_price,
                sl_pct=sl_p,
                equity=equity,
                spec=spec_by_sym.get(sym),
            )
            if plan is None:
                continue

            entry_fee = risk.fee_usdt(plan.notional_usdt, is_maker=False)
            open_positions.append(
                SimPosition(
                    symbol=sym,
                    direction=direction,
                    entry_ts=ts,
                    entry_price=plan.entry_price,
                    sl_price=plan.sl_price,
                    tp1_price=plan.tp1_price,
                    tp2_price=plan.tp2_price,
                    be_price=plan.be_price,
                    sl_pct=plan.sl_pct,
                    qty=plan.qty,
                    tp1_qty=plan.tp1_qty,
                    tp2_qty=plan.tp2_qty,
                    leverage=plan.leverage,
                    margin_usdt=plan.margin_usdt,
                    notional_usdt=plan.notional_usdt,
                    equity_at_entry=equity,
                    score=sc,
                    tp1_hit=False,
                    realized_gross=0.0,
                    realized_fees=entry_fee,
                    period=period_label,
                )
            )

    # Mark-to-market close any remaining open trades at end of dataset
    if open_positions and timeline:
        last_ts = timeline[-1]
        for pos in open_positions:
            df_sym = enriched_map[pos.symbol]
            cl = float(df_sym.iloc[-1]["close"])
            rem_qty = pos.tp2_qty if pos.tp1_hit else pos.qty
            rem_gross = risk.leg_gross_pnl(pos.direction, pos.entry_price, cl, rem_qty)
            rem_fee = risk.fee_usdt(rem_qty * cl, is_maker=False)
            total_gross = pos.realized_gross + rem_gross
            total_fees = pos.realized_fees + rem_fee
            net_pnl = total_gross - total_fees
            equity += net_pnl
            ret_pct = (net_pnl / max(pos.equity_at_entry, 1e-9)) * 100.0
            closed_trades.append(
                ClosedTrade(
                    symbol=pos.symbol,
                    direction=pos.direction,
                    period=pos.period,
                    entry_ts=pos.entry_ts,
                    exit_ts=last_ts,
                    entry_price=pos.entry_price,
                    sl_pct=pos.sl_pct,
                    leverage=pos.leverage,
                    score=pos.score,
                    exit_reason="EOD",
                    gross_pnl_usdt=round(total_gross, 6),
                    fees_usdt=round(total_fees, 6),
                    net_pnl_usdt=round(net_pnl, 6),
                    equity_return_pct=round(ret_pct, 6),
                    tp1_hit=pos.tp1_hit,
                )
            )

    is_trades = [t for t in closed_trades if t.period == "IS"]
    oos_trades = [t for t in closed_trades if t.period == "OOS"]

    metrics_is = compute_period_metrics(is_trades, is_days)
    metrics_oos = compute_period_metrics(oos_trades, oos_days)
    metrics_total = compute_period_metrics(closed_trades, days)
    coin_stats = compute_coin_rankings(closed_trades, min_trades_for_top=15)

    elapsed = round(time.time() - t0, 2)
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "days": days,
            "is_days": is_days,
            "oos_days": oos_days,
            "start_utc": datetime.fromtimestamp(sim_start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "split_utc": datetime.fromtimestamp(split_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "end_utc": datetime.fromtimestamp(sim_end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "symbols_tested": active_symbols,
            "initial_equity": initial_equity,
            "final_equity": round(equity, 2),
            "daily_halt_days": daily_halt_days,
            "elapsed_sec": elapsed,
            "config_snapshot": config,
        },
        "summary": {
            "in_sample": metrics_is,
            "out_of_sample": metrics_oos,
            "total": metrics_total,
        },
        "coins": coin_stats,
        "trades": [asdict(t) for t in closed_trades],
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Backtest complete in %.2fs. Saved full report to %s", elapsed, out_p)

    _print_console_summary(report)
    return report


def _print_console_summary(report: Dict[str, Any]) -> None:
    s_is = report["summary"]["in_sample"]
    s_oos = report["summary"]["out_of_sample"]
    s_tot = report["summary"]["total"]

    print("\n" + "=" * 92)
    print(f"BYBIT USDT PERPETUAL BACKTEST SUMMARY ({report['meta']['start_utc']} -> {report['meta']['end_utc']})")
    print("=" * 92)
    header = f"{'Metric':<30} | {'In-Sample (120d)':>18} | {'Out-of-Sample (60d)':>19} | {'Total (180d)':>16}"
    print(header)
    print("-" * 92)
    rows = [
        ("Trades (avg/day)", f"{s_is['trades']} ({s_is['trades_per_day']}/d)", f"{s_oos['trades']} ({s_oos['trades_per_day']}/d)", f"{s_tot['trades']} ({s_tot['trades_per_day']}/d)"),
        ("Win Rate (%)", f"{s_is['win_rate_pct']:.2f}%", f"{s_oos['win_rate_pct']:.2f}%", f"{s_tot['win_rate_pct']:.2f}%"),
        ("  TP1+TP2 (count / %)", f"{s_is['tp1_tp2_count']} ({s_is['tp1_tp2_pct']:.1f}%)", f"{s_oos['tp1_tp2_count']} ({s_oos['tp1_tp2_pct']:.1f}%)", f"{s_tot['tp1_tp2_count']} ({s_tot['tp1_tp2_pct']:.1f}%)"),
        ("  TP1+BE  (count / %)", f"{s_is['tp1_be_count']} ({s_is['tp1_be_pct']:.1f}%)", f"{s_oos['tp1_be_count']} ({s_oos['tp1_be_pct']:.1f}%)", f"{s_tot['tp1_be_count']} ({s_tot['tp1_be_pct']:.1f}%)"),
        ("  SL      (count / %)", f"{s_is['sl_count']} ({s_is['sl_pct']:.1f}%)", f"{s_oos['sl_count']} ({s_oos['sl_pct']:.1f}%)", f"{s_tot['sl_count']} ({s_tot['sl_pct']:.1f}%)"),
        ("Net PnL (%) [after fees]", f"{s_is['net_pnl_pct']:+.2f}%", f"{s_oos['net_pnl_pct']:+.2f}%", f"{s_tot['net_pnl_pct']:+.2f}%"),
        ("Profit Factor", f"{s_is['profit_factor']:.2f}", f"{s_oos['profit_factor']:.2f}", f"{s_tot['profit_factor']:.2f}"),
        ("Max Drawdown (%)", f"{s_is['max_drawdown_pct']:.2f}%", f"{s_oos['max_drawdown_pct']:.2f}%", f"{s_tot['max_drawdown_pct']:.2f}%"),
    ]
    for label, v1, v2, v3 in rows:
        print(f"{label:<30} | {v1:>18} | {v2:>19} | {v3:>16}")
    print("=" * 92)

    coins = report["coins"]
    print(f"\nTOP-10 BEST COINS (min {coins['min_trades_threshold']} trades; qualified: {coins['qualified_count']}):")
    if not coins["top_10_best"]:
        print("  No coins met the >= 15 trades threshold.")
    else:
        print(f"  {'Symbol':<14} {'Trades':>6} {'WinRate':>9} {'TP1+TP2':>8} {'TP1+BE':>7} {'SL':>5} {'Net PnL%':>10} {'PF':>6}")
        for c in coins["top_10_best"]:
            print(
                f"  {c['symbol']:<14} {c['trades']:>6} {c['win_rate_pct']:>8.2f}% "
                f"{c['tp1_tp2']:>8} {c['tp1_be']:>7} {c['sl']:>5} {c['net_pnl_pct']:>+9.2f}% {c['profit_factor']:>6.2f}"
            )

    print("\nWORST 5 COINS:")
    print(f"  {'Symbol':<14} {'Trades':>6} {'WinRate':>9} {'TP1+TP2':>8} {'TP1+BE':>7} {'SL':>5} {'Net PnL%':>10} {'PF':>6}")
    for c in coins["worst_5"]:
        print(
            f"  {c['symbol']:<14} {c['trades']:>6} {c['win_rate_pct']:>8.2f}% "
            f"{c['tp1_tp2']:>8} {c['tp1_be']:>7} {c['sl']:>5} {c['net_pnl_pct']:>+9.2f}% {c['profit_factor']:>6.2f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest Bybit USDT Perpetual Backtester")
    parser.add_argument("--days", type=int, default=180, help="Total historical days (default: 180)")
    parser.add_argument("--top", type=int, default=25, help="Top N pairs by turnover if whitelist empty (default: 25)")
    parser.add_argument("--is-days", type=int, default=120, help="In-Sample days (default: 120)")
    parser.add_argument("--oos-days", type=int, default=60, help="Out-of-Sample days (default: 60)")
    parser.add_argument("--initial-equity", type=float, default=1000.0, help="Starting equity in USDT")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--output", type=str, default="data/backtest_results.json", help="Output JSON report path")
    parser.add_argument("--ignore-whitelist", action="store_true", help="Ignore config whitelist and scan top N")
    args = parser.parse_args()

    run_backtest(
        days=args.days,
        top_n=args.top,
        is_days=args.is_days,
        oos_days=args.oos_days,
        initial_equity=args.initial_equity,
        config_path=args.config,
        output_path=args.output,
        ignore_whitelist=args.ignore_whitelist,
    )


if __name__ == "__main__":
    main()
