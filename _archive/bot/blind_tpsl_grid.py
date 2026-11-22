"""100% Blind Walk-Forward TP/SL/BE Grid for `fibo_trend_v2` (0.618-0.786, ADX>=20, M=15%, 10x, Max 5).

Strictly offline using cached Top-100 pure crypto data in data/cache/.
Does NOT download data and does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_adx_test import build_adx_signals_for_symbol
from bot.blind_walkforward import load_cached_15m, slice_sym_data


@dataclass
class _Pending:
    symbol: str
    direction: str
    limit_price: float
    expires_idx: int


@dataclass
class _OpenPos:
    symbol: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    be_trigger_price: Optional[float]
    be_stop_price: Optional[float]
    qty: float
    margin: float
    entry_fee: float
    be_active: bool = False


def simulate_combo(
    symbols: List[str],
    sym_data: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    tp_pct: float,
    sl_pct: float,
    smart_be: bool,
    margin_pct: float = 15.0,
    leverage: float = 10.0,
    max_open: int = 5,
    max_same_dir: int = 3,
    max_daily_stops: int = 4,
    be_trigger_ratio: float = 0.65,
    be_offset_pct: float = 0.15,
    maker_pullback_pct: float = 0.065,
    maker_ttl_bars: int = 2,
    cooldown_bars: int = 3,
    initial_deposit: float = 1000.0,
    strat_key: str = "adx20",
) -> Dict[str, Any]:
    maker_rate = 0.0002
    taker_rate = 0.00055
    tp_frac = tp_pct / 100.0
    sl_frac = sl_pct / 100.0
    pb_frac = maker_pullback_pct / 100.0
    be_trig_frac = (tp_pct * be_trigger_ratio) / 100.0
    be_off_frac = be_offset_pct / 100.0
    notional_mult = (margin_pct / 100.0) * leverage
    daily_dd_cap_pct = max_daily_stops * notional_mult * (sl_frac + maker_rate + taker_rate) * 100.0

    l_key = f"{strat_key}_L"
    s_key = f"{strat_key}_S"
    q_key = f"{strat_key}_Q"

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    open_pos: List[_OpenPos] = []
    pending: List[_Pending] = []
    cooldown_until: Dict[str, int] = {}
    trades: List[Tuple[str, float]] = []

    n_bars = len(timeline)
    cur_day = -1
    day_start_eq = equity
    day_sl_count = 0
    daily_halted = False

    def _close(pos: _OpenPos, exit_p: float, reason: str, exit_maker: bool, bar_i: int) -> None:
        nonlocal equity, peak_eq, max_dd_usd, max_dd_pct, day_sl_count, daily_halted
        gross = (
            (exit_p - pos.entry_price) * pos.qty
            if pos.direction == "LONG"
            else (pos.entry_price - exit_p) * pos.qty
        )
        exit_fee = abs(pos.qty * exit_p) * (maker_rate if exit_maker else taker_rate)
        net = gross - (pos.entry_fee + exit_fee)
        if net < -pos.margin:
            net = -pos.margin
        equity = max(1.0, equity + net)

        if reason == "SL":
            day_sl_count += 1
            if day_sl_count >= max_daily_stops:
                daily_halted = True

        if equity > peak_eq:
            peak_eq = equity
        dd_u = peak_eq - equity
        dd_p = (dd_u / peak_eq) * 100.0 if peak_eq > 0 else 0.0
        if dd_u > max_dd_usd:
            max_dd_usd = dd_u
        if dd_p > max_dd_pct:
            max_dd_pct = dd_p

        trades.append((reason, net))
        cooldown_until[pos.symbol] = bar_i + cooldown_bars

    for i in range(n_bars):
        ts = int(timeline[i])
        day_key = ts // 86_400_000
        if day_key != cur_day:
            cur_day = day_key
            day_start_eq = equity
            day_sl_count = 0
            daily_halted = False

        # 1. Manage open positions (conservative intra-bar priority)
        if open_pos:
            still_open: List[_OpenPos] = []
            for pos in open_pos:
                sd = sym_data[pos.symbol]
                hi = sd["high"][i]
                lo = sd["low"][i]
                if not np.isfinite(lo):
                    still_open.append(pos)
                    continue

                is_long = pos.direction == "LONG"
                if smart_be and pos.be_active and pos.be_stop_price is not None:
                    be_hit = (lo <= pos.be_stop_price) if is_long else (hi >= pos.be_stop_price)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if be_hit:
                        _close(pos, pos.be_stop_price, "BE", False, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, i)
                    else:
                        still_open.append(pos)
                else:
                    sl_hit = (lo <= pos.sl_price) if is_long else (hi >= pos.sl_price)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if sl_hit:
                        _close(pos, pos.sl_price, "SL", False, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, i)
                    else:
                        if smart_be and pos.be_trigger_price is not None:
                            if (hi >= pos.be_trigger_price) if is_long else (lo <= pos.be_trigger_price):
                                pos.be_active = True
                        still_open.append(pos)
            open_pos = still_open

        # 2. Check daily 4-stop limit
        if not daily_halted and open_pos:
            unreal = 0.0
            for pos in open_pos:
                cl = sym_data[pos.symbol]["close"][i]
                if np.isfinite(cl):
                    g = (cl - pos.entry_price) * pos.qty if pos.direction == "LONG" else (pos.entry_price - cl) * pos.qty
                    unreal += g - pos.entry_fee
            dd_day = ((day_start_eq - (equity + unreal)) / max(day_start_eq, 1e-9)) * 100.0
            if dd_day >= daily_dd_cap_pct or day_sl_count >= max_daily_stops:
                daily_halted = True

        # 3. Process pending Maker limit entries (TTL 2 bars)
        if pending:
            next_pend: List[_Pending] = []
            for ord_p in pending:
                if i > ord_p.expires_idx or daily_halted:
                    continue
                if any(p.symbol == ord_p.symbol for p in open_pos):
                    continue
                if i < cooldown_until.get(ord_p.symbol, -1):
                    continue

                sd = sym_data[ord_p.symbol]
                hi = sd["high"][i]
                lo = sd["low"][i]
                if not np.isfinite(lo):
                    if i < ord_p.expires_idx:
                        next_pend.append(ord_p)
                    continue

                is_long = ord_p.direction == "LONG"
                filled = (lo <= ord_p.limit_price) if is_long else (hi >= ord_p.limit_price)
                if not filled:
                    if i < ord_p.expires_idx:
                        next_pend.append(ord_p)
                    continue

                if len(open_pos) >= max_open:
                    continue
                if sum(1 for p in open_pos if p.direction == ord_p.direction) >= max_same_dir:
                    continue

                margin_req = equity * (margin_pct / 100.0)
                if sum(p.margin for p in open_pos) + margin_req > equity * 0.95:
                    continue

                notional = margin_req * leverage
                entry_p = ord_p.limit_price
                qty = notional / entry_p
                entry_fee = notional * maker_rate

                if is_long:
                    sl_p = entry_p * (1.0 - sl_frac)
                    tp_p = entry_p * (1.0 + tp_frac)
                    be_tr = entry_p * (1.0 + be_trig_frac) if smart_be else None
                    be_st = entry_p * (1.0 + be_off_frac) if smart_be else None
                else:
                    sl_p = entry_p * (1.0 + sl_frac)
                    tp_p = entry_p * (1.0 - tp_frac)
                    be_tr = entry_p * (1.0 - be_trig_frac) if smart_be else None
                    be_st = entry_p * (1.0 - be_off_frac) if smart_be else None

                pos = _OpenPos(
                    symbol=ord_p.symbol,
                    direction=ord_p.direction,
                    entry_price=entry_p,
                    sl_price=sl_p,
                    tp_price=tp_p,
                    be_trigger_price=be_tr,
                    be_stop_price=be_st,
                    qty=qty,
                    margin=margin_req,
                    entry_fee=entry_fee,
                )

                if (lo <= sl_p) if is_long else (hi >= sl_p):
                    _close(pos, sl_p, "SL", False, i)
                else:
                    open_pos.append(pos)
            pending = next_pend

        if daily_halted or len(open_pos) >= max_open:
            continue

        # 4. Scan closed 15M bar for new signals
        busy = {p.symbol for p in open_pos} | {o.symbol for o in pending}
        cands: List[Tuple[float, str, str, float]] = []
        for sym in symbols:
            if sym in busy or i < cooldown_until.get(sym, -1):
                continue
            sd = sym_data[sym]
            if sd[l_key][i]:
                cands.append((float(sd[q_key][i]), sym, "LONG", float(sd["close"][i])))
            elif sd[s_key][i]:
                cands.append((float(sd[q_key][i]), sym, "SHORT", float(sd["close"][i])))

        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)

        for _, sym, direction, cl_p in cands:
            if len(open_pos) + len(pending) >= max_open + 3:
                break
            if sum(1 for p in open_pos if p.direction == direction) >= max_same_dir:
                continue
            limit_p = cl_p * (1.0 - pb_frac) if direction == "LONG" else cl_p * (1.0 + pb_frac)
            pending.append(_Pending(symbol=sym, direction=direction, limit_price=limit_p, expires_idx=i + maker_ttl_bars))

    # Close remaining open positions at end of 60d window
    if open_pos and n_bars > 0:
        for pos in open_pos:
            cl = sym_data[pos.symbol]["close"][-1]
            if not np.isfinite(cl):
                cl = pos.entry_price
            _close(pos, float(cl), "EOD", False, n_bars - 1)

    n_tr = len(trades)
    wins = sum(1 for _, net in trades if net > 0)
    net_usd = equity - initial_deposit
    growth_2m = equity / initial_deposit
    est_6m_usd = initial_deposit * (growth_2m ** 3)

    return {
        "end_usd": round(equity, 2),
        "net_usd": round(net_usd, 2),
        "net_pct": round((net_usd / initial_deposit) * 100.0, 2),
        "est_6m_usd": round(est_6m_usd, 2),
        "trades": n_tr,
        "trades_per_day": round(n_tr / 60.0, 2),
        "win_rate_pct": round((wins / n_tr) * 100.0, 2) if n_tr > 0 else 0.0,
        "tp_count": sum(1 for r, _ in trades if r == "TP"),
        "be_count": sum(1 for r, _ in trades if r == "BE"),
        "sl_count": sum(1 for r, _ in trades if r == "SL"),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    symbols_100 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    sim_end_ms = int(df_btc["timestamp"].iloc[-1])
    sim_start_ms = sim_end_ms - 180 * 86_400_000
    split_ms = sim_start_ms + 120 * 86_400_000

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]
    timeline = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    train_mask = timeline < split_ms
    test_mask = timeline >= split_ms
    train_tl = timeline[train_mask]
    test_tl = timeline[test_mask]

    print("Loading cached Top-100 and building fibo_trend_v2 (0.618-0.786, ADX>=20) signals...")
    sym_data: Dict[str, Dict[str, np.ndarray]] = {}
    active_syms: List[str] = []
    for sym in symbols_100:
        if not (cache_dir / f"{sym}_15.csv.gz").exists():
            continue
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 400:
            continue
        sym_data[sym] = build_adx_signals_for_symbol(df_s, df_btc, timeline_df)
        active_syms.append(sym)

    train_data = slice_sym_data(sym_data, train_mask)
    test_data = slice_sym_data(sym_data, test_mask)

    results: List[Dict[str, Any]] = []
    for sl in (2.0, 2.5, 3.0, 3.5):
        for tp in (1.50, 1.75, 2.00):
            for smart_be in (True, False):
                # 1. STRICTLY BLIND Top-25 selection on first 4 months (120d)
                ranked: List[Tuple[float, float, int, str]] = []
                for sym in active_syms:
                    r_tr = simulate_combo(
                        symbols=[sym],
                        sym_data={sym: train_data[sym]},
                        timeline=train_tl,
                        tp_pct=tp,
                        sl_pct=sl,
                        smart_be=smart_be,
                        margin_pct=15.0,
                        max_open=1,
                        max_same_dir=1,
                        max_daily_stops=4,
                    )
                    ranked.append((r_tr["net_pct"], r_tr["win_rate_pct"], r_tr["trades"], sym))
                ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
                top_25 = [x[3] for x in ranked[:25]]

                # 2. BLIND 60-day portfolio test on those Top-25 starting at $1,000
                res = simulate_combo(
                    symbols=top_25,
                    sym_data=test_data,
                    timeline=test_tl,
                    tp_pct=tp,
                    sl_pct=sl,
                    smart_be=smart_be,
                    margin_pct=15.0,
                    max_open=5,
                    max_same_dir=3,
                    max_daily_stops=4,
                )
                be_str = "Smart BE (+0.15% @ 65%)" if smart_be else "NoBE"
                results.append(
                    {
                        "label": f"TP {tp:.2f}% / SL {sl:.1f}% / {be_str}",
                        "tp_pct": tp,
                        "sl_pct": sl,
                        "smart_be": smart_be,
                        **res,
                    }
                )

    results.sort(key=lambda x: (x["end_usd"], -x["max_dd_pct"]), reverse=True)

    print("\n" + "=" * 110)
    print("ALL 24 BLIND TP/SL/BE COMBINATIONS (Top-25 by 4m | 60d BLIND | M=15% 10x | Max 5(3/dir) | 4 stops/d)")
    print("=" * 110)
    for idx, r in enumerate(results, 1):
        print(
            f"  #{idx:>2} {r['label']:<42} | {r['trades']:>3} tr ({r['trades_per_day']:>4.2f}/d) | "
            f"WR {r['win_rate_pct']:>5.1f}% | $1k->${r['end_usd']:>8,.2f} ({r['net_pct']:>+6.1f}%) | "
            f"6m~${r['est_6m_usd']:>10,.0f} | DD ${r['max_dd_usd']:>6.1f} ({r['max_dd_pct']:>5.1f}%)"
        )

    Path("data/blind_tpsl_grid_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/blind_tpsl_grid_results.json")


if __name__ == "__main__":
    main()
