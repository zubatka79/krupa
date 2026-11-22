"""100% Blind Walk-Forward Scaling Test for `fibo_trend_v2` (0.618-0.786, TP 1.50%, SL 3.5%, Smart BE).

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

from bot.blind_walkforward import add_fibo_v2_signals, load_cached_15m, slice_sym_data
from bot.strategy_arena import ArenaConfig, compute_all_strategy_signals


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
    entry_ts: int
    entry_price: float
    sl_price: float
    tp_price: float
    be_trigger_price: float
    be_stop_price: float
    qty: float
    margin: float
    notional: float
    entry_fee: float
    be_active: bool = False


def simulate_scaled_10x(
    symbols: List[str],
    sym_data: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    margin_pct: float = 10.0,
    leverage: float = 10.0,
    max_open: int = 3,
    max_same_dir: int = 2,
    max_daily_stops: int = 4,
    tp_pct: float = 1.50,
    sl_pct: float = 3.50,
    be_trigger_ratio: float = 0.65,
    be_offset_pct: float = 0.15,
    maker_pullback_pct: float = 0.065,
    maker_ttl_bars: int = 2,
    cooldown_bars: int = 3,
    initial_deposit: float = 1000.0,
    strat_key: str = "fv2_618",
) -> Dict[str, Any]:
    """
    Isolated-margin 10x portfolio simulator with:
      - Margin = margin_pct % of current equity, Leverage = 10x (notional = equity * margin_pct/100 * 10)
      - Daily stop limit: max `max_daily_stops` SL closures per UTC day (or equivalent daily loss cap)
      - Maker entry (0.02%), Maker TP (0.02%), Taker SL/BE (0.055%)
    """
    maker_rate = 0.0002
    taker_rate = 0.00055
    tp_frac = tp_pct / 100.0
    sl_frac = sl_pct / 100.0
    pb_frac = maker_pullback_pct / 100.0
    be_trig_frac = (tp_pct * be_trigger_ratio) / 100.0
    be_off_frac = be_offset_pct / 100.0
    notional_mult = (margin_pct / 100.0) * leverage
    single_sl_eq_pct = notional_mult * (sl_frac + maker_rate + taker_rate) * 100.0
    daily_dd_cap_pct = max_daily_stops * single_sl_eq_pct

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
    trades: List[Dict[str, Any]] = []

    n_bars = len(timeline)
    cur_day = -1
    day_start_eq = equity
    day_sl_count = 0
    daily_halted = False

    def _close(pos: _OpenPos, exit_p: float, reason: str, exit_maker: bool, ts: int, bar_i: int) -> None:
        nonlocal equity, peak_eq, max_dd_usd, max_dd_pct, day_sl_count, daily_halted
        gross = (
            (exit_p - pos.entry_price) * pos.qty
            if pos.direction == "LONG"
            else (pos.entry_price - exit_p) * pos.qty
        )
        exit_fee = abs(pos.qty * exit_p) * (maker_rate if exit_maker else taker_rate)
        fees = pos.entry_fee + exit_fee
        net = gross - fees
        # Isolated margin protection: loss on one position cannot exceed allocated isolated margin
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

        trades.append(
            {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "reason": reason,
                "net_usd": net,
                "fees_usd": fees,
            }
        )
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
                if pos.be_active:
                    be_hit = (lo <= pos.be_stop_price) if is_long else (hi >= pos.be_stop_price)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if be_hit:
                        _close(pos, pos.be_stop_price, "BE", False, ts, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, ts, i)
                    else:
                        still_open.append(pos)
                else:
                    sl_hit = (lo <= pos.sl_price) if is_long else (hi >= pos.sl_price)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if sl_hit:
                        _close(pos, pos.sl_price, "SL", False, ts, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, ts, i)
                    else:
                        trig = (hi >= pos.be_trigger_price) if is_long else (lo <= pos.be_trigger_price)
                        if trig:
                            pos.be_active = True
                        still_open.append(pos)
            open_pos = still_open

        # 2. Check daily stop limit (4 stops/day or equivalent daily drawdown)
        if not daily_halted:
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
                used_margin = sum(p.margin for p in open_pos)
                if used_margin + margin_req > equity * 0.95:
                    continue

                notional = margin_req * leverage
                entry_p = ord_p.limit_price
                qty = notional / entry_p
                entry_fee = notional * maker_rate

                if is_long:
                    sl_p = entry_p * (1.0 - sl_frac)
                    tp_p = entry_p * (1.0 + tp_frac)
                    be_tr = entry_p * (1.0 + be_trig_frac)
                    be_st = entry_p * (1.0 + be_off_frac)
                else:
                    sl_p = entry_p * (1.0 + sl_frac)
                    tp_p = entry_p * (1.0 - tp_frac)
                    be_tr = entry_p * (1.0 - be_trig_frac)
                    be_st = entry_p * (1.0 - be_off_frac)

                pos = _OpenPos(
                    symbol=ord_p.symbol,
                    direction=ord_p.direction,
                    entry_ts=ts,
                    entry_price=entry_p,
                    sl_price=sl_p,
                    tp_price=tp_p,
                    be_trigger_price=be_tr,
                    be_stop_price=be_st,
                    qty=qty,
                    margin=margin_req,
                    notional=notional,
                    entry_fee=entry_fee,
                )

                # Conservative same-bar SL check on fill bar
                if (lo <= sl_p) if is_long else (hi >= sl_p):
                    _close(pos, sl_p, "SL", False, ts, i)
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
        last_ts = int(timeline[-1])
        for pos in open_pos:
            cl = sym_data[pos.symbol]["close"][-1]
            if not np.isfinite(cl):
                cl = pos.entry_price
            _close(pos, float(cl), "EOD", False, last_ts, n_bars - 1)

    n_tr = len(trades)
    wins = sum(1 for t in trades if t["net_usd"] > 0)
    net_usd = equity - initial_deposit
    growth_2m = equity / initial_deposit
    est_6m_usd = initial_deposit * (growth_2m ** 3)

    return {
        "start_usd": round(initial_deposit, 2),
        "end_usd": round(equity, 2),
        "net_usd": round(net_usd, 2),
        "net_pct": round((net_usd / initial_deposit) * 100.0, 2),
        "est_6m_usd": round(est_6m_usd, 2),
        "trades": n_tr,
        "trades_per_day": round(n_tr / 60.0, 2),
        "win_rate_pct": round((wins / n_tr) * 100.0, 2) if n_tr > 0 else 0.0,
        "tp_count": sum(1 for t in trades if t["reason"] == "TP"),
        "be_count": sum(1 for t in trades if t["reason"] == "BE"),
        "sl_count": sum(1 for t in trades if t["reason"] == "SL"),
        "fees_usd": round(sum(t["fees_usd"] for t in trades), 2),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    arena_meta = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]
    symbols_100 = arena_meta["symbols_100"]

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

    print("Loading cached Top-100 and computing fibo_trend_v2 (0.618-0.786) signals...")
    sym_data: Dict[str, Dict[str, np.ndarray]] = {}
    active_syms: List[str] = []
    for sym in symbols_100:
        p = cache_dir / f"{sym}_15.csv.gz"
        if not p.exists():
            continue
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 400:
            continue
        base_arr = compute_all_strategy_signals(df_s, df_btc, timeline_df)
        full_arr = add_fibo_v2_signals(df_s, df_btc, timeline_df, base_arr)
        sym_data[sym] = full_arr
        active_syms.append(sym)

    train_data = slice_sym_data(sym_data, train_mask)
    test_data = slice_sym_data(sym_data, test_mask)

    # STRICTLY BLIND RANKING ON FIRST 4 MONTHS (120 DAYS) ONLY
    train_ranked: List[Tuple[float, float, int, str]] = []
    for sym in active_syms:
        res_tr = simulate_scaled_10x(
            symbols=[sym],
            sym_data={sym: train_data[sym]},
            timeline=train_tl,
            margin_pct=10.0,
            leverage=10.0,
            max_open=1,
            max_same_dir=1,
            max_daily_stops=4,
        )
        train_ranked.append((res_tr["net_pct"], res_tr["win_rate_pct"], res_tr["trades"], sym))

    train_ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # 4 Scales strictly derived from first 4 months (ZERO peeking at last 60 days!)
    top_25_syms = [x[3] for x in train_ranked[:25]]
    top_50_syms = [x[3] for x in train_ranked[:50]]
    all_positive_syms = [x[3] for x in train_ranked if x[0] > 0.0 and x[2] >= 1]
    all_100_syms = list(active_syms)

    scales = [
        ("top25", f"Топ-25 лучших (по 4 мес.)", top_25_syms),
        ("top50", f"Топ-50 лучших (по 4 мес.)", top_50_syms),
        ("all_pos", f"Все плюсовые за 4 мес. ({len(all_positive_syms)} монет)", all_positive_syms),
        ("all_100", f"Все {len(all_100_syms)} монет без отбора", all_100_syms),
    ]

    risk_combos = [
        (10.0, 3, 2),
        (10.0, 5, 3),
        (15.0, 3, 2),
        (15.0, 5, 3),
    ]

    print(f"Blind Train (4m) stats: {len(all_positive_syms)} / {len(active_syms)} coins finished positive in first 4 months.")
    all_runs: List[Dict[str, Any]] = []

    for scale_id, scale_label, syms in scales:
        for margin_pct, max_open, max_same in risk_combos:
            res = simulate_scaled_10x(
                symbols=syms,
                sym_data=test_data,
                timeline=test_tl,
                margin_pct=margin_pct,
                leverage=10.0,
                max_open=max_open,
                max_same_dir=max_same,
                max_daily_stops=4,
            )
            rec = {
                "scale_id": scale_id,
                "scale_label": scale_label,
                "coins_count": len(syms),
                "margin_pct": margin_pct,
                "max_open": max_open,
                "max_same_dir": max_same,
                **res,
            }
            all_runs.append(rec)

    # Print ALL 16 runs clearly for full transparency, then rank Top-4
    print("\n" + "=" * 108)
    print("ALL 16 BLIND 60-DAY SCALING RUNS (fibo_trend_v2 0.618-0.786 | TP 1.50% / SL 3.5% / Smart BE | 10x)")
    print("=" * 108)
    for r in sorted(all_runs, key=lambda x: x["end_usd"], reverse=True):
        print(
            f"  {r['scale_label']:<36} | M={r['margin_pct']:>2.0f}% Max={r['max_open']}({r['max_same_dir']}) | "
            f"{r['trades']:>3} tr ({r['trades_per_day']:>4.2f}/d) | WR {r['win_rate_pct']:>5.1f}% | "
            f"$1k->${r['end_usd']:>8,.2f} ({r['net_pct']:>+6.1f}%) | 6m~${r['est_6m_usd']:>10,.0f} | "
            f"DD ${r['max_dd_usd']:>6.1f} ({r['max_dd_pct']:>5.1f}%)"
        )

    # Sort Top-4: prioritize high blind profit while keeping drawdown sensible (and show best overall)
    top_4_profit = sorted(all_runs, key=lambda x: (x["end_usd"], -x["max_dd_pct"]), reverse=True)[:4]
    # Also compute risk-adjusted Return/DD ratio for insight
    top_4_calmar = sorted(all_runs, key=lambda x: x["net_pct"] / max(x["max_dd_pct"], 1.0), reverse=True)[:4]

    out = {
        "positive_in_4m_count": len(all_positive_syms),
        "all_16_runs": sorted(all_runs, key=lambda x: x["end_usd"], reverse=True),
        "top_4_by_profit": top_4_profit,
        "top_4_by_return_dd": top_4_calmar,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path("data/blind_scale_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/blind_scale_results.json")


if __name__ == "__main__":
    main()
