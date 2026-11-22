"""STEP 3: Rolling 3-Month Blind Walk-Forward Exam of Blowup Fixes (`blind_blowup_fix.py`).

Fixes derived STRICTLY from Days 1-120 autopsy (Month 5 & Month 6 untouched during design):
  1. btc_regime_shield : No entries against BTC 4H trend + BTC 1D EMA20 bias
  2. anti_cluster_guard: Max 3 open (2 same dir) + strict 2 SL/day limit (cuts 87% cluster damage)
  3. liquidity_top25   : Trade Top-25 pure crypto by Bybit liquidity instead of chasing 90d PnL leaders
  4. strict_h1_adx30   : Raise H1 ADX threshold from 25 to 30 (keep H4 ADX >= 20)
  5. balanced_rr       : Healthier TP 1.50% / SL 2.50% / NoBE (breakeven WR ~63.8% vs ~80%)

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_final_tuning import build_tuning_bundle
from bot.blind_walkforward import load_cached_15m, slice_sym_data
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, ema


@dataclass
class FixConfig:
    name: str
    btc_regime_shield: bool = False
    anti_cluster_guard: bool = False  # max_open=3, max_same_dir=2, max_daily_stops=2
    liquidity_top25: bool = False     # use Top-25 by Bybit turnover instead of 90d PnL ranking
    h1_adx_min: float = 25.0
    tp_pct: float = 1.50
    sl_pct: float = 3.50
    smart_be: bool = True
    margin_pct: float = 15.0


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


def build_btc_regime_arrays(df_btc_15m: pd.DataFrame, timeline: np.ndarray) -> Dict[str, np.ndarray]:
    """Strictly causal BTC 4H & 1D trend alignment flags aligned onto timeline."""
    btc_4h = BybitClient.resample_klines(df_btc_15m, "240")
    btc_1d = BybitClient.resample_klines(df_btc_15m, "D")

    b4_e20, b4_e50 = ema(btc_4h["close"], 20), ema(btc_4h["close"], 50)
    btc_4h_df = pd.DataFrame({
        "close_ts": (btc_4h["timestamp"] + INTERVAL_TO_MS["240"]).astype("int64"),
        "btc_4h_strict_bear": (btc_4h["close"] < b4_e50) & (b4_e20 < b4_e50),
        "btc_4h_strict_bull": (btc_4h["close"] > b4_e50) & (b4_e20 > b4_e50),
    }).sort_values("close_ts")

    bd_e20 = ema(btc_1d["close"], 20)
    btc_1d_df = pd.DataFrame({
        "close_ts": (btc_1d["timestamp"] + INTERVAL_TO_MS["D"]).astype("int64"),
        "btc_1d_bull": btc_1d["close"] > bd_e20,
    }).sort_values("close_ts")

    ctx = pd.DataFrame({"timestamp": timeline, "close_ts": timeline + INTERVAL_TO_MS["15"]})
    ctx = pd.merge_asof(ctx, btc_4h_df, on="close_ts", direction="backward")
    ctx = pd.merge_asof(ctx, btc_1d_df, on="close_ts", direction="backward")

    strict_bear = ctx["btc_4h_strict_bear"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    strict_bull = ctx["btc_4h_strict_bull"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    d1_bull = ctx["btc_1d_bull"].astype("boolean").fillna(True).to_numpy(dtype=bool)

    return {
        "allow_long": (~strict_bear) & d1_bull,
        "allow_short": (~strict_bull) & (~d1_bull),
    }


def simulate_window_fix(
    symbols: List[str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    btc_regime: Dict[str, np.ndarray],
    timeline: np.ndarray,
    cfg: FixConfig,
    single_coin_mode: bool = False,
    initial_deposit: float = 1000.0,
) -> Dict[str, Any]:
    max_open = 1 if single_coin_mode else (3 if cfg.anti_cluster_guard else 5)
    max_same_dir = 1 if single_coin_mode else (2 if cfg.anti_cluster_guard else 3)
    max_daily_stops = 4 if single_coin_mode else (2 if cfg.anti_cluster_guard else 4)

    maker_rate = 0.0002
    taker_rate = 0.00055
    tp_frac = cfg.tp_pct / 100.0
    sl_frac = cfg.sl_pct / 100.0
    pb_frac = 0.00065
    be_trig_frac = (cfg.tp_pct * 0.65) / 100.0
    be_off_frac = 0.0015
    leverage = 10.0
    notional_mult = (cfg.margin_pct / 100.0) * leverage
    daily_dd_cap_pct = max_daily_stops * notional_mult * (sl_frac + maker_rate + taker_rate) * 100.0

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    open_pos: List[_OpenPos] = []
    pending: List[_Pending] = []
    cooldown_until: Dict[str, int] = {}
    trades: List[Tuple[str, float]] = []

    allow_long = btc_regime["allow_long"]
    allow_short = btc_regime["allow_short"]

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
        cooldown_until[pos.symbol] = bar_i + 3

    for i in range(n_bars):
        ts = int(timeline[i])
        day_key = ts // 86_400_000
        if day_key != cur_day:
            cur_day = day_key
            day_start_eq = equity
            day_sl_count = 0
            daily_halted = False

        if open_pos:
            still_open: List[_OpenPos] = []
            for pos in open_pos:
                sd = bundle[pos.symbol]
                hi = sd["high"][i]
                lo = sd["low"][i]
                if not np.isfinite(lo):
                    still_open.append(pos)
                    continue

                is_long = pos.direction == "LONG"
                if cfg.smart_be and pos.be_active and pos.be_stop_price is not None:
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
                        if cfg.smart_be and pos.be_trigger_price is not None:
                            if (hi >= pos.be_trigger_price) if is_long else (lo <= pos.be_trigger_price):
                                pos.be_active = True
                        still_open.append(pos)
            open_pos = still_open

        if not daily_halted and open_pos:
            unreal = 0.0
            for pos in open_pos:
                cl = bundle[pos.symbol]["close"][i]
                if np.isfinite(cl):
                    g = (cl - pos.entry_price) * pos.qty if pos.direction == "LONG" else (pos.entry_price - cl) * pos.qty
                    unreal += g - pos.entry_fee
            dd_day = ((day_start_eq - (equity + unreal)) / max(day_start_eq, 1e-9)) * 100.0
            if dd_day >= daily_dd_cap_pct or day_sl_count >= max_daily_stops:
                daily_halted = True

        if pending:
            next_pend: List[_Pending] = []
            for ord_p in pending:
                if i > ord_p.expires_idx or daily_halted:
                    continue
                if any(p.symbol == ord_p.symbol for p in open_pos):
                    continue
                if i < cooldown_until.get(ord_p.symbol, -1):
                    continue

                sd = bundle[ord_p.symbol]
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

                margin_req = equity * (cfg.margin_pct / 100.0)
                if sum(p.margin for p in open_pos) + margin_req > equity * 0.95:
                    continue

                notional = margin_req * leverage
                entry_p = ord_p.limit_price
                qty = notional / entry_p
                entry_fee = notional * maker_rate

                if is_long:
                    sl_p = entry_p * (1.0 - sl_frac)
                    tp_p = entry_p * (1.0 + tp_frac)
                    be_tr = entry_p * (1.0 + be_trig_frac) if cfg.smart_be else None
                    be_st = entry_p * (1.0 + be_off_frac) if cfg.smart_be else None
                else:
                    sl_p = entry_p * (1.0 + sl_frac)
                    tp_p = entry_p * (1.0 - tp_frac)
                    be_tr = entry_p * (1.0 - be_trig_frac) if cfg.smart_be else None
                    be_st = entry_p * (1.0 - be_off_frac) if cfg.smart_be else None

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

        busy = {p.symbol for p in open_pos} | {o.symbol for o in pending}
        cands: List[Tuple[float, str, str, float]] = []
        can_L = (not cfg.btc_regime_shield) or bool(allow_long[i])
        can_S = (not cfg.btc_regime_shield) or bool(allow_short[i])

        for sym in symbols:
            if sym in busy or i < cooldown_until.get(sym, -1):
                continue
            sd = bundle[sym]
            if sd["h1_adx"][i] < cfg.h1_adx_min or sd["h4_adx"][i] < 20.0 or sd["atr_pct"][i] > 1.50:
                continue
            if can_L and sd["raw_L"][i]:
                cands.append((float(sd["q"][i]), sym, "LONG", float(sd["close"][i])))
            elif can_S and sd["raw_S"][i]:
                cands.append((float(sd["q"][i]), sym, "SHORT", float(sd["close"][i])))

        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)

        for _, sym, direction, cl_p in cands:
            if len(open_pos) + len(pending) >= max_open + 3:
                break
            if sum(1 for p in open_pos if p.direction == direction) >= max_same_dir:
                continue
            limit_p = cl_p * (1.0 - pb_frac) if direction == "LONG" else cl_p * (1.0 + pb_frac)
            pending.append(_Pending(symbol=sym, direction=direction, limit_price=limit_p, expires_idx=i + 2))

    if open_pos and n_bars > 0:
        for pos in open_pos:
            cl = bundle[pos.symbol]["close"][-1]
            if not np.isfinite(cl):
                cl = pos.entry_price
            _close(pos, float(cl), "EOD", False, n_bars - 1)

    n_tr = len(trades)
    wins = sum(1 for _, net in trades if net > 0)
    net_usd = equity - initial_deposit
    return {
        "end_usd": round(equity, 2),
        "net_usd": round(net_usd, 2),
        "net_pct": round((net_usd / initial_deposit) * 100.0, 2),
        "trades": n_tr,
        "win_rate_pct": round((wins / n_tr) * 100.0, 2) if n_tr > 0 else 0.0,
        "tp_count": sum(1 for r, _ in trades if r == "TP"),
        "be_count": sum(1 for r, _ in trades if r == "BE"),
        "sl_count": sum(1 for r, _ in trades if r == "SL"),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
    }


def eval_rolling_3m(
    cfg: FixConfig,
    active_syms: List[str],
    full_bundle: Dict[str, Dict[str, np.ndarray]],
    btc_regime_full: Dict[str, np.ndarray],
    timeline: np.ndarray,
    sim_start_ms: int,
    sim_end_ms: int,
) -> Dict[str, Any]:
    """
    Rolling 3-Month Exam (each month trades Top-25 selected strictly from prior 90 days):
      - M4: Train d1-90   -> Trade d91-120
      - M5: Train d31-120 -> BLIND d121-150
      - M6: Train d61-150 -> BLIND d151-180
    Compounds $1,000 continuously across M4 -> M5 -> M6!
    """
    day_ms = 86_400_000
    windows = [
        ("M4", sim_start_ms + 0 * day_ms, sim_start_ms + 90 * day_ms, sim_start_ms + 90 * day_ms, sim_start_ms + 120 * day_ms),
        ("M5", sim_start_ms + 30 * day_ms, sim_start_ms + 120 * day_ms, sim_start_ms + 120 * day_ms, sim_start_ms + 150 * day_ms),
        ("M6", sim_start_ms + 60 * day_ms, sim_start_ms + 150 * day_ms, sim_start_ms + 150 * day_ms, sim_end_ms),
    ]

    eq = 1000.0
    peak_global = 1000.0
    global_max_dd_pct = 0.0
    m_res: List[Dict[str, Any]] = []
    tot_tr = tot_wins = tot_tp = tot_be = tot_sl = 0

    for label, tr_s, tr_e, te_s, te_e in windows:
        tr_mask = (timeline >= tr_s) & (timeline < tr_e)
        te_mask = (timeline >= te_s) & (timeline <= te_e)
        tr_b = slice_sym_data(full_bundle, tr_mask)
        te_b = slice_sym_data(full_bundle, te_mask)
        tr_btc = {k: v[tr_mask] for k, v in btc_regime_full.items()}
        te_btc = {k: v[te_mask] for k, v in btc_regime_full.items()}
        tr_tl = timeline[tr_mask]
        te_tl = timeline[te_mask]

        if cfg.liquidity_top25:
            selected = active_syms[:25]
        else:
            ranked: List[Tuple[float, float, int, str]] = []
            for sym in active_syms:
                r = simulate_window_fix([sym], {sym: tr_b[sym]}, tr_btc, tr_tl, cfg, single_coin_mode=True)
                ranked.append((r["net_pct"], r["win_rate_pct"], r["trades"], sym))
            ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
            selected = [x[3] for x in ranked[:25]]

        r_m = simulate_window_fix(selected, te_b, te_btc, te_tl, cfg, single_coin_mode=False, initial_deposit=eq)
        month_pct = round(((r_m["end_usd"] - eq) / max(eq, 1.0)) * 100.0, 2)
        eq = r_m["end_usd"]
        if eq > peak_global:
            peak_global = eq
        global_max_dd_pct = max(global_max_dd_pct, r_m["max_dd_pct"], ((peak_global - eq) / peak_global) * 100.0)

        tot_tr += r_m["trades"]
        tot_tp += r_m["tp_count"]
        tot_be += r_m["be_count"]
        tot_sl += r_m["sl_count"]
        tot_wins += int(round(r_m["trades"] * r_m["win_rate_pct"] / 100.0))
        m_res.append({"month": label, "net_pct": month_pct, "end_usd": eq, "trades": r_m["trades"], "sl": r_m["sl_count"], "dd_pct": r_m["max_dd_pct"]})

    return {
        "name": cfg.name,
        "margin_pct": cfg.margin_pct,
        "m4_pct": m_res[0]["net_pct"],
        "m5_pct": m_res[1]["net_pct"],
        "m6_pct": m_res[2]["net_pct"],
        "end_usd": round(eq, 2),
        "total_net_pct": round((eq - 1000.0) / 10.0, 2),
        "trades": tot_tr,
        "trades_per_day": round(tot_tr / 90.0, 2),
        "win_rate_pct": round((tot_wins / tot_tr) * 100.0, 2) if tot_tr > 0 else 0.0,
        "tp_count": tot_tp,
        "be_count": tot_be,
        "sl_count": tot_sl,
        "max_dd_pct": round(global_max_dd_pct, 2),
        "all_3_positive": all(m["net_pct"] > 0.0 for m in m_res),
        "worst_month_pct": min(m["net_pct"] for m in m_res),
        "months": m_res,
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    symbols_100 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    sim_end_ms = int(df_btc["timestamp"].iloc[-1])
    sim_start_ms = sim_end_ms - 180 * 86_400_000

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]
    timeline = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    timeline_df = pd.DataFrame({"timestamp": timeline})
    btc_regime_full = build_btc_regime_arrays(df_btc, timeline)

    print("Loading cached Top-100 bundles...")
    bundle: Dict[str, Dict[str, np.ndarray]] = {}
    active_syms: List[str] = []
    for sym in symbols_100:
        if not (cache_dir / f"{sym}_15.csv.gz").exists():
            continue
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 400:
            continue
        bundle[sym] = build_tuning_bundle(df_s, df_btc, timeline_df)
        active_syms.append(sym)

    # 0. Exact Baseline check
    base_cfg = FixConfig("БАЗА (H1 ADX>=25, 15%x10x, 5 сл., 4 SL/д, TP1.5/SL3.5/BE)")
    base_res = eval_rolling_3m(base_cfg, active_syms, bundle, btc_regime_full, timeline, sim_start_ms, sim_end_ms)

    # 1. Individual Fixes (strictly one at a time at baseline 15% margin)
    individual_cfgs = [
        ("btc_regime_shield", FixConfig("1. Фильтр фазы BTC (4H + 1D EMA20)", btc_regime_shield=True)),
        ("anti_cluster_guard", FixConfig("2. Щит от групповых стопов (макс 3 сд / 2 в сторону / 2 SL в день)", anti_cluster_guard=True)),
        ("liquidity_top25", FixConfig("3. Ликвидные Топ-25 Bybit (вместо отбора по 90д PnL)", liquidity_top25=True)),
        ("strict_h1_adx30", FixConfig("4. Строгий тренд H1 ADX >= 30 (вместо 25)", h1_adx_min=30.0)),
        ("balanced_rr", FixConfig("5. Здоровый TP/SL: TP 1.50% / SL 2.50% / NoBE", tp_pct=1.50, sl_pct=2.50, smart_be=False)),
    ]

    print("\n" + "=" * 110)
    print(
        f"BASELINE 3M ROLLING: M4={base_res['m4_pct']:+.1f}% | M5={base_res['m5_pct']:+.1f}% | M6={base_res['m6_pct']:+.1f}% "
        f"=> $1,000 -> ${base_res['end_usd']:,.2f} | WR={base_res['win_rate_pct']:.1f}% | SL={base_res['sl_count']} | DD={base_res['max_dd_pct']:.1f}%"
    )
    print("=" * 110)

    singles: List[Tuple[str, Dict[str, Any]]] = []
    for key, cfg in individual_cfgs:
        r = eval_rolling_3m(cfg, active_syms, bundle, btc_regime_full, timeline, sim_start_ms, sim_end_ms)
        singles.append((key, r))
        print(
            f"\n>>> {r['name']}\n"
            f"  M4: {r['m4_pct']:>+7.2f}% | M5(blind): {r['m5_pct']:>+7.2f}% | M6(blind): {r['m6_pct']:>+7.2f}%\n"
            f"  3m Total: $1,000 -> ${r['end_usd']:,.2f} | {r['trades']} tr ({r['trades_per_day']:.2f}/d) | WR={r['win_rate_pct']:.1f}% | SL={r['sl_count']} | MaxDD={r['max_dd_pct']:.1f}%"
        )

    # 2. Test clean logical combinations of the strongest fixes (all 2-way & 3-way subsets of top fixes!)
    # Select fixes that improved over baseline ($631) or notably reduced M4/M6 blowup
    good_keys = [k for k, r in singles if r["end_usd"] > base_res["end_usd"] or r["worst_month_pct"] > base_res["worst_month_pct"] + 15.0]
    if len(good_keys) < 2:
        good_keys = [k for k, _ in sorted(singles, key=lambda x: x[1]["end_usd"], reverse=True)[:4]]

    combo_runs: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = []
    for r_len in (2, 3, 4):
        if r_len > len(good_keys):
            break
        for comb in combinations(good_keys, r_len):
            c_set = set(comb)
            cfg = FixConfig(
                name=" + ".join(comb),
                btc_regime_shield=("btc_regime_shield" in c_set),
                anti_cluster_guard=("anti_cluster_guard" in c_set),
                liquidity_top25=("liquidity_top25" in c_set),
                h1_adx_min=30.0 if ("strict_h1_adx30" in c_set) else 25.0,
                tp_pct=1.50,
                sl_pct=2.50 if ("balanced_rr" in c_set) else 3.50,
                smart_be=not ("balanced_rr" in c_set),
                margin_pct=15.0,
            )
            r = eval_rolling_3m(cfg, active_syms, bundle, btc_regime_full, timeline, sim_start_ms, sim_end_ms)
            combo_runs.append((comb, r))

    # Rank combinations by: (all 3 months positive, highest end_usd, least bad worst_month, lowest DD)
    combo_runs.sort(
        key=lambda x: (
            1.0 if x[1]["all_3_positive"] else 0.0,
            1.0 if x[1]["m6_pct"] > 0 else 0.0,
            x[1]["end_usd"],
            -x[1]["max_dd_pct"],
        ),
        reverse=True,
    )

    print("\n" + "=" * 110)
    print(f"TOP-5 COMBINATIONS ON ROLLING 3-MONTH EXAM (15% margin x 10x | candidates={good_keys})")
    print("=" * 110)
    for rank, (comb, r) in enumerate(combo_runs[:5], 1):
        flag = "ALL 3 POS!" if r["all_3_positive"] else "MIXED"
        print(
            f"  #{rank} [{flag:<10}] {r['name']:<65} | M4={r['m4_pct']:>+6.1f}% M5={r['m5_pct']:>+6.1f}% M6={r['m6_pct']:>+6.1f}% "
            f"=> ${r['end_usd']:>8,.2f} | {r['trades']:>3}tr WR={r['win_rate_pct']:>5.1f}% SL={r['sl_count']:>2} DD={r['max_dd_pct']:>5.1f}%"
        )

    # Evaluate the #1 best combination at BOTH 10% margin and 15% margin (10x leverage)
    best_comb, best_15 = combo_runs[0]
    b_set = set(best_comb)
    best_cfg_10 = FixConfig(
        name=best_15["name"] + " [Margin 10%]",
        btc_regime_shield=("btc_regime_shield" in b_set),
        anti_cluster_guard=("anti_cluster_guard" in b_set),
        liquidity_top25=("liquidity_top25" in b_set),
        h1_adx_min=30.0 if ("strict_h1_adx30" in b_set) else 25.0,
        tp_pct=1.50,
        sl_pct=2.50 if ("balanced_rr" in b_set) else 3.50,
        smart_be=not ("balanced_rr" in b_set),
        margin_pct=10.0,
    )
    best_10 = eval_rolling_3m(best_cfg_10, active_syms, bundle, btc_regime_full, timeline, sim_start_ms, sim_end_ms)

    print("\n" + "=" * 110)
    print("BEST COMBINATION: 10% MARGIN vs 15% MARGIN (STRICTLY 10x LEVERAGE)")
    print("=" * 110)
    for label, r in (("Margin 10% (10x)", best_10), ("Margin 15% (10x)", best_15)):
        print(
            f"  {label}: {r['name']}\n"
            f"    Trades: {r['trades']} ({r['trades_per_day']:.2f}/d) | WR: {r['win_rate_pct']:.2f}% | SL: {r['sl_count']} (TP={r['tp_count']}, BE={r['be_count']})\n"
            f"    M4: {r['m4_pct']:+.2f}% | M5(blind): {r['m5_pct']:+.2f}% | M6(blind): {r['m6_pct']:+.2f}% => $1,000 -> ${r['end_usd']:,.2f} | Max DD: {r['max_dd_pct']:.2f}%"
        )

    out = {
        "baseline": base_res,
        "singles": [{"key": k, **r} for k, r in singles],
        "top_combos_15": [{"keys": list(c), **r} for c, r in combo_runs[:5]],
        "best_combo_10": best_10,
        "best_combo_15": best_15,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path("data/blind_blowup_fix_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/blind_blowup_fix_results.json")


if __name__ == "__main__":
    main()
