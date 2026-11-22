"""Double-Exam Final Tuning for `fibo_trend_v2_adx25`:
  1. Main 2-month Blind Walk-Forward (Train d1-120 -> Blind d121-180) across 4 tuning blocks.
  2. Rolling 3-Month Walk-Forward Anti-Overfitting Exam (M4: d91-120, M5: d121-150, M6: d151-180)
     for Champion (`H1 ADX >= 25`) + Top-3 candidates.

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
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

from bot.blind_walkforward import load_cached_15m, slice_sym_data
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, compute_rsi, ema


@dataclass
class TuningSpec:
    block: str
    name: str
    h1_adx_min: float = 25.0
    h1_adx_max: float = 999.0
    h4_adx_min: float = 20.0
    top_k: int = 25
    sl_cooldown_bars: int = 3
    max_atr_pct: float = 1.50
    quality_select: bool = False  # rank by PF * trades with WR >= 80%


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
    be_trigger_price: float
    be_stop_price: float
    qty: float
    margin: float
    entry_fee: float
    be_active: bool = False


def build_tuning_bundle(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    timeline_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Precompute raw signal base (without ADX & max_atr cutoffs) + exact H1/H4 ADX and ATR% arrays."""
    entry_ms = INTERVAL_TO_MS["15"]
    h1_ms = INTERVAL_TO_MS["60"]
    h4_ms = INTERVAL_TO_MS["240"]

    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0

    c_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_ratio = ((df["close"] - df["open"]).abs() / c_range).fillna(0.0)

    bar_range = df["high"] - df["low"]
    range_3 = df["high"].rolling(3, min_periods=3).max() - df["low"].rolling(3, min_periods=3).min()
    roc_3_abs = (df["close"] - df["close"].shift(3)).abs() / df["close"].shift(3).replace(0.0, np.nan)
    atr_frac = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)

    is_abn = ((bar_range > 2.2 * df["atr"]) | (range_3 > 3.8 * df["atr"]) | (roc_3_abs > 2.2 * atr_frac)).fillna(False)
    vol_cd = is_abn | is_abn.shift(1, fill_value=False) | is_abn.shift(2, fill_value=False) | is_abn.shift(3, fill_value=False)

    btc = df_btc_15m[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc["btc_move"] = ((btc["close"] - btc["open"]).abs() / btc["open"].replace(0.0, np.nan)) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_move"]], on="timestamp", direction="backward")
    common_no_max_atr = (~vol_cd) & (df["btc_move"].fillna(0.0) <= 0.8) & (df["atr_pct"] >= 0.12)

    df_1h = BybitClient.resample_klines(df_15m, "60")
    df_4h = BybitClient.resample_klines(df_15m, "240")

    h1_e20 = ema(df_1h["close"], 20)
    h1_e50 = ema(df_1h["close"], 50)
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out = pd.DataFrame({
        "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
        "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
        "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
        "h1_adx": h1_adx.values,
    }).sort_values("close_ts")

    h4_e20 = ema(df_4h["close"], 20)
    h4_e50 = ema(df_4h["close"], 50)
    h4_adx, _, _ = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    h4_out = pd.DataFrame({
        "close_ts": (df_4h["timestamp"] + h4_ms).astype("int64"),
        "h4_bull": (df_4h["close"] > h4_e50) & (h4_e20 > h4_e50),
        "h4_bear": (df_4h["close"] < h4_e50) & (h4_e20 < h4_e50),
        "h4_adx": h4_adx.values,
    }).sort_values("close_ts")

    df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
    df = pd.merge_asof(df, h4_out, on="close_ts", direction="backward")

    htf_bull = df["h1_bull"].astype("boolean").fillna(False) & df["h4_bull"].astype("boolean").fillna(False)
    htf_bear = df["h1_bear"].astype("boolean").fillna(False) & df["h4_bear"].astype("boolean").fillna(False)
    h1_adx_s = df["h1_adx"].fillna(0.0)
    h4_adx_s = df["h4_adx"].fillna(0.0)
    adx_avg = (h1_adx_s + h4_adx_s) / 2.0

    rsi_min3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).min(axis=1)
    rsi_max3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).max(axis=1)
    rsi_L = (rsi_min3 < 45.0) & (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"] < 62.0)
    rsi_S = (rsi_max3 > 55.0) & (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"] > 38.0)

    bounce_L = (df["close"] > df["open"]) & (body_ratio >= 0.35) & (df["close"] > df["close"].shift(1))
    bounce_S = (df["close"] < df["open"]) & (body_ratio >= 0.35) & (df["close"] < df["close"].shift(1))

    w_hi = df["high"].shift(1).rolling(36, min_periods=36).max()
    w_lo = df["low"].shift(1).rolling(36, min_periods=36).min()
    w_rng = w_hi - w_lo
    w_ok = w_rng >= (2.2 * df["atr"])
    min_low_2 = pd.concat([df["low"], df["low"].shift(1)], axis=1).min(axis=1)
    max_high_2 = pd.concat([df["high"], df["high"].shift(1)], axis=1).max(axis=1)

    z618_L = w_ok & (min_low_2 <= w_hi - 0.60 * w_rng) & (min_low_2 >= w_hi - 0.80 * w_rng) & (df["close"] > w_hi - 0.786 * w_rng)
    z618_S = w_ok & (max_high_2 >= w_lo + 0.60 * w_rng) & (max_high_2 <= w_lo + 0.80 * w_rng) & (df["close"] < w_lo + 0.786 * w_rng)

    df["raw_L"] = common_no_max_atr & htf_bull & z618_L & rsi_L & bounce_L
    df["raw_S"] = common_no_max_atr & htf_bear & z618_S & rsi_S & bounce_S
    df["q"] = (adx_avg + body_ratio * 25.0).fillna(0.0)

    aligned = timeline_df.merge(
        df[["timestamp", "high", "low", "close", "q", "raw_L", "raw_S", "h1_adx", "h4_adx", "atr_pct"]],
        on="timestamp",
        how="left",
    )
    return {
        "high": aligned["high"].to_numpy(dtype=np.float64),
        "low": aligned["low"].to_numpy(dtype=np.float64),
        "close": aligned["close"].to_numpy(dtype=np.float64),
        "q": aligned["q"].fillna(0.0).to_numpy(dtype=np.float64),
        "raw_L": aligned["raw_L"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "raw_S": aligned["raw_S"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "h1_adx": aligned["h1_adx"].fillna(0.0).to_numpy(dtype=np.float64),
        "h4_adx": aligned["h4_adx"].fillna(0.0).to_numpy(dtype=np.float64),
        "atr_pct": aligned["atr_pct"].fillna(99.0).to_numpy(dtype=np.float64),
    }


def materialize_signals(
    bundle: Dict[str, Dict[str, np.ndarray]],
    spec: TuningSpec,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for sym, d in bundle.items():
        gate = (
            (d["h1_adx"] >= spec.h1_adx_min)
            & (d["h1_adx"] <= spec.h1_adx_max)
            & (d["h4_adx"] >= spec.h4_adx_min)
            & (d["atr_pct"] <= spec.max_atr_pct)
        )
        out[sym] = {
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "q": d["q"],
            "sig_L": d["raw_L"] & gate,
            "sig_S": d["raw_S"] & gate,
        }
    return out


def simulate_tuning_window(
    symbols: List[str],
    sig_data: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    sl_cooldown_bars: int = 3,
    margin_pct: float = 15.0,
    leverage: float = 10.0,
    max_open: int = 5,
    max_same_dir: int = 3,
    max_daily_stops: int = 4,
    tp_pct: float = 1.50,
    sl_pct: float = 3.50,
    initial_deposit: float = 1000.0,
) -> Dict[str, Any]:
    maker_rate = 0.0002
    taker_rate = 0.00055
    tp_frac = tp_pct / 100.0
    sl_frac = sl_pct / 100.0
    pb_frac = 0.00065
    be_trig_frac = (tp_pct * 0.65) / 100.0
    be_off_frac = 0.0015
    notional_mult = (margin_pct / 100.0) * leverage
    daily_dd_cap_pct = max_daily_stops * notional_mult * (sl_frac + maker_rate + taker_rate) * 100.0

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    open_pos: List[_OpenPos] = []
    pending: List[_Pending] = []
    cooldown_until: Dict[str, int] = {}
    trades: List[Tuple[str, float, float]] = []  # (reason, net_usd, ret_on_eq_pct)

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
        eq_before = max(equity, 1.0)
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

        trades.append((reason, net, (net / eq_before) * 100.0))
        cd = sl_cooldown_bars if reason == "SL" else 3
        cooldown_until[pos.symbol] = bar_i + cd

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
                sd = sig_data[pos.symbol]
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
                        if (hi >= pos.be_trigger_price) if is_long else (lo <= pos.be_trigger_price):
                            pos.be_active = True
                        still_open.append(pos)
            open_pos = still_open

        if not daily_halted and open_pos:
            unreal = 0.0
            for pos in open_pos:
                cl = sig_data[pos.symbol]["close"][i]
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

                sd = sig_data[ord_p.symbol]
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
        for sym in symbols:
            if sym in busy or i < cooldown_until.get(sym, -1):
                continue
            sd = sig_data[sym]
            if sd["sig_L"][i]:
                cands.append((float(sd["q"][i]), sym, "LONG", float(sd["close"][i])))
            elif sd["sig_S"][i]:
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
            cl = sig_data[pos.symbol]["close"][-1]
            if not np.isfinite(cl):
                cl = pos.entry_price
            _close(pos, float(cl), "EOD", False, n_bars - 1)

    n_tr = len(trades)
    wins = sum(1 for _, net, _ in trades if net > 0)
    g_win = sum(r for _, _, r in trades if r > 0)
    g_loss = sum(abs(r) for _, _, r in trades if r < 0)
    pf = (g_win / g_loss) if g_loss > 0 else (5.0 if g_win > 0 else 0.0)
    net_usd = equity - initial_deposit

    return {
        "end_usd": round(equity, 2),
        "net_usd": round(net_usd, 2),
        "net_pct": round((net_usd / initial_deposit) * 100.0, 2),
        "trades": n_tr,
        "win_rate_pct": round((wins / n_tr) * 100.0, 2) if n_tr > 0 else 0.0,
        "profit_factor": round(pf, 3),
        "tp_count": sum(1 for r, _, _ in trades if r == "TP"),
        "be_count": sum(1 for r, _, _ in trades if r == "BE"),
        "sl_count": sum(1 for r, _, _ in trades if r == "SL"),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
    }


def select_coins_on_train(
    spec: TuningSpec,
    active_syms: List[str],
    train_sig: Dict[str, Dict[str, np.ndarray]],
    train_tl: np.ndarray,
) -> List[str]:
    rows: List[Tuple[float, float, float, int, str]] = []
    for sym in active_syms:
        r = simulate_tuning_window(
            symbols=[sym],
            sig_data={sym: train_sig[sym]},
            timeline=train_tl,
            sl_cooldown_bars=spec.sl_cooldown_bars,
            margin_pct=15.0,
            max_open=1,
            max_same_dir=1,
            max_daily_stops=4,
        )
        if spec.quality_select:
            # Priority tier: WR >= 80% & trades >= 3 ranked by min(PF, 6.0) * trades
            tier = 1.0 if (r["win_rate_pct"] >= 80.0 and r["trades"] >= 3 and r["net_pct"] > 0) else 0.0
            score = min(r["profit_factor"], 6.0) * r["trades"]
            rows.append((tier, score, r["net_pct"], r["trades"], sym))
        else:
            rows.append((0.0, r["net_pct"], r["win_rate_pct"], r["trades"], sym))

    rows.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [x[4] for x in rows[: spec.top_k]]


def run_single_window_blind(
    spec: TuningSpec,
    active_syms: List[str],
    full_bundle: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    train_start_ms: int,
    train_end_ms: int,
    test_start_ms: int,
    test_end_ms: int,
    initial_deposit: float = 1000.0,
) -> Dict[str, Any]:
    sig_all = materialize_signals(full_bundle, spec)
    tr_mask = (timeline >= train_start_ms) & (timeline < train_end_ms)
    te_mask = (timeline >= test_start_ms) & (timeline <= test_end_ms)

    tr_sig = slice_sym_data(sig_all, tr_mask)
    te_sig = slice_sym_data(sig_all, te_mask)
    tr_tl = timeline[tr_mask]
    te_tl = timeline[te_mask]

    selected = select_coins_on_train(spec, active_syms, tr_sig, tr_tl)
    res = simulate_tuning_window(
        symbols=selected,
        sig_data=te_sig,
        timeline=te_tl,
        sl_cooldown_bars=spec.sl_cooldown_bars,
        margin_pct=15.0,
        max_open=5,
        max_same_dir=3,
        max_daily_stops=4,
        initial_deposit=initial_deposit,
    )
    return res


def run_rolling_3m_exam(
    spec: TuningSpec,
    active_syms: List[str],
    full_bundle: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    sim_start_ms: int,
    sim_end_ms: int,
) -> Dict[str, Any]:
    """
    Rolling 3-Month Blind Walk-Forward:
      - Month 4: Train d1-90   -> Blind d91-120
      - Month 5: Train d31-120 -> Blind d121-150
      - Month 6: Train d61-150 -> Blind d151-180
    """
    day_ms = 86_400_000
    windows = [
        ("M4 (d91-120)", sim_start_ms + 0 * day_ms, sim_start_ms + 90 * day_ms, sim_start_ms + 90 * day_ms, sim_start_ms + 120 * day_ms),
        ("M5 (d121-150)", sim_start_ms + 30 * day_ms, sim_start_ms + 120 * day_ms, sim_start_ms + 120 * day_ms, sim_start_ms + 150 * day_ms),
        ("M6 (d151-180)", sim_start_ms + 60 * day_ms, sim_start_ms + 150 * day_ms, sim_start_ms + 150 * day_ms, sim_end_ms),
    ]
    eq = 1000.0
    m_results: List[Dict[str, Any]] = []
    max_dd_any = 0.0
    total_tr = 0
    total_sl = 0
    total_wins = 0

    for label, tr_s, tr_e, te_s, te_e in windows:
        r = run_single_window_blind(spec, active_syms, full_bundle, timeline, tr_s, tr_e, te_s, te_e, initial_deposit=1000.0)
        eq *= 1.0 + r["net_pct"] / 100.0
        max_dd_any = max(max_dd_any, r["max_dd_pct"])
        total_tr += r["trades"]
        total_sl += r["sl_count"]
        total_wins += int(round(r["trades"] * r["win_rate_pct"] / 100.0))
        m_results.append({"window": label, **r})

    all_pos = all(m["net_pct"] > 0.0 for m in m_results)
    return {
        "name": spec.name,
        "months": m_results,
        "m4_pct": m_results[0]["net_pct"],
        "m5_pct": m_results[1]["net_pct"],
        "m6_pct": m_results[2]["net_pct"],
        "all_3_positive": all_pos,
        "compounded_end_usd": round(eq, 2),
        "compounded_net_pct": round((eq - 1000.0) / 10.0, 2),
        "worst_month_pct": min(m["net_pct"] for m in m_results),
        "max_dd_pct": round(max_dd_any, 2),
        "total_trades": total_tr,
        "total_sl": total_sl,
        "avg_wr_pct": round((total_wins / total_tr) * 100.0, 2) if total_tr > 0 else 0.0,
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

    print("Building causal feature bundles across cached Top-100...")
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

    # Champion baseline
    champ_spec = TuningSpec("base", "CHAMPION (H1 ADX>=25, H4>=20, Top-25, SL-CD 3, MaxATR 1.5%)")
    champ_2m = run_single_window_blind(champ_spec, active_syms, bundle, timeline, sim_start_ms, split_ms, split_ms, sim_end_ms)

    # 4 Tuning Blocks (strictly one change at a time!)
    candidates: List[TuningSpec] = [
        # Block 1: Fine ADX Grid
        TuningSpec("block1_adx", "H1 ADX >= 28 (H4 >= 20)", h1_adx_min=28.0),
        TuningSpec("block1_adx", "H1 ADX >= 30 (H4 >= 20)", h1_adx_min=30.0),
        TuningSpec("block1_adx", "H1 ADX >= 32 (H4 >= 20)", h1_adx_min=32.0),
        TuningSpec("block1_adx", "H1 ADX >= 25 + H4 ADX >= 25", h1_adx_min=25.0, h4_adx_min=25.0),
        TuningSpec("block1_adx", "H1 ADX коридор 25..55", h1_adx_min=25.0, h1_adx_max=55.0),
        TuningSpec("block1_adx", "H1 ADX коридор 25..60", h1_adx_min=25.0, h1_adx_max=60.0),
        # Block 2: Portfolio Width
        TuningSpec("block2_width", "Ширина портфеля Топ-30 (H1 ADX >= 25)", top_k=30),
        TuningSpec("block2_width", "Ширина портфеля Топ-35 (H1 ADX >= 25)", top_k=35),
        # Block 3: Per-coin SL Cooldown
        TuningSpec("block3_slcd", "SL-Cooldown 16 свечей (4ч) по монете", sl_cooldown_bars=16),
        TuningSpec("block3_slcd", "SL-Cooldown 32 свечи (8ч) по монете", sl_cooldown_bars=32),
        # Block 4: Max ATR% & Smart Quality Coin Selection
        TuningSpec("block4_vol_sel", "Max ATR(14)% <= 1.2% (строже базы 1.5%)", max_atr_pct=1.20),
        TuningSpec("block4_vol_sel", "Max ATR(14)% <= 2.0% (шире базы 1.5%)", max_atr_pct=2.00),
        TuningSpec("block4_vol_sel", "Умный отбор Топ-25 (PF * сделки при WR >= 80%)", quality_select=True),
    ]

    print("\n" + "=" * 108)
    print(
        f"CHAMPION 2M BLIND: $1,000 -> ${champ_2m['end_usd']:,.2f} ({champ_2m['net_pct']:+.2f}%) | "
        f"{champ_2m['trades']} tr ({champ_2m['trades']/60:.2f}/d) | WR {champ_2m['win_rate_pct']:.2f}% | "
        f"SL={champ_2m['sl_count']} | DD={champ_2m['max_dd_pct']:.2f}%"
    )
    print("=" * 108)

    all_2m: List[Tuple[TuningSpec, Dict[str, Any]]] = []
    by_block: Dict[str, List[Tuple[TuningSpec, Dict[str, Any]]]] = {}
    for spec in candidates:
        res = run_single_window_blind(spec, active_syms, bundle, timeline, sim_start_ms, split_ms, split_ms, sim_end_ms)
        all_2m.append((spec, res))
        by_block.setdefault(spec.block, []).append((spec, res))
        tag = "ЛУЧШЕ" if res["end_usd"] > champ_2m["end_usd"] else "ХУЖЕ"
        print(
            f"  [{spec.block:<14}] {spec.name:<48} | {tag:<5} | {res['trades']:>3} tr WR {res['win_rate_pct']:>5.1f}% SL={res['sl_count']:>2} | "
            f"$1k->${res['end_usd']:>8,.2f} ({res['net_pct']:>+6.1f}%) | DD {res['max_dd_pct']:>5.1f}%"
        )

    # Pick top-3 overall contenders from the 2m blind test for the Rolling 3-Month Anti-Overfitting Exam
    sorted_cands = sorted(all_2m, key=lambda x: (x[1]["end_usd"], -x[1]["max_dd_pct"]), reverse=True)
    top3_specs = [x[0] for x in sorted_cands[:3]]

    print("\n" + "=" * 108)
    print("DOUBLE EXAM: ROLLING 3-MONTH WALK-FORWARD (M4: d91-120 | M5: d121-150 | M6: d151-180)")
    print("=" * 108)

    rolling_results: List[Dict[str, Any]] = []
    for sp in [champ_spec] + top3_specs:
        rr = run_rolling_3m_exam(sp, active_syms, bundle, timeline, sim_start_ms, sim_end_ms)
        rolling_results.append(rr)
        pos_flag = "ALL 3 POS" if rr["all_3_positive"] else "HAS NEG MONTH"
        print(
            f"\n>>> {rr['name']} [{pos_flag}]\n"
            f"  M4: {rr['m4_pct']:>+7.2f}% | M5: {rr['m5_pct']:>+7.2f}% | M6: {rr['m6_pct']:>+7.2f}%\n"
            f"  3m Total: $1,000 -> ${rr['compounded_end_usd']:,.2f} ({rr['compounded_net_pct']:+.2f}%) | "
            f"Trades={rr['total_trades']} | WR={rr['avg_wr_pct']:.1f}% | SL={rr['total_sl']} | Worst DD={rr['max_dd_pct']:.1f}%"
        )

    out = {
        "champion_2m": champ_2m,
        "all_2m": [{"block": s.block, "name": s.name, **r} for s, r in all_2m],
        "best_per_block": {
            b: {"name": max(items, key=lambda x: x[1]["end_usd"])[0].name, **max(items, key=lambda x: x[1]["end_usd"])[1]}
            for b, items in by_block.items()
        },
        "rolling_3m": rolling_results,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path("data/blind_final_tuning_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/blind_final_tuning_results.json")


if __name__ == "__main__":
    main()
