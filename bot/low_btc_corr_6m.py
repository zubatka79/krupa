"""POLYGRAPH 6-MONTH EXAM (`bot/low_btc_corr_6m.py`):
Low-BTC-Correlation Liquid Crypto Coins + Pure Local Technical Analysis (ZERO BTC Filter).

HONESTY PROTOCOL:
  1. Coin selection (liquidity + lowest BTC correlation) and strategy rules were locked
     STRICTLY on Days 1-120 (M1-M4) with Days 121-180 physically excluded.
  2. Months 5 & 6 (Days 121-180) are executed ONCE blindly with zero parameter changes.
  3. Exact Bybit math: $1,000 start compounded, 15% isolated margin x 10x, max 5 open /
     3 same direction, 4 SL/day, 0.02% Maker / 0.055% Taker + 0.04% slippage on SL/BE.

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_walkforward import load_cached_15m
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, compute_rsi, ema


@dataclass
class PureLocalSpec:
    name: str
    is_locked_primary: bool = False
    h1_adx_min: float = 30.0
    h4_adx_min: float = 20.0
    max_atr_pct: float = 1.50
    local_cat_r24: float = 5.0
    local_cat_v4h: float = 5.0
    align_c24: bool = True
    tp_pct: float = 1.75
    sl_pct: float = 3.50
    smart_be: bool = True
    be_trigger_ratio: float = 0.65
    be_offset_pct: float = 0.25
    margin_pct: float = 15.0
    max_open: int = 5
    max_same_dir: int = 3
    max_daily_stops: int = 4


def compute_corr_metrics(
    df_coin_win: pd.DataFrame,
    btc_15: pd.Series,
    btc_1h: pd.Series,
    btc_4h: pd.Series,
) -> Dict[str, float]:
    c15 = df_coin_win.set_index("timestamp")["close"].pct_change()
    j15 = pd.concat([btc_15, c15], axis=1, keys=["btc", "coin"]).dropna()
    corr_15m = float(j15["btc"].corr(j15["coin"])) if len(j15) > 20 else 0.0

    c1h = BybitClient.resample_klines(df_coin_win, "60").set_index("timestamp")["close"].pct_change()
    j1h = pd.concat([btc_1h, c1h], axis=1, keys=["btc", "coin"]).dropna()
    corr_1h = float(j1h["btc"].corr(j1h["coin"])) if len(j1h) > 20 else 0.0

    c4h = BybitClient.resample_klines(df_coin_win, "240").set_index("timestamp")["close"].pct_change()
    j4h = pd.concat([btc_4h, c4h], axis=1, keys=["btc", "coin"]).dropna()
    corr_4h = float(j4h["btc"].corr(j4h["coin"])) if len(j4h) > 10 else 0.0

    # Downside behavior during BTC drops (BTC 1H <= -0.5% and <= -1.0%)
    dn05 = j1h[j1h["btc"] <= -0.005]
    dn10 = j1h[j1h["btc"] <= -0.010]
    dn05_same = float((dn05["coin"] < 0).mean() * 100.0) if len(dn05) >= 5 else 0.0
    dn05_beta = float(dn05["coin"].cov(dn05["btc"]) / max(dn05["btc"].var(), 1e-12)) if len(dn05) >= 5 else 0.0
    dn10_same = float((dn10["coin"] < 0).mean() * 100.0) if len(dn10) >= 3 else 0.0
    dn10_avg_ret = float(dn10["coin"].mean() * 100.0) if len(dn10) >= 3 else 0.0

    return {
        "corr_15m": round(corr_15m, 3),
        "corr_1h": round(corr_1h, 3),
        "corr_4h": round(corr_4h, 3),
        "corr_avg": round((corr_15m + corr_1h + corr_4h) / 3.0, 3),
        "dn05_same_pct": round(dn05_same, 1),
        "dn05_beta": round(dn05_beta, 2),
        "dn10_same_pct": round(dn10_same, 1),
        "dn10_avg_ret_pct": round(dn10_avg_ret, 2),
        "dn10_bars": int(len(dn10)),
    }


def select_low_corr_pool_120d(
    syms_100: List[str],
    df_btc: pd.DataFrame,
    cache_dir: Path,
    sim_start: int,
    split_120: int,
    sim_end: int,
    top_n: int = 15,
) -> Tuple[List[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """Select Top-N lowest-BTC-correlation liquid coins STRICTLY on Days 1-120,
    then measure their out-of-sample correlation on Blind Days 121-180 for diagnostics."""
    day_ms = 86_400_000
    btc_dev = df_btc[(df_btc["timestamp"] >= sim_start) & (df_btc["timestamp"] < split_120)].sort_values("timestamp")
    btc_dev_15 = btc_dev.set_index("timestamp")["close"].pct_change()
    btc_dev_1h = BybitClient.resample_klines(btc_dev, "60").set_index("timestamp")["close"].pct_change()
    btc_dev_4h = BybitClient.resample_klines(btc_dev, "240").set_index("timestamp")["close"].pct_change()

    btc_bl = df_btc[(df_btc["timestamp"] >= split_120) & (df_btc["timestamp"] <= sim_end)].sort_values("timestamp")
    btc_bl_15 = btc_bl.set_index("timestamp")["close"].pct_change()
    btc_bl_1h = BybitClient.resample_klines(btc_bl, "60").set_index("timestamp")["close"].pct_change()
    btc_bl_4h = BybitClient.resample_klines(btc_bl, "240").set_index("timestamp")["close"].pct_change()

    n_btc_dev = len(btc_dev)
    candidates: List[Dict[str, Any]] = []
    full_dfs: Dict[str, pd.DataFrame] = {}

    for sym in syms_100:
        try:
            df_full = load_cached_15m(sym, cache_dir)
        except Exception:
            continue
        df_dev = df_full[(df_full["timestamp"] >= sim_start) & (df_full["timestamp"] < split_120)].sort_values("timestamp")
        if len(df_dev) / max(n_btc_dev, 1) < 0.90:
            continue
        daily_turn = df_dev.assign(day=df_dev["timestamp"] // day_ms).groupby("day")["turnover"].sum()
        med_turn_m = float(daily_turn.median()) / 1e6
        p10_turn_m = float(daily_turn.quantile(0.10)) / 1e6
        flat_pct = float((df_dev["high"] == df_dev["low"]).mean() * 100.0)
        if med_turn_m < 10.0 or p10_turn_m < 4.0 or flat_pct >= 0.5:
            continue

        dev_m = compute_corr_metrics(df_dev, btc_dev_15, btc_dev_1h, btc_dev_4h)
        candidates.append({
            "sym": sym,
            "med_turn_m_120d": round(med_turn_m, 2),
            "p10_turn_m_120d": round(p10_turn_m, 2),
            "dev_120d": dev_m,
        })
        full_dfs[sym] = df_full

    candidates.sort(key=lambda x: x["dev_120d"]["corr_avg"])
    selected = candidates[:top_n]

    # Out-of-sample correlation diagnostics on Blind Months 5-6 (post-selection!)
    for row in selected:
        df_bl = full_dfs[row["sym"]]
        df_bl = df_bl[(df_bl["timestamp"] >= split_120) & (df_bl["timestamp"] <= sim_end)].sort_values("timestamp")
        row["blind_60d"] = compute_corr_metrics(df_bl, btc_bl_15, btc_bl_1h, btc_bl_4h)

    return selected, {r["sym"]: full_dfs[r["sym"]] for r in selected}


def build_pure_local_bundle(df_15m: pd.DataFrame, timeline_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Pure local TA features for a single coin — ZERO BTC inputs."""
    entry_ms = INTERVAL_TO_MS["15"]
    h1_ms = INTERVAL_TO_MS["60"]
    h4_ms = INTERVAL_TO_MS["240"]

    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0

    df["c_r24h"] = df["close"].pct_change(96) * 100.0
    df["c_vol4h"] = (
        (df["high"].rolling(16, min_periods=16).max() - df["low"].rolling(16, min_periods=16).min())
        / df["close"].replace(0.0, np.nan)
        * 100.0
    )

    c_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_ratio = ((df["close"] - df["open"]).abs() / c_range).fillna(0.0)

    bar_range = df["high"] - df["low"]
    range_3 = df["high"].rolling(3, min_periods=3).max() - df["low"].rolling(3, min_periods=3).min()
    roc_3_abs = (df["close"] - df["close"].shift(3)).abs() / df["close"].shift(3).replace(0.0, np.nan)
    atr_frac = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)

    is_abn = ((bar_range > 2.2 * df["atr"]) | (range_3 > 3.8 * df["atr"]) | (roc_3_abs > 2.2 * atr_frac)).fillna(False)
    vol_cd = is_abn | is_abn.shift(1, fill_value=False) | is_abn.shift(2, fill_value=False) | is_abn.shift(3, fill_value=False)

    # Pure local common filter (NO BTC inputs!)
    common_local = (~vol_cd) & (df["atr_pct"] >= 0.12)

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

    df["raw_L"] = common_local & htf_bull & z618_L & rsi_L & bounce_L
    df["raw_S"] = common_local & htf_bear & z618_S & rsi_S & bounce_S
    df["q"] = (adx_avg + body_ratio * 25.0).fillna(0.0)

    aligned = timeline_df.merge(
        df[["timestamp", "high", "low", "close", "q", "raw_L", "raw_S", "h1_adx", "h4_adx", "atr_pct", "c_r24h", "c_vol4h"]],
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
        "c_r24h": aligned["c_r24h"].fillna(0.0).to_numpy(dtype=np.float64),
        "c_vol4h": aligned["c_vol4h"].fillna(0.0).to_numpy(dtype=np.float64),
    }


def simulate_6m_pure_local(
    cfg: PureLocalSpec,
    syms: List[str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    btc_diag: Dict[str, np.ndarray],
    timeline: np.ndarray,
    month_bounds: List[Tuple[str, int, int]],
    blind_split_ms: int,
) -> Dict[str, Any]:
    """Pure local TA simulation (btc_diag is ONLY used post-exit for SL diagnostics, NEVER in strategy!)."""
    mk, tk, slip = 0.0002, 0.00055, 0.0004
    tp_f, sl_f, pb_f = cfg.tp_pct / 100.0, cfg.sl_pct / 100.0, 0.00065
    be_tr_f, be_off_f = tp_f * cfg.be_trigger_ratio, cfg.be_offset_pct / 100.0

    btc_r1h = btc_diag["r1h"]
    btc_r4h = btc_diag["r4h"]

    equity = 1000.0
    peak_eq = 1000.0
    max_dd = 0.0
    open_pos, pending, cd_until, trades = [], [], {}, []
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end = {}, {}

    def _close(p, trig_p, reason, is_mk, bar_i, ts):
        nonlocal equity, peak_eq, max_dd, day_sl, halted
        is_l = p["dir"] == "LONG"
        ex_p = trig_p if is_mk else (trig_p * (1 - slip) if is_l else trig_p * (1 + slip))
        gross = (ex_p - p["ep"]) * p["qty"] if is_l else (p["ep"] - ex_p) * p["qty"]
        net = max(-p["margin"], gross - p["efee"] - abs(p["qty"] * ex_p) * (mk if is_mk else tk))
        equity = max(1.0, equity + net)
        if reason == "SL":
            day_sl += 1
            if day_sl >= cfg.max_daily_stops:
                halted = True
        if equity > peak_eq:
            peak_eq = equity
        dd = (peak_eq - equity) / peak_eq * 100.0
        if dd > max_dd:
            max_dd = dd

        # Pure diagnostic check: did BTC move sharply against this trade during exit?
        adv_btc_1h = (btc_r1h[bar_i] <= -1.0) if is_l else (btc_r1h[bar_i] >= 1.0)
        adv_btc_4h = (btc_r4h[bar_i] <= -2.0) if is_l else (btc_r4h[bar_i] >= 2.0)
        trades.append({
            "sym": p["sym"],
            "dir": p["dir"],
            "reason": reason,
            "net": float(net),
            "is_blind": bool(ts >= blind_split_ms),
            "exit_adv_btc": bool(adv_btc_1h or adv_btc_4h),
            "exit_btc_dump": bool(btc_r1h[bar_i] <= -1.0 or btc_r4h[bar_i] <= -2.0),
        })
        cd_until[p["sym"]] = bar_i + 3

    for i in range(len(timeline)):
        ts = int(timeline[i])
        for mn, ms, me in month_bounds:
            if mn not in m_start and ts >= ms:
                m_start[mn] = equity
            if ts <= me:
                m_end[mn] = equity
        dk = ts // 86_400_000
        if dk != cur_day:
            cur_day, day_sl, halted = dk, 0, False

        if open_pos:
            still = []
            for p in open_pos:
                sd = bundle[p["sym"]]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    still.append(p)
                    continue
                is_l = p["dir"] == "LONG"
                if cfg.smart_be and p["be_on"]:
                    if (lo <= p["be_p"]) if is_l else (hi >= p["be_p"]):
                        _close(p, p["be_p"], "BE", False, i, ts)
                        continue
                    if (hi >= p["tp_p"]) if is_l else (lo <= p["tp_p"]):
                        _close(p, p["tp_p"], "TP", True, i, ts)
                        continue
                else:
                    if (lo <= p["sl_p"]) if is_l else (hi >= p["sl_p"]):
                        _close(p, p["sl_p"], "SL", False, i, ts)
                        continue
                    if (hi >= p["tp_p"]) if is_l else (lo <= p["tp_p"]):
                        _close(p, p["tp_p"], "TP", True, i, ts)
                        continue
                    if cfg.smart_be and ((hi >= p["be_tr"]) if is_l else (lo <= p["be_tr"])):
                        p["be_on"] = True
                still.append(p)
            open_pos = still

        if pending:
            npend = []
            for o in pending:
                if i > o["exp"] or halted:
                    continue
                if any(p["sym"] == o["sym"] for p in open_pos) or i < cd_until.get(o["sym"], -1):
                    continue
                sd = bundle[o["sym"]]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    if i < o["exp"]:
                        npend.append(o)
                    continue
                is_l = o["dir"] == "LONG"
                if not ((lo < o["lp"]) if is_l else (hi > o["lp"])):
                    if i < o["exp"]:
                        npend.append(o)
                    continue
                if len(open_pos) >= cfg.max_open or sum(1 for p in open_pos if p["dir"] == o["dir"]) >= cfg.max_same_dir:
                    continue
                margin = equity * (cfg.margin_pct / 100.0)
                if sum(p["margin"] for p in open_pos) + margin > equity * 0.95:
                    continue
                notional = margin * 10.0
                ep = o["lp"]
                pos = {
                    "sym": o["sym"], "dir": o["dir"], "ep": ep, "qty": notional / ep, "margin": margin,
                    "efee": notional * mk, "be_on": False,
                    "sl_p": ep * (1 - sl_f) if is_l else ep * (1 + sl_f),
                    "tp_p": ep * (1 + tp_f) if is_l else ep * (1 - tp_f),
                    "be_tr": ep * (1 + be_tr_f) if is_l else ep * (1 - be_tr_f),
                    "be_p": ep * (1 + be_off_f) if is_l else ep * (1 - be_off_f),
                }
                if (lo <= pos["sl_p"]) if is_l else (hi >= pos["sl_p"]):
                    _close(pos, pos["sl_p"], "SL", False, i, ts)
                else:
                    open_pos.append(pos)
            pending = npend

        if halted or len(open_pos) >= cfg.max_open:
            continue
        busy = {p["sym"] for p in open_pos} | {o["sym"] for o in pending}
        cands = []
        for sym in syms:
            if sym in busy or i < cd_until.get(sym, -1):
                continue
            sd = bundle[sym]
            if sd["h1_adx"][i] < cfg.h1_adx_min or sd["h4_adx"][i] < cfg.h4_adx_min or sd["atr_pct"][i] > cfg.max_atr_pct:
                continue
            c24 = sd["c_r24h"][i]
            v4h = sd["c_vol4h"][i]
            if cfg.local_cat_r24 > 0.0 and (abs(c24) < cfg.local_cat_r24 and v4h < cfg.local_cat_v4h):
                continue
            if sd["raw_L"][i]:
                if cfg.align_c24 and c24 <= 0.0:
                    continue
                cands.append((float(sd["q"][i]), sym, "LONG", float(sd["close"][i])))
            elif sd["raw_S"][i]:
                if cfg.align_c24 and c24 >= 0.0:
                    continue
                cands.append((float(sd["q"][i]), sym, "SHORT", float(sd["close"][i])))
        cands.sort(key=lambda x: x[0], reverse=True)
        for _, sym, d, cl in cands:
            if len(open_pos) + len(pending) >= cfg.max_open + 3:
                break
            if sum(1 for p in open_pos if p["dir"] == d) >= cfg.max_same_dir:
                continue
            lp = cl * (1 - pb_f) if d == "LONG" else cl * (1 + pb_f)
            pending.append({"sym": sym, "dir": d, "lp": lp, "exp": i + 2})

    if open_pos:
        last_ts = int(timeline[-1])
        for p in open_pos:
            cl = bundle[p["sym"]]["close"][-1]
            _close(p, float(cl) if np.isfinite(cl) else p["ep"], "EOD", False, len(timeline) - 1, last_ts)
        for mn, _, me in month_bounds:
            if last_ts <= me:
                m_end[mn] = equity

    mp = {mn: round((m_end.get(mn, 1000.0) - m_start.get(mn, 1000.0)) / max(m_start.get(mn, 1000.0), 1.0) * 100.0, 2) for mn, _, _ in month_bounds}
    n = len(trades)
    sl_all = [t for t in trades if t["reason"] == "SL"]
    bl_tr = [t for t in trades if t["is_blind"]]
    bl_sl = [t for t in bl_tr if t["reason"] == "SL"]

    return {
        "name": cfg.name,
        "is_locked_primary": cfg.is_locked_primary,
        "spec": asdict(cfg),
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n,
        "wr_pct": round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2),
        "sl_count": len(sl_all),
        "tp_count": sum(1 for t in trades if t["reason"] == "TP"),
        "be_count": sum(1 for t in trades if t["reason"] == "BE"),
        "max_dd_pct": round(max_dd, 2),
        "blind_trades": len(bl_tr),
        "blind_wr_pct": round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2),
        "blind_sl": len(bl_sl),
        "blind_sl_on_adv_btc": sum(1 for t in bl_sl if t["exit_adv_btc"]),
        "all_sl_on_adv_btc": sum(1 for t in sl_all if t["exit_adv_btc"]),
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    syms_100 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    day_ms = 86_400_000
    sim_end = int(df_btc["timestamp"].iloc[-1])
    sim_start = sim_end - 180 * day_ms
    blind_split = sim_start + 120 * day_ms

    selected_meta, sel_dfs = select_low_corr_pool_120d(syms_100, df_btc, cache_dir, sim_start, blind_split, sim_end, top_n=15)
    sel_syms = [r["sym"] for r in selected_meta]

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start) & (df_btc["timestamp"] <= sim_end)].sort_values("timestamp")
    tl = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    tl_df = pd.DataFrame({"timestamp": tl})

    # Post-exit BTC diagnostic series ONLY (never passed to strategy rules!)
    btc_tmp = df_btc[["timestamp", "close"]].copy().sort_values("timestamp")
    btc_tmp["r1h"] = btc_tmp["close"].pct_change(4) * 100.0
    btc_tmp["r4h"] = btc_tmp["close"].pct_change(16) * 100.0
    btc_al = tl_df.merge(btc_tmp[["timestamp", "r1h", "r4h"]], on="timestamp", how="left")
    btc_diag = {
        "r1h": btc_al["r1h"].fillna(0.0).to_numpy(dtype=np.float64),
        "r4h": btc_al["r4h"].fillna(0.0).to_numpy(dtype=np.float64),
    }

    m_bounds = [
        ("M1", sim_start + 0 * day_ms, sim_start + 30 * day_ms - 1),
        ("M2", sim_start + 30 * day_ms, sim_start + 60 * day_ms - 1),
        ("M3", sim_start + 60 * day_ms, sim_start + 90 * day_ms - 1),
        ("M4", sim_start + 90 * day_ms, sim_start + 120 * day_ms - 1),
        ("M5", sim_start + 120 * day_ms, sim_start + 150 * day_ms - 1),
        ("M6", sim_start + 150 * day_ms, sim_end),
    ]

    bundle = {sym: build_pure_local_bundle(sel_dfs[sym], tl_df) for sym in sel_syms}

    specs = [
        PureLocalSpec(
            name="LOCKED PRIMARY (S3: Pure Local fibo_v2 + LocalCat 5% + H1 ADX>=30 + TP1.75/SL3.50/BE+0.25)",
            is_locked_primary=True,
            h1_adx_min=30.0,
            local_cat_r24=5.0,
            local_cat_v4h=5.0,
            align_c24=True,
            tp_pct=1.75,
            be_offset_pct=0.25,
        ),
        PureLocalSpec(
            name="REF-1 (S2: Pure Local fibo_v2 + LocalCat 4% + H1 ADX>=30 + TP1.75/SL3.50/BE+0.25)",
            is_locked_primary=False,
            h1_adx_min=30.0,
            local_cat_r24=4.0,
            local_cat_v4h=4.0,
            align_c24=True,
            tp_pct=1.75,
            be_offset_pct=0.25,
        ),
        PureLocalSpec(
            name="REF-0 (Raw Pure Local fibo_v2: H1 ADX>=25, NO LocalCat, TP1.50/SL3.50/BE+0.15)",
            is_locked_primary=False,
            h1_adx_min=25.0,
            local_cat_r24=0.0,
            local_cat_v4h=0.0,
            align_c24=False,
            tp_pct=1.50,
            be_offset_pct=0.15,
        ),
    ]

    print("\n" + "=" * 116)
    print("LOW-BTC-CORRELATION LIQUID COINS (SELECTED STRICTLY ON DAYS 1-120 | MED TURN >= $10M, P10 >= $4M)")
    print("=" * 116)
    print(f"{'Coin':<13} {'Med$M/d':>7} | {'Dev Corr(15m/1h/4h/Avg)':<24} {'DevDnSame%':>10} {'DevBeta':>7} | {'Blind CorrAvg':>13} {'BlDn05%':>8} {'BlDn10%':>8} {'BlDn10Ret':>9}")
    dev_avgs, bl_avgs, bl_dn05s, bl_dn10s = [], [], [], []
    for r in selected_meta:
        d, b = r["dev_120d"], r["blind_60d"]
        dev_avgs.append(d["corr_avg"])
        bl_avgs.append(b["corr_avg"])
        bl_dn05s.append(b["dn05_same_pct"])
        bl_dn10s.append(b["dn10_same_pct"])
        print(
            f"{r['sym']:<13} {r['med_turn_m_120d']:>7.1f} | "
            f"{d['corr_15m']:.2f}/{d['corr_1h']:.2f}/{d['corr_4h']:.2f} ({d['corr_avg']:.3f})      "
            f"{d['dn05_same_pct']:>9.1f}% {d['dn05_beta']:>7.2f} | "
            f"{b['corr_avg']:>13.3f} {b['dn05_same_pct']:>7.1f}% {b['dn10_same_pct']:>7.1f}% {b['dn10_avg_ret_pct']:>+8.2f}%"
        )
    print(
        f"POOL AVERAGE: Dev CorrAvg={np.mean(dev_avgs):.3f} -> Blind CorrAvg={np.mean(bl_avgs):.3f} | "
        f"Blind BTC-Dump(-0.5% 1H) Same-Dir={np.mean(bl_dn05s):.1f}% | Blind BTC-Dump(-1.0% 1H) Same-Dir={np.mean(bl_dn10s):.1f}%"
    )

    print("\n" + "=" * 116)
    print("POLYGRAPH 6-MONTH EXAM: PURE LOCAL TA ON LOW-BTC-CORR POOL (15% x 10x ISOLATED | BYBIT FEES + 0.04% SLIPPAGE)")
    print("=" * 116)
    sim_results = []
    for sp in specs:
        res = simulate_6m_pure_local(sp, sel_syms, bundle, btc_diag, tl, m_bounds, blind_split)
        sim_results.append(res)
        mp = res["monthly_pct"]
        print(f"\n>>> {res['name']}")
        print(
            f"  M1:{mp['M1']:>+6.1f}% / M2:{mp['M2']:>+6.1f}% / M3:{mp['M3']:>+6.1f}% / M4:{mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5:{mp['M5']:>+6.1f}% / M6:{mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого: $1 000 -> ${res['end_usd']:,.2f} | Сделок: {res['trades']} | WR: {res['wr_pct']:.1f}% "
            f"(Blind WR: {res['blind_wr_pct']:.1f}%) | Стопов: {res['sl_count']} | Макс. DD: {res['max_dd_pct']:.1f}%"
        )
        print(
            f"  Blind SL BTC-Exposure: {res['blind_sl_on_adv_btc']}/{res['blind_sl']} blind SLs "
            f"({res['all_sl_on_adv_btc']}/{res['sl_count']} total SLs) occurred during adverse BTC 1H>=1% or 4H>=2% move"
        )

    out = {
        "selected_pool_120d": selected_meta,
        "pool_summary": {
            "dev_corr_avg_mean": round(float(np.mean(dev_avgs)), 3),
            "blind_corr_avg_mean": round(float(np.mean(bl_avgs)), 3),
            "blind_btc_dump_05_same_dir_pct": round(float(np.mean(bl_dn05s)), 1),
            "blind_btc_dump_10_same_dir_pct": round(float(np.mean(bl_dn10s)), 1),
        },
        "results": sim_results,
        "v4_benchmark": {
            "end_usd": 4344.67,
            "wr_pct": 84.5,
            "max_dd_pct": 20.4,
            "monthly_pct": {"M1": 54.5, "M2": 6.8, "M3": 47.0, "M4": 30.5, "M5": 34.7, "M6": 2.0},
        },
    }
    Path("data/low_btc_corr_6m_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/low_btc_corr_6m_results.json")


if __name__ == "__main__":
    main()
