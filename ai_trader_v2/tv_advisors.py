"""
TradingView-Inspired Analytical Advisors (`ai_trader_v2/tv_advisors.py`):
  1) Liquidity Sweep / Swing Failure Pattern (SFP) against 1H, 4H swings & PDH/PDL
  2) 4-Day Volume Profile (VPVR: POC, VAH, VAL, Value Area 70%, POC obstacle check)
  3) UTC Session Levels (PDH, PDL, ASIA_HIGH, ASIA_LOW 00:00-08:00 UTC)

Strict Design Guarantees:
  - Advisory only: NEVER hard-blocks entries in Python (feeds AI Judge JSON, PNG chart,
    and candidate priority_score).
  - Zero extra HTTP requests: computes strictly from already-loaded closed 15m, 1H, 4H klines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
M15_MS = 900_000


# ----------------------------------------------------------------------
# Advisor 3: UTC Previous Day (PDH/PDL) & Asia Session (00:00-08:00 UTC)
# ----------------------------------------------------------------------
def compute_session_levels(
    df15: pd.DataFrame,
    df1h: Optional[pd.DataFrame],
    close_now: float,
    direction: str,
) -> Dict[str, Any]:
    """Compute completed UTC Previous Day High/Low (PDH/PDL) and Asia High/Low (00:00-08:00 UTC)."""
    empty: Dict[str, Any] = {
        "pdh": None,
        "pdl": None,
        "asia_high": None,
        "asia_low": None,
        "dist_pdh_pct": None,
        "dist_pdl_pct": None,
        "dist_asia_high_pct": None,
        "dist_asia_low_pct": None,
        "nearest_session_level": None,
        "session_obstacle_ahead": False,
    }
    if df15 is None or df15.empty or close_now <= 0:
        return empty

    ts15 = df15["timestamp"].to_numpy(dtype=np.int64)
    hi15 = df15["high"].to_numpy(dtype=np.float64)
    lo15 = df15["low"].to_numpy(dtype=np.float64)

    # Current UTC calendar day of the latest closed bar
    last_ts = int(ts15[-1])
    cur_day_start = (last_ts // DAY_MS) * DAY_MS
    prev_day_start = cur_day_start - DAY_MS

    # 1. PDH / PDL from strictly completed previous UTC calendar day [prev_day_start, cur_day_start)
    prev_mask = (ts15 >= prev_day_start) & (ts15 < cur_day_start)
    pdh: Optional[float] = None
    pdl: Optional[float] = None
    if int(np.sum(prev_mask)) >= 4:
        pdh = float(np.max(hi15[prev_mask]))
        pdl = float(np.min(lo15[prev_mask]))
    elif df1h is not None and not df1h.empty:
        ts1h = df1h["timestamp"].to_numpy(dtype=np.int64)
        pm1h = (ts1h >= prev_day_start) & (ts1h < cur_day_start)
        if np.any(pm1h):
            pdh = float(np.max(df1h["high"].to_numpy(dtype=np.float64)[pm1h]))
            pdl = float(np.min(df1h["low"].to_numpy(dtype=np.float64)[pm1h]))

    # 2. Asia session (00:00 - 08:00 UTC): current UTC day if >= 4 bars exist, else previous completed Asia
    cur_asia_mask = (ts15 >= cur_day_start) & (ts15 < cur_day_start + 8 * HOUR_MS)
    prev_asia_mask = (ts15 >= prev_day_start) & (ts15 < prev_day_start + 8 * HOUR_MS)
    asia_high: Optional[float] = None
    asia_low: Optional[float] = None
    if int(np.sum(cur_asia_mask)) >= 4:
        asia_high = float(np.max(hi15[cur_asia_mask]))
        asia_low = float(np.min(lo15[cur_asia_mask]))
    elif int(np.sum(prev_asia_mask)) >= 4:
        asia_high = float(np.max(hi15[prev_asia_mask]))
        asia_low = float(np.min(lo15[prev_asia_mask]))

    def _dist(lvl: Optional[float]) -> Optional[float]:
        if lvl is None or close_now <= 0:
            return None
        return round(((lvl - close_now) / close_now) * 100.0, 2)

    d_pdh = _dist(pdh)
    d_pdl = _dist(pdl)
    d_ah = _dist(asia_high)
    d_al = _dist(asia_low)

    candidates = [
        ("PDH", pdh, d_pdh),
        ("PDL", pdl, d_pdl),
        ("ASIA_HIGH", asia_high, d_ah),
        ("ASIA_LOW", asia_low, d_al),
    ]
    valid_cands = [c for c in candidates if c[1] is not None and c[2] is not None]
    nearest: Optional[Dict[str, Any]] = None
    if valid_cands:
        best = min(valid_cands, key=lambda x: abs(float(x[2])))
        nearest = {"name": best[0], "price": round(float(best[1]), 6), "dist_pct": float(best[2])}

    is_long = direction.upper() == "LONG"
    obstacle = False
    if is_long and d_pdh is not None and 0.0 < d_pdh <= 0.45:
        obstacle = True
    elif (not is_long) and d_pdl is not None and -0.45 <= d_pdl < 0.0:
        obstacle = True

    return {
        "pdh": round(pdh, 6) if pdh is not None else None,
        "pdl": round(pdl, 6) if pdl is not None else None,
        "asia_high": round(asia_high, 6) if asia_high is not None else None,
        "asia_low": round(asia_low, 6) if asia_low is not None else None,
        "dist_pdh_pct": d_pdh,
        "dist_pdl_pct": d_pdl,
        "dist_asia_high_pct": d_ah,
        "dist_asia_low_pct": d_al,
        "nearest_session_level": nearest,
        "session_obstacle_ahead": obstacle,
    }


# ----------------------------------------------------------------------
# Advisor 1: Liquidity Sweep / Swing Failure Pattern (SFP)
# ----------------------------------------------------------------------
def _extract_swing_points(df: pd.DataFrame, lookback: int, left: int = 2, right: int = 2) -> Tuple[List[float], List[float]]:
    """Extract confirmed swing highs & lows from prior bars (excluding the most recent bar)."""
    if df is None or len(df) < left + right + 3:
        return [], []
    # Exclude the latest closed bar so we evaluate sweeps against PRIOR structure
    sub = df.iloc[:-1].tail(lookback).reset_index(drop=True)
    hi = sub["high"].to_numpy(dtype=np.float64)
    lo = sub["low"].to_numpy(dtype=np.float64)
    n = len(sub)
    highs: List[float] = []
    lows: List[float] = []
    for i in range(left, n - right):
        if hi[i] >= float(np.max(hi[i - left : i])) and hi[i] > float(np.max(hi[i + 1 : i + right + 1])):
            highs.append(float(hi[i]))
        if lo[i] <= float(np.min(lo[i - left : i])) and lo[i] < float(np.min(lo[i + 1 : i + right + 1])):
            lows.append(float(lo[i]))
    if n >= 5:
        highs.append(float(np.max(hi)))
        lows.append(float(np.min(lo)))
    return highs, lows


def detect_liquidity_sweep(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    session_levels: Dict[str, Any],
    direction: str,
    rvol_min: float = 1.2,
    check_last_bars: int = 3,
) -> Dict[str, Any]:
    """
    Check the last 1-3 closed 15m candles for a Liquidity Sweep / SFP against:
      - 1H swings (last ~48h)
      - 4H swings (last ~6-7 days = 42 bars)
      - PDH / PDL
    """
    none_res: Dict[str, Any] = {
        "sweep_type": "NONE",
        "swept_level": None,
        "swept_tf": None,
        "bars_ago": None,
        "with_volume": False,
        "aligns_with_direction": False,
        "trap_warning": False,
    }
    if df15 is None or len(df15) < 25:
        return none_res

    h1_hi, h1_lo = _extract_swing_points(df1h, lookback=48, left=2, right=2)
    h4_hi, h4_lo = _extract_swing_points(df4h, lookback=42, left=2, right=2)

    # Tagged resistance (high) and support (low) pools: (tf_label, priority_rank, level)
    high_levels: List[Tuple[str, int, float]] = []
    low_levels: List[Tuple[str, int, float]] = []
    for lvl in h4_hi:
        high_levels.append(("4H", 3, lvl))
    for lvl in h4_lo:
        low_levels.append(("4H", 3, lvl))
    if session_levels.get("pdh") is not None:
        high_levels.append(("PDH", 2, float(session_levels["pdh"])))
    if session_levels.get("pdl") is not None:
        low_levels.append(("PDL", 2, float(session_levels["pdl"])))
    for lvl in h1_hi:
        high_levels.append(("1H", 1, lvl))
    for lvl in h1_lo:
        low_levels.append(("1H", 1, lvl))

    vol = df15["volume"].astype(float)
    vol_sma = vol.rolling(20, min_periods=10).mean().to_numpy(dtype=np.float64)
    hi15 = df15["high"].to_numpy(dtype=np.float64)
    lo15 = df15["low"].to_numpy(dtype=np.float64)
    cl15 = df15["close"].to_numpy(dtype=np.float64)
    v15 = vol.to_numpy(dtype=np.float64)
    n = len(df15)

    best_key: Optional[Tuple[int, int, int]] = None
    best_found: Optional[Dict[str, Any]] = None
    is_long = direction.upper() == "LONG"

    for bars_ago in range(min(check_last_bars, n)):
        idx = n - 1 - bars_ago
        bar_h, bar_l, bar_c = hi15[idx], lo15[idx], cl15[idx]
        bar_rvol = (v15[idx] / vol_sma[idx]) if (np.isfinite(vol_sma[idx]) and vol_sma[idx] > 0) else 1.0
        with_vol = bool(bar_rvol >= rvol_min)

        # BULL_SFP: wick broke below significant low, candle closed back above it
        for tf_lbl, tf_rank, lvl in low_levels:
            if bar_l < lvl * 0.9999 and bar_c > lvl * 1.0001:
                key = (tf_rank, 1 if with_vol else 0, -bars_ago)
                if best_key is None or key > best_key:
                    best_key = key
                    best_found = {
                        "sweep_type": "BULL_SFP",
                        "swept_level": round(lvl, 6),
                        "swept_tf": tf_lbl,
                        "bars_ago": bars_ago,
                        "with_volume": with_vol,
                        "aligns_with_direction": is_long,
                        "trap_warning": not is_long,
                    }

        # BEAR_SFP: wick broke above significant high, candle closed back below it
        for tf_lbl, tf_rank, lvl in high_levels:
            if bar_h > lvl * 1.0001 and bar_c < lvl * 0.9999:
                key = (tf_rank, 1 if with_vol else 0, -bars_ago)
                if best_key is None or key > best_key:
                    best_key = key
                    best_found = {
                        "sweep_type": "BEAR_SFP",
                        "swept_level": round(lvl, 6),
                        "swept_tf": tf_lbl,
                        "bars_ago": bars_ago,
                        "with_volume": with_vol,
                        "aligns_with_direction": not is_long,
                        "trap_warning": is_long,
                    }

    return best_found if best_found is not None else none_res


# ----------------------------------------------------------------------
# Advisor 2: 4-Day Volume Profile (VPVR: POC, VAH, VAL)
# ----------------------------------------------------------------------
def compute_volume_profile(
    df15: pd.DataFrame,
    close_now: float,
    direction: str,
    bins: int = 50,
    value_area_pct: float = 70.0,
) -> Dict[str, Any]:
    """
    Build a 4-day fixed-range Volume Profile over all loaded closed 15m candles,
    uniformly distributing each candle's volume across bins overlapping [low, high].
    """
    empty: Dict[str, Any] = {
        "poc": None,
        "vah": None,
        "val": None,
        "price_zone": "UNKNOWN",
        "dist_to_poc_pct": None,
        "poc_obstacle_ahead": False,
    }
    if df15 is None or len(df15) < 20 or close_now <= 0:
        return empty

    hi = df15["high"].to_numpy(dtype=np.float64)
    lo = df15["low"].to_numpy(dtype=np.float64)
    vol = df15["volume"].to_numpy(dtype=np.float64)

    p_min = float(np.min(lo))
    p_max = float(np.max(hi))
    if not (np.isfinite(p_min) and np.isfinite(p_max)) or p_max <= p_min:
        return empty

    n_bins = max(10, int(bins))
    edges = np.linspace(p_min, p_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    step = (p_max - p_min) / n_bins
    bin_vols = np.zeros(n_bins, dtype=np.float64)

    for i in range(len(vol)):
        v = vol[i]
        if v <= 0.0 or not np.isfinite(v):
            continue
        i0 = int(np.floor((lo[i] - p_min) / step))
        i1 = int(np.floor((hi[i] - p_min) / step))
        i0 = max(0, min(n_bins - 1, i0))
        i1 = max(i0, min(n_bins - 1, i1))
        cnt = i1 - i0 + 1
        bin_vols[i0 : i1 + 1] += v / cnt

    tot_vol = float(np.sum(bin_vols))
    if tot_vol <= 0.0:
        return empty

    poc_idx = int(np.argmax(bin_vols))
    poc = float(centers[poc_idx])

    # Expand 70% Value Area outward from POC
    target = tot_vol * (max(10.0, min(95.0, float(value_area_pct))) / 100.0)
    lo_i = poc_idx
    hi_i = poc_idx
    acc = float(bin_vols[poc_idx])
    while acc < target and (lo_i > 0 or hi_i < n_bins - 1):
        v_dn = float(bin_vols[lo_i - 1]) if lo_i > 0 else -1.0
        v_up = float(bin_vols[hi_i + 1]) if hi_i < n_bins - 1 else -1.0
        if v_up >= v_dn and hi_i < n_bins - 1:
            hi_i += 1
            acc += float(bin_vols[hi_i])
        elif lo_i > 0:
            lo_i -= 1
            acc += float(bin_vols[lo_i])
        else:
            break

    val = float(edges[lo_i])
    vah = float(edges[hi_i + 1])

    if close_now > vah:
        zone = "ABOVE_VAH"
    elif close_now < val:
        zone = "BELOW_VAL"
    else:
        zone = "IN_VALUE"

    dist_poc = round(((poc - close_now) / close_now) * 100.0, 2)
    is_long = direction.upper() == "LONG"
    poc_obstacle = (0.15 <= dist_poc <= 1.2) if is_long else (-1.2 <= dist_poc <= -0.15)

    return {
        "poc": round(poc, 6),
        "vah": round(vah, 6),
        "val": round(val, 6),
        "price_zone": zone,
        "dist_to_poc_pct": dist_poc,
        "poc_obstacle_ahead": poc_obstacle,
    }


# ----------------------------------------------------------------------
# Combined Entrypoint + Priority Adjustment + PNG Chart Header Banner
# ----------------------------------------------------------------------
def analyze_tv_advisors(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    direction: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all enabled TradingView-style advisors on already-loaded closed candles (zero HTTP calls)."""
    close_now = float(df15["close"].iloc[-1]) if (df15 is not None and not df15.empty) else 0.0

    sess_enabled = bool(config.get("tv_session_enabled", True))
    sfp_enabled = bool(config.get("tv_sfp_enabled", True))
    vp_enabled = bool(config.get("tv_vp_enabled", True))

    # Session levels also supply PDH/PDL to SFP detector when either is enabled
    sess_block = (
        compute_session_levels(df15, df1h, close_now, direction)
        if (sess_enabled or sfp_enabled)
        else {"nearest_session_level": None, "session_obstacle_ahead": False}
    )
    sfp_block = (
        detect_liquidity_sweep(df15, df1h, df4h, sess_block, direction, rvol_min=1.2)
        if sfp_enabled
        else {"sweep_type": "NONE", "aligns_with_direction": False, "trap_warning": False}
    )
    vp_block = (
        compute_volume_profile(
            df15,
            close_now,
            direction,
            bins=int(config.get("vp_bins", 50)),
            value_area_pct=float(config.get("vp_value_area_pct", 70.0)),
        )
        if vp_enabled
        else {"poc": None, "vah": None, "val": None, "price_zone": "DISABLED", "dist_to_poc_pct": None, "poc_obstacle_ahead": False}
    )

    return {
        "liquidity_sweep": sfp_block,
        "volume_profile": vp_block,
        "session_levels": sess_block if sess_enabled else {"disabled": True, "nearest_session_level": None, "session_obstacle_ahead": False},
    }


def compute_tv_advisors_priority_delta(tv_block: Optional[Dict[str, Any]]) -> float:
    """Soft advisory priority_score adjustment for sorting top-3 candidates per bar (never blocks)."""
    if not tv_block:
        return 0.0
    delta = 0.0
    sfp = tv_block.get("liquidity_sweep") or {}
    if sfp.get("aligns_with_direction"):
        delta += 15.0 if sfp.get("with_volume") else 9.0
    elif sfp.get("trap_warning"):
        delta -= 20.0 if sfp.get("with_volume") else 14.0

    vp = tv_block.get("volume_profile") or {}
    if vp.get("poc_obstacle_ahead"):
        delta -= 10.0

    sess = tv_block.get("session_levels") or {}
    if sess.get("session_obstacle_ahead"):
        delta -= 10.0

    return delta


def format_tv_advisors_banner(tv_block: Optional[Dict[str, Any]]) -> str:
    """Format compact single-line status string for PNG chart header below DIV line."""
    if not tv_block:
        return "ADV: SFP=NONE | VP=NONE | SESS=NONE"

    sfp = tv_block.get("liquidity_sweep") or {}
    st = str(sfp.get("sweep_type", "NONE"))
    if st == "BULL_SFP":
        sfp_txt = f"BULL({sfp.get('swept_tf', '?')})"
    elif st == "BEAR_SFP":
        sfp_txt = f"BEAR({sfp.get('swept_tf', '?')})"
    else:
        sfp_txt = "NONE"

    vp = tv_block.get("volume_profile") or {}
    zone = str(vp.get("price_zone", "NONE"))
    dpoc = vp.get("dist_to_poc_pct")
    vp_txt = f"{zone} (POC {dpoc:+.1f}%)" if dpoc is not None else zone

    sess = tv_block.get("session_levels") or {}
    near = sess.get("nearest_session_level")
    if isinstance(near, dict) and near.get("name") is not None and near.get("dist_pct") is not None:
        sess_txt = f"{near['name']} {float(near['dist_pct']):+.1f}%"
    else:
        sess_txt = "NONE"

    return f"ADV: SFP={sfp_txt} | VP={vp_txt} | SESS={sess_txt}"
