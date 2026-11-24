"""
Multi-Indicator Divergence Engine (`ai_trader_v2/divergences.py`), inspired by
TradingView "Divergence for Many Indicators v4" across 3 timeframes (15m, 1H, 4H).

Strict Design Guarantees:
  - Uses ONLY already-loaded closed klines (zero extra Bybit HTTP requests).
  - Evaluates 4 independent indicators from 4 distinct groups:
      1) RSI(14)            — momentum oscillator
      2) MACD_HIST(12,26,9) — trend-momentum histogram
      3) OBV                — cumulative volume flow
      4) MFI(14)            — price-volume money flow oscillator
  - Detects BOTH Regular (reversal) and Hidden (trend-continuation) divergences.
  - Zero-repaint protection via confirmed asymmetric pivots strictly on closed bars:
      * 15m: left=4, right=2, freshness window <= 8 bars (2h)
      * 1H : left=4, right=2, freshness window <= 8 bars (8h)
      * 4H : left=3, right=2, freshness window <= 6 bars (24h)
  - Advisory only: NEVER hard-blocks entries in Python (feeds AI Judge JSON, PNG chart header,
    and candidate priority_score).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

INDICATOR_NAMES: Tuple[str, ...] = ("RSI", "MACD_HIST", "OBV", "MFI")

TF_PIVOT_SPECS: Dict[str, Dict[str, int]] = {
    "15m": {"left": 4, "right": 2, "max_bars_ago": 8},
    "1H": {"left": 4, "right": 2, "max_bars_ago": 8},
    "4H": {"left": 3, "right": 2, "max_bars_ago": 6},
}

_SHORT_TYPE_LABELS: Dict[str, str] = {
    "REGULAR_BULL": "BULL",
    "REGULAR_BEAR": "BEAR",
    "HIDDEN_BULL": "H_BULL",
    "HIDDEN_BEAR": "H_BEAR",
    "NONE": "NONE",
}


def compute_rsi14(close: pd.Series, period: int = 14) -> np.ndarray:
    d = close.diff()
    ag = d.clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    al = (-d).clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rsi = (100.0 - 100.0 / (1.0 + ag / al.replace(0.0, np.nan))).fillna(50.0)
    return rsi.to_numpy(dtype=np.float64)


def compute_macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig).fillna(0.0).to_numpy(dtype=np.float64)


def compute_obv(close: pd.Series, volume: pd.Series) -> np.ndarray:
    c = close.to_numpy(dtype=np.float64)
    v = volume.to_numpy(dtype=np.float64)
    diff = np.diff(c, prepend=c[0]) if len(c) > 0 else np.array([], dtype=np.float64)
    signed = np.where(diff > 0.0, v, np.where(diff < 0.0, -v, 0.0))
    return np.cumsum(signed)


def compute_mfi14(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> np.ndarray:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    tp_diff = tp.diff()
    pos_mf = rmf.where(tp_diff > 0.0, 0.0).rolling(period, min_periods=period).sum()
    neg_mf = rmf.where(tp_diff < 0.0, 0.0).rolling(period, min_periods=period).sum()
    # When neg_mf == 0 and pos_mf > 0 -> MFI = 100; when both 0 -> MFI = 50
    ratio = pos_mf / neg_mf.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    mfi = np.where((neg_mf == 0.0) & (pos_mf > 0.0), 100.0, mfi)
    mfi = np.where((neg_mf == 0.0) & (pos_mf == 0.0), 50.0, mfi)
    return pd.Series(mfi).fillna(50.0).to_numpy(dtype=np.float64)


def find_confirmed_pivots(
    arr: np.ndarray,
    left: int,
    right: int,
    is_high: bool,
) -> List[int]:
    """
    Find strictly confirmed asymmetric pivots (zero repaint).
    Bar `i` is a valid pivot ONLY when `i <= len(arr) - 1 - right`, guaranteeing
    at least `right` subsequent closed candles have confirmed it.
    """
    n = len(arr)
    pivots: List[int] = []
    if n < left + right + 1:
        return pivots
    end = n - right  # exclusive upper bound -> max index is n - 1 - right
    for i in range(left, end):
        val = arr[i]
        left_win = arr[i - left : i]
        right_win = arr[i + 1 : i + right + 1]
        if is_high:
            if val >= float(np.max(left_win)) and val > float(np.max(right_win)):
                pivots.append(i)
        else:
            if val <= float(np.min(left_win)) and val < float(np.min(right_win)):
                pivots.append(i)
    return pivots


def _no_div_entry() -> Dict[str, Any]:
    return {
        "type": "NONE",
        "count": 0,
        "total": len(INDICATOR_NAMES),
        "indicators": [],
        "strength": "NONE",
        "bars_ago": None,
    }


def detect_tf_divergence(
    df: pd.DataFrame,
    left: int = 4,
    right: int = 2,
    max_bars_ago: int = 8,
    min_indicators: int = 2,
    max_pivots_back: int = 4,
    max_span_bars: int = 50,
) -> Dict[str, Any]:
    """
    Detect confirmed Regular & Hidden divergences across RSI(14), MACD_HIST, OBV, MFI(14)
    on a single timeframe's closed candles.
    """
    if df is None or len(df) < max(35, left + right + 15):
        return _no_div_entry()

    hi = df["high"].to_numpy(dtype=np.float64)
    lo = df["low"].to_numpy(dtype=np.float64)
    n = len(df)

    inds: Dict[str, np.ndarray] = {
        "RSI": compute_rsi14(df["close"]),
        "MACD_HIST": compute_macd_hist(df["close"]),
        "OBV": compute_obv(df["close"], df["volume"]),
        "MFI": compute_mfi14(df["high"], df["low"], df["close"], df["volume"]),
    }

    piv_lows = find_confirmed_pivots(lo, left=left, right=right, is_high=False)
    piv_highs = find_confirmed_pivots(hi, left=left, right=right, is_high=True)

    # Candidate findings: (sort_key, entry_dict)
    # Priority: count desc -> REGULAR over HIDDEN -> smaller bars_ago
    best_tuple: Optional[Tuple[Tuple[int, int, int], Dict[str, Any]]] = None

    def _consider(div_type: str, matched: List[str], p2: int) -> None:
        nonlocal best_tuple
        cnt = len(matched)
        if cnt < min_indicators:
            return
        bars_ago = (n - 1) - p2
        is_reg = 1 if div_type.startswith("REGULAR_") else 0
        key = (cnt, is_reg, -bars_ago)
        entry = {
            "type": div_type,
            "count": cnt,
            "total": len(INDICATOR_NAMES),
            "indicators": matched,
            "strength": "STRONG" if cnt >= 3 else "MODERATE",
            "bars_ago": int(bars_ago),
        }
        if best_tuple is None or key > best_tuple[0]:
            best_tuple = (key, entry)

    # 1. Bullish divergences on confirmed Pivot Lows
    fresh_lows = [p for p in piv_lows if right <= (n - 1) - p <= max_bars_ago]
    for p2 in reversed(fresh_lows):
        idx2 = piv_lows.index(p2)
        prior_lows = piv_lows[max(0, idx2 - max_pivots_back) : idx2]
        for p1 in reversed(prior_lows):
            span = p2 - p1
            if span < 3 or span > max_span_bars:
                continue
            # Clean swing structure: no intermediate bar deeply undercutting both pivot lows
            mid_min = float(np.min(lo[p1 + 1 : p2]))
            if mid_min < min(lo[p1], lo[p2]):
                continue
            if lo[p2] < lo[p1]:
                matched = [name for name, arr in inds.items() if arr[p2] > arr[p1]]
                _consider("REGULAR_BULL", matched, p2)
            elif lo[p2] > lo[p1]:
                matched = [name for name, arr in inds.items() if arr[p2] < arr[p1]]
                _consider("HIDDEN_BULL", matched, p2)

    # 2. Bearish divergences on confirmed Pivot Highs
    fresh_highs = [p for p in piv_highs if right <= (n - 1) - p <= max_bars_ago]
    for p2 in reversed(fresh_highs):
        idx2 = piv_highs.index(p2)
        prior_highs = piv_highs[max(0, idx2 - max_pivots_back) : idx2]
        for p1 in reversed(prior_highs):
            span = p2 - p1
            if span < 3 or span > max_span_bars:
                continue
            mid_max = float(np.max(hi[p1 + 1 : p2]))
            if mid_max > max(hi[p1], hi[p2]):
                continue
            if hi[p2] > hi[p1]:
                matched = [name for name, arr in inds.items() if arr[p2] < arr[p1]]
                _consider("REGULAR_BEAR", matched, p2)
            elif hi[p2] < hi[p1]:
                matched = [name for name, arr in inds.items() if arr[p2] > arr[p1]]
                _consider("HIDDEN_BEAR", matched, p2)

    return best_tuple[1] if best_tuple is not None else _no_div_entry()


def analyze_mtf_divergences(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    df4h: pd.DataFrame,
    direction: str,
    min_indicators: int = 2,
) -> Dict[str, Any]:
    """
    Compute 15m, 1H, 4H multi-indicator divergences and directional advisory flags
    relative to candidate `direction` ("LONG" or "SHORT").
    """
    d15 = detect_tf_divergence(df15, min_indicators=min_indicators, **TF_PIVOT_SPECS["15m"])
    d1h = detect_tf_divergence(df1h, min_indicators=min_indicators, **TF_PIVOT_SPECS["1H"])
    d4h = detect_tf_divergence(df4h, min_indicators=min_indicators, **TF_PIVOT_SPECS["4H"])

    is_long = direction.upper() == "LONG"
    confirm_types = {"REGULAR_BULL", "HIDDEN_BULL"} if is_long else {"REGULAR_BEAR", "HIDDEN_BEAR"}
    oppose_reg_type = "REGULAR_BEAR" if is_long else "REGULAR_BULL"

    confirms_15m = d15["type"] in confirm_types
    confirms_1h = d1h["type"] in confirm_types
    opposes_1h = d1h["type"] == oppose_reg_type
    opposes_4h = d4h["type"] == oppose_reg_type

    return {
        "15m": d15,
        "1H": d1h,
        "4H": d4h,
        "summary": {
            "confirms_15m": confirms_15m,
            "confirms_1h": confirms_1h,
            "opposes_1h": opposes_1h,
            "opposes_4h": opposes_4h,
            "h4_exhaustion_warning": opposes_4h,
        },
    }


def compute_divergence_priority_delta(div_block: Dict[str, Any]) -> float:
    """Advisory priority_score adjustment for sorting top-3 candidates per bar (never blocks)."""
    if not div_block:
        return 0.0
    summ = div_block.get("summary") or {}
    delta = 0.0
    if summ.get("confirms_15m"):
        st15 = (div_block.get("15m") or {}).get("strength")
        delta += 12.0 if st15 == "STRONG" else 8.0
    if summ.get("confirms_1h"):
        st1h = (div_block.get("1H") or {}).get("strength")
        delta += 15.0 if st1h == "STRONG" else 10.0
    if summ.get("opposes_1h"):
        st1h = (div_block.get("1H") or {}).get("strength")
        delta -= 12.0 if st1h == "STRONG" else 8.0
    if summ.get("h4_exhaustion_warning"):
        st4h = (div_block.get("4H") or {}).get("strength")
        delta -= 25.0 if st4h == "STRONG" else 16.0
    return delta


def format_divergence_banner(div_block: Optional[Dict[str, Any]]) -> str:
    """Format compact single-line status string for PNG chart header."""
    if not div_block:
        return "DIV: 15m NONE | 1H NONE | 4H NONE"
    parts: List[str] = []
    for tf in ("15m", "1H", "4H"):
        item = div_block.get(tf) or {}
        t = str(item.get("type", "NONE"))
        short = _SHORT_TYPE_LABELS.get(t, t)
        if short == "NONE":
            parts.append(f"{tf} NONE")
        else:
            parts.append(f"{tf} {short}({int(item.get('count', 0))}/{int(item.get('total', 4))})")
    return "DIV: " + " | ".join(parts)
