"""
VIKA_N2_N7 Signal & Filter Engine — 100% formula-identical to the verified backtests
(`backtest_top100_crypto_keys_wr80.py` & `analyze_vika_65stops_2m_15m.py`).

Strictly operates on CLOSED candles only (15m, 1H, 4H) — zero lookahead.
Implements:
  - 4 TA Keys: K1_EMA_TREND_PB, K6_VWAP_RECLAIM, K7_SFP_LIQ_SWEEP, K8_SUPERTREND_BO
  - Filter 1: 4H STRONG opposing Regular Divergence (>= 3/4 of RSI14, MACD_HIST, OBV, MFI14)
  - Filter 2 (N2): Anti-Climax 15m (ADX15 >= 40 | BBW15 >= 6.5% | ATR15% >= 1.5%)
  - Filter 3 (N7): Funding Rate crowding (LONG blocked if FR >= +1.5 bps; SHORT blocked if FR <= -1.5 bps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

M15_MS = 15 * 60 * 1000
H1_MS = 60 * 60 * 1000
H4_MS = 4 * 60 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000

KEY_SHORT_NAMES: Dict[str, str] = {
    "K1_EMA_TREND_PB": "K1 (EMA Pullback)",
    "K3_MACD_STOCH": "K3 (MACD+Stoch)",
    "K6_VWAP_RECLAIM": "K6 (VWAP Reclaim)",
    "K7_SFP_LIQ_SWEEP": "K7 (SFP Sweep)",
    "K8_SUPERTREND_BO": "K8 (Supertrend BO)",
}


@dataclass
class CoinEvaluation:
    symbol: str
    key: str
    bar_ts: int
    close_price: float
    rvol: float
    adx15: float
    bbw15: float
    atr15_pct: float
    funding_rate: float
    funding_bps: float
    div4h_bear_strong: bool
    div4h_bull_strong: bool
    n2_blocked: bool
    n2_reasons: Tuple[str, ...] = field(default_factory=tuple)
    n7_blocked_long: bool = False
    n7_blocked_short: bool = False
    slope4h_pct: float = 0.0
    t4_blocked_long: bool = False
    t4_blocked_short: bool = False
    raw_signal: Optional[str] = None     # "LONG" | "SHORT" | None (before filters)
    final_signal: Optional[str] = None   # "LONG" | "SHORT" | None (after ALL filters)
    block_reason: Optional[str] = None   # why raw_signal was blocked, if any

    @property
    def div4h_label(self) -> str:
        if self.div4h_bear_strong and self.div4h_bull_strong:
            return "BEAR+BULL STRONG"
        if self.div4h_bear_strong:
            return "BEAR STRONG (блок LONG)"
        if self.div4h_bull_strong:
            return "BULL STRONG (блок SHORT)"
        return "OK"

    @property
    def n2_label(self) -> str:
        if self.n2_blocked:
            return f"BLOCK ({', '.join(self.n2_reasons)})"
        return "OK"

    @property
    def funding_label(self) -> str:
        s = f"{self.funding_bps:+.2f} bps"
        if self.n7_blocked_long:
            return f"{s} (блок LONG)"
        if self.n7_blocked_short:
            return f"{s} (блок SHORT)"
        return f"{s} (OK)"


# ---------------------------------------------------------------------------
# Exact Indicator Primitives (identical to verified backtest scripts)
# ---------------------------------------------------------------------------

def ema_arr(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(float)


def atr_arr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, period: int = 14) -> np.ndarray:
    pc = np.roll(cl, 1)
    pc[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)


def rsi_arr(cl: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(cl, prepend=cl[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag = pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)
    al = pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)
    rs = ag / np.maximum(al, 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def macd_arr(cl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    line = ema_arr(cl, 12) - ema_arr(cl, 26)
    sig = ema_arr(line, 9)
    return line, sig, line - sig


def stoch_arr(
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    k_len: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic(14,3,3) formula-identical to backtest_tv_daily_top100_wr80_15m.py."""
    hh = pd.Series(hi).rolling(k_len, min_periods=k_len).max().to_numpy(float)
    ll = pd.Series(lo).rolling(k_len, min_periods=k_len).min().to_numpy(float)
    raw_k = np.where((hh - ll) > 1e-12, (cl - ll) / np.maximum(hh - ll, 1e-12) * 100.0, 50.0)
    k = pd.Series(raw_k).rolling(smooth_k, min_periods=1).mean().fillna(50.0).to_numpy(float)
    d = pd.Series(k).rolling(smooth_d, min_periods=1).mean().fillna(50.0).to_numpy(float)
    return k, d


def obv_arr(cl: np.ndarray, vol: np.ndarray) -> np.ndarray:
    d = np.diff(cl, prepend=cl[0])
    return np.cumsum(np.where(d > 0, vol, np.where(d < 0, -vol, 0.0)))


def mfi_arr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, vol: np.ndarray, period: int = 14) -> np.ndarray:
    tp = (hi + lo + cl) / 3.0
    mf = tp * vol
    d = np.diff(tp, prepend=tp[0])
    pos = pd.Series(np.where(d > 0, mf, 0.0)).rolling(period, min_periods=period).sum().to_numpy(float)
    neg = pd.Series(np.where(d < 0, mf, 0.0)).rolling(period, min_periods=period).sum().to_numpy(float)
    mfr = pos / np.maximum(neg, 1e-12)
    return np.nan_to_num(100.0 - 100.0 / (1.0 + mfr), nan=50.0)


def adx_arr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, period: int = 14) -> np.ndarray:
    up = np.diff(hi, prepend=hi[0])
    dn = -np.diff(lo, prepend=lo[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = np.roll(cl, 1)
    pc[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)
    pdi = 100.0 * pd.Series(pdm).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float) / np.maximum(atr, 1e-12)
    mdi = 100.0 * pd.Series(mdm).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float) / np.maximum(atr, 1e-12)
    dx = 100.0 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    return pd.Series(dx).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy(float)


def supertrend_dir(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, atr: np.ndarray, mult: float = 3.0) -> np.ndarray:
    n = len(cl)
    hl2 = 0.5 * (hi + lo)
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    f_ub = np.copy(ub)
    f_lb = np.copy(lb)
    st_dir = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        f_ub[i] = ub[i] if (ub[i] < f_ub[i - 1] or cl[i - 1] > f_ub[i - 1]) else f_ub[i - 1]
        f_lb[i] = lb[i] if (lb[i] > f_lb[i - 1] or cl[i - 1] < f_lb[i - 1]) else f_lb[i - 1]
        if st_dir[i - 1] == 1:
            st_dir[i] = -1 if cl[i] < f_lb[i] else 1
        else:
            st_dir[i] = 1 if cl[i] > f_ub[i] else -1
    return st_dir


def compute_4h_strong_divergences(df4h: pd.DataFrame, hold_bars: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strictly causal 4H STRONG (>= 3/4 of RSI14, MACD_HIST, OBV, MFI14) opposing Regular Divergences.
    Returns (bear_strong_blocks_long, bull_strong_blocks_short).
    """
    n = len(df4h)
    hi = df4h["high"].to_numpy(float)
    lo = df4h["low"].to_numpy(float)
    cl = df4h["close"].to_numpy(float)
    vol = df4h["volume"].to_numpy(float)

    rsi = rsi_arr(cl, 14)
    _, _, mh = macd_arr(cl)
    obv = obv_arr(cl, vol)
    mfi = mfi_arr(hi, lo, cl, vol, 14)

    bear_3 = np.zeros(n, dtype=bool)
    bull_3 = np.zeros(n, dtype=bool)
    last_hi, last_lo = -1, -1

    for k in range(35, n):
        p = k - 2
        if hi[p] == np.max(hi[p - 3 : p + 3]) and hi[p] > hi[p - 1]:
            if last_hi >= 0 and 4 <= (p - last_hi) <= 50 and hi[p] > hi[last_hi]:
                votes = (
                    int(rsi[p] < rsi[last_hi] - 1.0)
                    + int(mh[p] < mh[last_hi])
                    + int(obv[p] < obv[last_hi])
                    + int(mfi[p] < mfi[last_hi] - 1.0)
                )
                if votes >= 3:
                    bear_3[k : min(n, k + hold_bars)] = True
            last_hi = p
        if lo[p] == np.min(lo[p - 3 : p + 3]) and lo[p] < lo[p - 1]:
            if last_lo >= 0 and 4 <= (p - last_lo) <= 50 and lo[p] < lo[last_lo]:
                votes = (
                    int(rsi[p] > rsi[last_lo] + 1.0)
                    + int(mh[p] > mh[last_lo])
                    + int(obv[p] > obv[last_lo])
                    + int(mfi[p] > mfi[last_lo] + 1.0)
                )
                if votes >= 3:
                    bull_3[k : min(n, k + hold_bars)] = True
            last_lo = p
    return bear_3, bull_3


def compute_session_levels(
    ts: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, vol: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strictly causal daily UTC VWAP, PDH, PDL, Asia High (00-08 UTC), Asia Low (00-08 UTC)."""
    n = len(ts)
    vwap = np.zeros(n, dtype=float)
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)
    asia_hi = np.full(n, np.nan)
    asia_lo = np.full(n, np.nan)

    cur_day = -1
    cum_pv = 0.0
    cum_v = 0.0
    day_hi, day_lo = -1e18, 1e18
    prev_hi, prev_lo = np.nan, np.nan
    cur_ah, cur_al = -1e18, 1e18
    ready_ah, ready_al = np.nan, np.nan

    for i in range(n):
        d = int(ts[i] // DAY_MS)
        hr = int((ts[i] % DAY_MS) // H1_MS)
        if d != cur_day:
            if cur_day != -1:
                prev_hi, prev_lo = day_hi, day_lo
            cur_day = d
            cum_pv = 0.0
            cum_v = 0.0
            day_hi, day_lo = hi[i], lo[i]
            cur_ah, cur_al = -1e18, 1e18
            ready_ah, ready_al = np.nan, np.nan

        tp = (hi[i] + lo[i] + cl[i]) / 3.0
        cum_pv += tp * vol[i]
        cum_v += vol[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else cl[i]

        if hr < 8:
            if hi[i] > cur_ah:
                cur_ah = hi[i]
            if lo[i] < cur_al:
                cur_al = lo[i]
        else:
            if np.isnan(ready_ah) and cur_ah > -1e17:
                ready_ah, ready_al = cur_ah, cur_al
            asia_hi[i] = ready_ah
            asia_lo[i] = ready_al

        if np.isfinite(prev_hi):
            pdh[i] = prev_hi
            pdl[i] = prev_lo

        if hi[i] > day_hi:
            day_hi = hi[i]
        if lo[i] < day_lo:
            day_lo = lo[i]

    return vwap, pdh, pdl, asia_hi, asia_lo


def drop_unclosed(df: pd.DataFrame, interval_ms: int, now_ms: int) -> pd.DataFrame:
    """Drop any currently forming candle strictly enforcing closed-candle discipline."""
    if df.empty:
        return df
    out = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    closed = out[(out["timestamp"].astype("int64") + interval_ms) <= now_ms].reset_index(drop=True)
    return closed


# ---------------------------------------------------------------------------
# Main Per-Coin Signal & Filter Evaluator
# ---------------------------------------------------------------------------

def evaluate_coin(
    symbol: str,
    key: str,
    df15_raw: pd.DataFrame,
    df1h_raw: pd.DataFrame,
    df4h_raw: pd.DataFrame,
    funding_rate: float,
    now_ms: int,
    filters_cfg: Dict[str, Any],
) -> CoinEvaluation:
    """
    Evaluate the latest CLOSED 15m candle for `symbol` using its assigned VIKA key
    and the 3 mandatory filters (4H STRONG div, N2 Anti-Climax, N7 Funding).
    """
    df15 = drop_unclosed(df15_raw, M15_MS, now_ms)
    df1h = drop_unclosed(df1h_raw, H1_MS, now_ms)
    df4h = drop_unclosed(df4h_raw, H4_MS, now_ms)

    if len(df15) < 100 or len(df1h) < 60 or len(df4h) < 60:
        raise ValueError(
            f"Insufficient closed candles for {symbol}: 15m={len(df15)}, 1h={len(df1h)}, 4h={len(df4h)}"
        )

    ts = df15["timestamp"].to_numpy(np.int64)
    close_ts = ts + M15_MS
    op = df15["open"].to_numpy(float)
    hi = df15["high"].to_numpy(float)
    lo = df15["low"].to_numpy(float)
    cl = df15["close"].to_numpy(float)
    vol = df15["volume"].to_numpy(float)

    # 1H trend (strictly closed 1H bar <= close_ts of last closed 15m bar)
    c1 = df1h["close"].to_numpy(float)
    e20_1, e50_1 = ema_arr(c1, 20), ema_arr(c1, 50)
    h1_up_raw = (c1 > e50_1) & (e20_1 > e50_1)
    h1_dn_raw = (c1 < e50_1) & (e20_1 < e50_1)
    h1_cts = df1h["timestamp"].to_numpy(np.int64) + H1_MS
    i1 = int(np.searchsorted(h1_cts, close_ts[-1], side="right") - 1)
    h1_up = bool(h1_up_raw[i1]) if i1 >= 50 else False
    h1_dn = bool(h1_dn_raw[i1]) if i1 >= 50 else False

    # 4H trend + 4H EMA50 slope (T4) + 4H STRONG opposing divergence (strictly closed 4H bar <= close_ts)
    c4 = df4h["close"].to_numpy(float)
    e20_4, e50_4 = ema_arr(c4, 20), ema_arr(c4, 50)
    h4_up_raw = (c4 > e50_4) & (e20_4 > e50_4)
    h4_dn_raw = (c4 < e50_4) & (e20_4 < e50_4)
    e50_4_p3 = np.roll(e50_4, 3)
    e50_4_p3[:3] = e50_4[0]
    slope4_arr = (e50_4 / np.maximum(e50_4_p3, 1e-12) - 1.0) * 100.0
    hold_bars = int(filters_cfg.get("div_4h_hold_bars", 8))
    bear4_str_arr, bull4_str_arr = compute_4h_strong_divergences(df4h, hold_bars=hold_bars)
    h4_cts = df4h["timestamp"].to_numpy(np.int64) + H4_MS
    i4 = int(np.searchsorted(h4_cts, close_ts[-1], side="right") - 1)
    h4_up = bool(h4_up_raw[i4]) if i4 >= 50 else False
    h4_dn = bool(h4_dn_raw[i4]) if i4 >= 50 else False
    c_slope4h = float(slope4_arr[i4]) if i4 >= 50 else 0.0
    div4h_bear_strong = bool(bear4_str_arr[i4]) if i4 >= 35 else False
    div4h_bull_strong = bool(bull4_str_arr[i4]) if i4 >= 35 else False

    # 15m Indicators (exact backtest formulas)
    e9, e21, e50 = ema_arr(cl, 9), ema_arr(cl, 21), ema_arr(cl, 50)
    atr15 = atr_arr(hi, lo, cl, 14)
    atr15_pct_arr = np.where(cl > 0, atr15 / np.maximum(cl, 1e-12) * 100.0, 99.0)
    rsi = rsi_arr(cl, 14)
    _, _, mh = macd_arr(cl)
    st_k, st_d = stoch_arr(hi, lo, cl, 14, 3, 3)
    rvol_arr = (
        pd.Series(vol) / pd.Series(vol).rolling(20, min_periods=10).mean().replace(0.0, np.nan)
    ).fillna(0.0).to_numpy(float)
    adx15_arr = adx_arr(hi, lo, cl, 14)
    sd20 = pd.Series(cl).rolling(20, min_periods=20).std().fillna(0.0).to_numpy(float)
    bbw15_arr = 4.0 * sd20 / np.maximum(cl, 1e-12) * 100.0

    vwap, pdh, pdl, asia_hi, asia_lo = compute_session_levels(ts, hi, lo, cl, vol)
    hi20 = pd.Series(hi).shift(1).rolling(20, min_periods=20).max().fillna(0.0).to_numpy(float)
    lo20 = pd.Series(lo).shift(1).rolling(20, min_periods=20).min().fillna(0.0).to_numpy(float)
    hi16 = pd.Series(hi).shift(1).rolling(16, min_periods=16).max().fillna(0.0).to_numpy(float)
    lo16 = pd.Series(lo).shift(1).rolling(16, min_periods=16).min().fillna(0.0).to_numpy(float)
    st_d10 = supertrend_dir(hi, lo, cl, atr15, 3.0)

    # Latest closed 15m bar index
    i = len(ts) - 1
    c_op, c_hi, c_lo, c_cl = op[i], hi[i], lo[i], cl[i]
    c_cl_p = cl[i - 1]
    c_rng = max(c_hi - c_lo, 1e-12)
    u_wick = (c_hi - max(c_op, c_cl)) / c_rng
    l_wick = (min(c_op, c_cl) - c_lo) / c_rng
    c_rvol = float(rvol_arr[i])
    c_adx15 = float(adx15_arr[i])
    c_bbw15 = float(bbw15_arr[i])
    c_atr15_pct = float(atr15_pct_arr[i])
    ok_vol = (c_atr15_pct > 0.05) and (c_atr15_pct <= 2.2)

    # Raw key signal (before N2, N7, 4H STRONG div)
    sig_l = False
    sig_s = False
    if key == "K1_EMA_TREND_PB":
        sig_l = (
            h1_up and h4_up and ok_vol
            and ((c_lo <= e21[i] * 1.002) or (c_lo <= e50[i] * 1.002))
            and (c_cl > e21[i]) and (c_cl > c_op)
            and (rsi[i - 1] <= 48.0) and (rsi[i] > rsi[i - 1])
        )
        sig_s = (
            h1_dn and h4_dn and ok_vol
            and ((c_hi >= e21[i] * 0.998) or (c_hi >= e50[i] * 0.998))
            and (c_cl < e21[i]) and (c_cl < c_op)
            and (rsi[i - 1] >= 52.0) and (rsi[i] < rsi[i - 1])
        )
    elif key == "K3_MACD_STOCH":
        sig_l = (
            h4_up and ok_vol
            and (mh[i] > mh[i - 1]) and (mh[i - 1] <= mh[i - 2])
            and (st_k[i] > st_d[i]) and (st_k[i] <= 65.0)
            and (c_rvol >= 1.1) and (c_cl > c_op)
        )
        sig_s = (
            h4_dn and ok_vol
            and (mh[i] < mh[i - 1]) and (mh[i - 1] >= mh[i - 2])
            and (st_k[i] < st_d[i]) and (st_k[i] >= 35.0)
            and (c_rvol >= 1.1) and (c_cl < c_op)
        )
    elif key == "K6_VWAP_RECLAIM":
        sig_l = (
            h1_up and ok_vol
            and ((c_lo < vwap[i]) or (c_cl_p < vwap[i]))
            and (c_cl > vwap[i]) and (e9[i] > e21[i])
            and (c_rvol >= 1.2) and (c_cl > c_op)
        )
        sig_s = (
            h1_dn and ok_vol
            and ((c_hi > vwap[i]) or (c_cl_p > vwap[i]))
            and (c_cl < vwap[i]) and (e9[i] < e21[i])
            and (c_rvol >= 1.2) and (c_cl < c_op)
        )
    elif key == "K7_SFP_LIQ_SWEEP":
        sw_l = (
            ((lo20[i] > 0) and (c_lo < lo20[i]) and (c_cl > lo20[i]))
            or (np.isfinite(pdl[i]) and (c_lo < pdl[i]) and (c_cl > pdl[i]))
            or (np.isfinite(asia_lo[i]) and (c_lo < asia_lo[i]) and (c_cl > asia_lo[i]))
        )
        sw_s = (
            ((hi20[i] > 0) and (c_hi > hi20[i]) and (c_cl < hi20[i]))
            or (np.isfinite(pdh[i]) and (c_hi > pdh[i]) and (c_cl < pdh[i]))
            or (np.isfinite(asia_hi[i]) and (c_hi > asia_hi[i]) and (c_cl < asia_hi[i]))
        )
        sig_l = h4_up and ok_vol and sw_l and (l_wick >= 0.30) and (c_rvol >= 1.25)
        sig_s = h4_dn and ok_vol and sw_s and (u_wick >= 0.30) and (c_rvol >= 1.25)
    elif key == "K8_SUPERTREND_BO":
        sig_l = (
            h1_up and h4_up and ok_vol
            and (((st_d10[i] == 1) and (st_d10[i - 1] == -1)) or ((hi16[i] > 0) and (c_cl > hi16[i]) and (c_cl_p <= hi16[i])))
            and (c_rvol >= 1.25) and (c_cl > c_op)
        )
        sig_s = (
            h1_dn and h4_dn and ok_vol
            and (((st_d10[i] == -1) and (st_d10[i - 1] == 1)) or ((lo16[i] > 0) and (c_cl < lo16[i]) and (c_cl_p >= lo16[i])))
            and (c_rvol >= 1.25) and (c_cl < c_op)
        )
    else:
        raise ValueError(f"Unsupported VIKA key: {key}")

    raw_signal: Optional[str] = "LONG" if sig_l else ("SHORT" if sig_s else None)

    # Filter N2: Anti-Climax 15m
    adx_max = float(filters_cfg.get("n2_adx15_max", 40.0))
    bbw_max = float(filters_cfg.get("n2_bbw15_max", 6.5))
    atr_max = float(filters_cfg.get("n2_atr15_pct_max", 1.5))
    n2_reasons = []
    if c_adx15 >= adx_max:
        n2_reasons.append(f"ADX15={c_adx15:.1f}>={adx_max:g}")
    if c_bbw15 >= bbw_max:
        n2_reasons.append(f"BBW15={c_bbw15:.2f}%>={bbw_max:g}%")
    if c_atr15_pct >= atr_max:
        n2_reasons.append(f"ATR15={c_atr15_pct:.2f}%>={atr_max:g}%")
    n2_blocked = len(n2_reasons) > 0

    # Filter N7: Funding Rate crowding
    fr_bps = float(funding_rate) * 10000.0
    max_pay_bps = float(filters_cfg.get("n7_max_pay_funding_bps", 1.5))
    n7_blocked_long = fr_bps >= max_pay_bps
    n7_blocked_short = (-fr_bps) >= max_pay_bps

    # Filter T4: 4H EMA50 aligned slope > +0.15% for trend keys
    min_slope4 = float(filters_cfg.get("t4_min_4h_ema50_slope_pct", 0.15))
    is_trend_key = key != "K5_BB_MEAN_REV"
    t4_blocked_long = is_trend_key and (c_slope4h <= min_slope4)
    t4_blocked_short = is_trend_key and ((-c_slope4h) <= min_slope4)

    # Apply all mandatory filters (4H STRONG Div, N2, N7, T4)
    final_signal = raw_signal
    block_reason: Optional[str] = None
    if raw_signal == "LONG":
        if filters_cfg.get("div_4h_strong", True) and div4h_bear_strong:
            final_signal = None
            block_reason = "4H STRONG Bearish Divergence"
        elif n2_blocked:
            final_signal = None
            block_reason = f"N2 Anti-Climax ({', '.join(n2_reasons)})"
        elif n7_blocked_long:
            final_signal = None
            block_reason = f"N7 Funding ({fr_bps:+.2f} bps >= +{max_pay_bps:g})"
        elif t4_blocked_long:
            final_signal = None
            block_reason = f"T4 4H Slope ({c_slope4h:+.2f}% <= +{min_slope4:g}%)"
    elif raw_signal == "SHORT":
        if filters_cfg.get("div_4h_strong", True) and div4h_bull_strong:
            final_signal = None
            block_reason = "4H STRONG Bullish Divergence"
        elif n2_blocked:
            final_signal = None
            block_reason = f"N2 Anti-Climax ({', '.join(n2_reasons)})"
        elif n7_blocked_short:
            final_signal = None
            block_reason = f"N7 Funding ({fr_bps:+.2f} bps <= -{max_pay_bps:g})"
        elif t4_blocked_short:
            final_signal = None
            block_reason = f"T4 4H Slope ({c_slope4h:+.2f}%, aligned {-c_slope4h:+.2f}% <= +{min_slope4:g}%)"

    return CoinEvaluation(
        symbol=symbol,
        key=key,
        bar_ts=int(ts[i]),
        close_price=round(float(c_cl), 8),
        rvol=round(c_rvol, 2),
        adx15=round(c_adx15, 2),
        bbw15=round(c_bbw15, 2),
        atr15_pct=round(c_atr15_pct, 3),
        funding_rate=float(funding_rate),
        funding_bps=round(fr_bps, 2),
        div4h_bear_strong=div4h_bear_strong,
        div4h_bull_strong=div4h_bull_strong,
        n2_blocked=n2_blocked,
        n2_reasons=tuple(n2_reasons),
        n7_blocked_long=n7_blocked_long,
        n7_blocked_short=n7_blocked_short,
        slope4h_pct=round(c_slope4h, 2),
        t4_blocked_long=t4_blocked_long,
        t4_blocked_short=t4_blocked_short,
        raw_signal=raw_signal,
        final_signal=final_signal,
        block_reason=block_reason,
    )
