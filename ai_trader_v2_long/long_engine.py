"""
VIKA_LONG_V1 Signal & Filter Engine — 100% formula-identical to the verified
`research_long_v3/deep_ta_scan.py` & `scan_expanded_universe.py` backtests.

Strictly operates on CLOSED candles only (15m, 1H, 4H) — zero lookahead.
Implements:
  - 14 Clean 15m LONG TA Keys (T1..T14) with their exact 1H/4H trend contexts
  - Macro Switch: BTC 4H UP (BTC 4H close > EMA50 & EMA20 > EMA50) — sleeps on bear markets!
  - Filter S1: Skip 08:00-12:00 UTC morning chop window
  - Filter S5: Anti-overextension (coin 4H close < 4H EMA50 * 1.10)
  - Volatility Guard: 0.10% <= ATR15% <= 1.80%
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd

M15_MS = 15 * 60 * 1000
H1_MS = 60 * 60 * 1000
H4_MS = 4 * 60 * 60 * 1000
DAY_MS = 24 * 60 * 60 * 1000

LONG_KEY_SHORT_NAMES: Dict[str, str] = {
    "T1_BB_BOUNCE": "T1 (BB Dip)",
    "T2_KELTNER_PULLBACK": "T2 (Keltner)",
    "T3_RSI7_OVERSOLD": "T3 (RSI7 Snap)",
    "T4_STOCH_CROSS_LOW": "T4 (Stoch Cross)",
    "T5_CCI_REBOUND": "T5 (CCI -100)",
    "T6_MFI_DIP_BUY": "T6 (MFI Dip)",
    "T7_VWAP_RECLAIM": "T7 (VWAP Reclaim)",
    "T8_EMA20_50_BOUNCE": "T8 (EMA20/50)",
    "T9_LIQ_SWEEP_HAMMER": "T9 (Liq Sweep)",
    "T10_2RED_ENGULF": "T10 (2-Red Engulf)",
    "T11_HL_PIVOT_BREAK": "T11 (HL Pivot)",
    "T12_COIL_BREAKOUT": "T12 (Coil BO)",
    "T13_MACD_HIST_TURN": "T13 (MACD Turn)",
    "T14_DONCHIAN_BO": "T14 (Donchian BO)",
    "N1_EMA20_RECLAIM_L": "N1 (EMA20 Reclaim)",
    "N2_BB_SQZ_REJECT_L": "N2 (BB Sqz Reject)",
    "K1_EMA_TREND_PB_L": "K1L (EMA Pullback)",
    "K3_MACD_STOCH_L": "K3L (MACD+Stoch)",
    "K6_VWAP_RECLAIM_L": "K6L (VWAP Reclaim)",
    "K7_SFP_LIQ_SWEEP_L": "K7L (SFP Sweep)",
    "K8_SUPERTREND_BO_L": "K8L (Supertrend BO)",
}


@dataclass
class BtcRegimeState:
    bar_ts_4h: int
    close_4h: float
    ema20_4h: float
    ema50_4h: float
    is_4h_up: bool
    dist_ema50_pct: float

    @property
    def label(self) -> str:
        if self.is_4h_up:
            return f"🟢 BTC 4H UP (${self.close_4h:,.0f}, EMA50 {self.dist_ema50_pct:+.1f}%)"
        return f"🔴 BTC 4H DOWN — СПИМ (${self.close_4h:,.0f}, EMA50 {self.dist_ema50_pct:+.1f}%)"


@dataclass
class LongCoinEvaluation:
    symbol: str
    key: str
    bar_ts: int
    close_price: float
    high_price: float
    rvol: float
    atr15_pct: float
    h1_up: bool
    h4_up: bool
    dist4h_ema50_pct: float
    btc_4h_up: bool
    utc_hour: int
    raw_signal: Optional[str] = None     # "LONG" | None
    final_signal: Optional[str] = None   # "LONG" | None
    block_reason: Optional[str] = None

    # Compatibility fields so shared VikaExecutor/VikaJournal work seamlessly
    adx15: float = 0.0
    bbw15: float = 0.0
    funding_rate: float = 0.0
    funding_bps: float = 0.0
    div4h_bear_strong: bool = False
    div4h_bull_strong: bool = False
    n2_blocked: bool = False
    n7_blocked_long: bool = False
    n7_blocked_short: bool = False
    slope4h_pct: float = 0.0
    retail_ls_z: float = 0.0
    retail_ls_available: bool = False
    retail_ls_blocked: bool = False
    btc_1d_up: bool = True

    @property
    def div4h_label(self) -> str:
        return f"4H_E50 {self.dist4h_ema50_pct:+.1f}%"

    @property
    def n2_label(self) -> str:
        return "BTC_4H_UP" if self.btc_4h_up else "BTC_4H_DOWN"

    @property
    def funding_label(self) -> str:
        return f"{self.funding_bps:+.2f} bps"


# ---------------------------------------------------------------------------
# Exact Indicator Primitives (identical to research_long_v3/deep_ta_scan.py)
# ---------------------------------------------------------------------------

def ema_arr(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy(float)


def rsi_arr(cl: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(cl, prepend=cl[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy(float)
    ad = pd.Series(dn).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy(float)
    rs = au / np.maximum(ad, 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def atr_arr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, n: int = 14) -> np.ndarray:
    pc = np.roll(cl, 1)
    pc[0] = cl[0]
    tr = np.maximum(hi - lo, np.maximum(np.abs(hi - pc), np.abs(lo - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy(float)


def stoch_arr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, k_len: int = 14, d_len: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    ll = pd.Series(lo).rolling(k_len, min_periods=k_len).min().to_numpy(float)
    hh = pd.Series(hi).rolling(k_len, min_periods=k_len).max().to_numpy(float)
    raw_k = np.where((hh - ll) > 1e-12, (cl - ll) / np.maximum(hh - ll, 1e-12) * 100.0, 50.0)
    k = pd.Series(raw_k).rolling(3, min_periods=1).mean().fillna(50.0).to_numpy(float)
    d = pd.Series(k).rolling(d_len, min_periods=1).mean().fillna(50.0).to_numpy(float)
    return k, d


def cci_arr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, n: int = 20) -> np.ndarray:
    tp = (hi + lo + cl) / 3.0
    s = pd.Series(tp)
    ma = s.rolling(n, min_periods=n).mean()
    md = s.rolling(n, min_periods=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return ((s - ma) / np.maximum(0.015 * md, 1e-12)).fillna(0.0).to_numpy(float)


def mfi_arr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, vol: np.ndarray, n: int = 14) -> np.ndarray:
    tp = (hi + lo + cl) / 3.0
    mf = tp * vol
    tp_p = np.roll(tp, 1)
    tp_p[0] = tp[0]
    pos = np.where(tp > tp_p, mf, 0.0)
    neg = np.where(tp < tp_p, mf, 0.0)
    ps = pd.Series(pos).rolling(n, min_periods=n).sum().to_numpy(float)
    ns = pd.Series(neg).rolling(n, min_periods=n).sum().to_numpy(float)
    mr = ps / np.maximum(ns, 1e-12)
    return np.nan_to_num(100.0 - 100.0 / (1.0 + mr), nan=50.0)


def daily_vwap_arr(ts: np.ndarray, hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, vol: np.ndarray) -> np.ndarray:
    tp = (hi + lo + cl) / 3.0
    days = ts // DAY_MS
    out = np.empty(len(ts), dtype=float)
    cv = 0.0
    cpv = 0.0
    cur = -1
    for i in range(len(ts)):
        if days[i] != cur:
            cur = days[i]
            cv = 0.0
            cpv = 0.0
        cv += vol[i]
        cpv += tp[i] * vol[i]
        out[i] = cpv / cv if cv > 0 else cl[i]
    return out


def strip_forming_bars(df: pd.DataFrame, tf_ms: int, now_ms: int) -> pd.DataFrame:
    """Keep strictly CLOSED candles (timestamp + tf_ms <= now_ms)."""
    if df.empty:
        return df
    out = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    closed = out[out["timestamp"].astype(np.int64) + tf_ms <= now_ms].reset_index(drop=True)
    return closed


def evaluate_btc_4h_regime(df_btc_4h_raw: pd.DataFrame, now_ms: int) -> BtcRegimeState:
    """Evaluate strictly closed BTC 4H trend switch (close > EMA50 & EMA20 > EMA50)."""
    df4 = strip_forming_bars(df_btc_4h_raw, H4_MS, now_ms)
    if len(df4) < 55:
        raise ValueError(f"Insufficient closed BTC 4H bars ({len(df4)} < 55)")
    c4 = df4["close"].to_numpy(float)
    e20 = ema_arr(c4, 20)
    e50 = ema_arr(c4, 50)
    idx = len(c4) - 1
    is_up = bool((c4[idx] > e50[idx]) and (e20[idx] > e50[idx]))
    dist_pct = float((c4[idx] - e50[idx]) / max(e50[idx], 1e-12) * 100.0)
    return BtcRegimeState(
        bar_ts_4h=int(df4["timestamp"].iloc[idx]),
        close_4h=float(c4[idx]),
        ema20_4h=float(e20[idx]),
        ema50_4h=float(e50[idx]),
        is_4h_up=is_up,
        dist_ema50_pct=round(dist_pct, 2),
    )


def evaluate_long_coin(
    symbol: str,
    key: str,
    df15_raw: pd.DataFrame,
    df1h_raw: pd.DataFrame,
    df4h_raw: pd.DataFrame,
    btc_regime: BtcRegimeState,
    funding_rate: float,
    now_ms: int,
    filters_cfg: Optional[Dict[str, Any]] = None,
    *,
    retail_ls_z: Optional[float] = None,
    btc_1d_up: Optional[bool] = None,
) -> LongCoinEvaluation:
    """Evaluate a single coin strictly on the latest CLOSED 15m/1H/4H candles."""
    fcfg = filters_cfg or {}
    btc_switch_on = bool(fcfg.get("btc_4h_up_switch", True))
    skip_hours: Set[int] = {int(h) for h in fcfg.get("skip_utc_hours", [8, 9, 10, 11])}
    max_dist4h = float(fcfg.get("max_coin_dist4h_ema50_pct", 10.0))
    min_atr = float(fcfg.get("min_atr15_pct", 0.10))
    max_atr = float(fcfg.get("max_atr15_pct", 1.80))

    df15 = strip_forming_bars(df15_raw, M15_MS, now_ms)
    df1h = strip_forming_bars(df1h_raw, H1_MS, now_ms)
    df4h = strip_forming_bars(df4h_raw, H4_MS, now_ms)
    if len(df15) < 60 or len(df1h) < 55 or len(df4h) < 55:
        raise ValueError(f"Insufficient closed bars for {symbol}: 15m={len(df15)}, 1h={len(df1h)}, 4h={len(df4h)}")

    ts = df15["timestamp"].to_numpy(np.int64)
    op = df15["open"].to_numpy(float)
    hi = df15["high"].to_numpy(float)
    lo = df15["low"].to_numpy(float)
    cl = df15["close"].to_numpy(float)
    vol = df15["volume"].to_numpy(float)
    idx = len(cl) - 1
    close_ts_last = int(ts[idx] + M15_MS)

    # Causal 1H trend aligned to close_ts_last
    c1 = df1h["close"].to_numpy(float)
    e20_1, e50_1 = ema_arr(c1, 20), ema_arr(c1, 50)
    i1 = int(np.searchsorted(df1h["timestamp"].to_numpy(np.int64) + H1_MS, close_ts_last, side="right") - 1)
    h1_up = bool(i1 >= 50 and (c1[i1] > e50_1[i1]) and (e20_1[i1] > e50_1[i1]))

    # Causal 4H trend & stretch aligned to close_ts_last
    c4 = df4h["close"].to_numpy(float)
    e20_4, e50_4 = ema_arr(c4, 20), ema_arr(c4, 50)
    i4 = int(np.searchsorted(df4h["timestamp"].to_numpy(np.int64) + H4_MS, close_ts_last, side="right") - 1)
    h4_up = bool(i4 >= 50 and (c4[i4] > e50_4[i4]) and (e20_4[i4] > e50_4[i4]))
    dist4h_pct = float((c4[i4] - e50_4[i4]) / max(e50_4[i4], 1e-12) * 100.0) if i4 >= 50 else 0.0

    # 15m indicators (exact deep_ta_scan.py formulas)
    e10, e20, e50 = ema_arr(cl, 10), ema_arr(cl, 20), ema_arr(cl, 50)
    a14 = atr_arr(hi, lo, cl, 14)
    atr_pct = a14 / np.maximum(cl, 1e-12) * 100.0
    r14 = rsi_arr(cl, 14)
    r7 = rsi_arr(cl, 7)
    st_k, st_d = stoch_arr(hi, lo, cl, 14, 3)
    cci20 = cci_arr(hi, lo, cl, 20)
    mfi14 = mfi_arr(hi, lo, cl, vol, 14)
    vwap = daily_vwap_arr(ts, hi, lo, cl, vol)
    macd_line = ema_arr(cl, 12) - ema_arr(cl, 26)
    mh = macd_line - ema_arr(macd_line, 9)
    rvol = (pd.Series(vol) / pd.Series(vol).rolling(20, min_periods=10).mean().replace(0.0, np.nan)).fillna(0.0).to_numpy(float)

    sma20 = pd.Series(cl).rolling(20, min_periods=20).mean().fillna(cl[0]).to_numpy(float)
    sd20 = pd.Series(cl).rolling(20, min_periods=20).std().fillna(0.0).to_numpy(float)
    bbw15 = 4.0 * sd20 / np.maximum(cl, 1e-12) * 100.0
    bb_lo = sma20 - 2.0 * sd20
    kc_lo = e20 - 1.5 * a14
    lo12 = pd.Series(lo).shift(1).rolling(12, min_periods=12).min().fillna(cl[0]).to_numpy(float)
    lo16 = pd.Series(lo).shift(1).rolling(16, min_periods=16).min().fillna(cl[0]).to_numpy(float)
    hi20 = pd.Series(hi).shift(1).rolling(20, min_periods=20).max().fillna(cl[0]).to_numpy(float)

    rng = max(hi[idx] - lo[idx], 1e-12)
    body_ratio = (cl[idx] - op[idx]) / rng
    l_wick = (min(op[idx], cl[idx]) - lo[idx]) / rng
    mid_p1 = 0.5 * (op[idx - 1] + cl[idx - 1])
    green = cl[idx] > op[idx]
    red_p1 = cl[idx - 1] < op[idx - 1]
    red_p2 = cl[idx - 2] < op[idx - 2]

    ok_v = (min_atr <= atr_pct[idx] <= max_atr)
    up_both = h1_up and h4_up and ok_v
    up_1h = h1_up and ok_v
    up_4h = h4_up and ok_v

    fired = False
    if key == "T1_BB_BOUNCE":
        fired = up_both and ((lo[idx] <= bb_lo[idx]) or (lo[idx - 1] <= bb_lo[idx - 1])) and green and (cl[idx] > bb_lo[idx]) and (cl[idx] > cl[idx - 1])
    elif key == "T2_KELTNER_PULLBACK":
        fired = up_1h and ((lo[idx] <= kc_lo[idx]) or (lo[idx - 1] <= kc_lo[idx - 1])) and green and (cl[idx] > kc_lo[idx]) and (cl[idx] > cl[idx - 1])
    elif key == "T3_RSI7_OVERSOLD":
        fired = up_both and (r7[idx - 1] <= 32.0) and green and (cl[idx] > cl[idx - 1]) and (cl[idx] > e50[idx] * 0.992)
    elif key == "T4_STOCH_CROSS_LOW":
        fired = up_4h and (st_k[idx] > st_d[idx]) and (st_k[idx - 1] <= st_d[idx - 1]) and (st_k[idx] <= 38.0) and green
    elif key == "T5_CCI_REBOUND":
        fired = up_1h and (cci20[idx - 1] <= -100.0) and (cci20[idx] > -100.0) and green and (cl[idx] > e50[idx] * 0.992)
    elif key == "T6_MFI_DIP_BUY":
        fired = up_1h and (mfi14[idx - 1] <= 30.0) and (mfi14[idx] > mfi14[idx - 1]) and green and (cl[idx] > cl[idx - 1])
    elif key == "T7_VWAP_RECLAIM":
        fired = up_1h and ((lo[idx] < vwap[idx]) or (cl[idx - 1] < vwap[idx])) and (cl[idx] > vwap[idx]) and (e10[idx] > e20[idx]) and green and (rvol[idx] >= 1.1)
    elif key == "T8_EMA20_50_BOUNCE":
        fired = up_both and ((lo[idx] <= e20[idx] * 1.002) or (lo[idx] <= e50[idx] * 1.002)) and green and (cl[idx] > e20[idx]) and (body_ratio >= 0.40) and (rvol[idx] >= 0.9)
    elif key == "T9_LIQ_SWEEP_HAMMER":
        fired = up_4h and (lo[idx] < lo16[idx]) and (cl[idx] > lo16[idx]) and (l_wick >= 0.35) and green and (rvol[idx] >= 1.1)
    elif key == "T10_2RED_ENGULF":
        fired = up_both and red_p1 and red_p2 and green and (cl[idx] > op[idx - 1]) and (cl[idx] > e20[idx]) and (rvol[idx] >= 0.9)
    elif key == "T11_HL_PIVOT_BREAK":
        fired = up_both and red_p1 and green and (lo[idx] > lo12[idx]) and (cl[idx] > hi[idx - 1]) and (r7[idx] < 62.0) and (rvol[idx] >= 1.0)
    elif key == "T12_COIL_BREAKOUT":
        fired = up_both and ((hi[idx - 1] - lo[idx - 1]) < 0.75 * a14[idx]) and green and (cl[idx] > hi[idx - 1]) and (cl[idx] > e10[idx]) and (rvol[idx] >= 1.2) and (r14[idx] < 68.0)
    elif key == "T13_MACD_HIST_TURN":
        fired = up_4h and (mh[idx - 1] < 0.0) and (mh[idx] > mh[idx - 1]) and (mh[idx - 1] <= mh[idx - 2]) and (cl[idx] > e50[idx]) and green and (rvol[idx] >= 1.0)
    elif key == "T14_DONCHIAN_BO":
        fired = up_both and (cl[idx] > hi20[idx]) and (cl[idx - 1] <= hi20[idx]) and green and (r14[idx] < 70.0) and (rvol[idx] >= 1.3)
    elif key == "N1_EMA20_RECLAIM_L":
        fired = (
            up_both
            and (cl[idx - 1] < e20[idx - 1]) and (cl[idx - 1] > e50[idx - 1] * 0.985)
            and (cl[idx] > e20[idx]) and (cl[idx] > hi[idx - 1])
            and green and (body_ratio >= 0.45) and (rvol[idx] >= 1.15) and (r14[idx] < 65.0)
        )
    elif key == "N2_BB_SQZ_REJECT_L":
        fired = (
            up_4h
            and ((cl[idx - 1] <= bb_lo[idx - 1]) or (lo[idx - 1] < bb_lo[idx - 1]))
            and (bbw15[idx - 1] < 4.5)
            and (cl[idx] > bb_lo[idx]) and (cl[idx] > mid_p1)
            and green and (body_ratio >= 0.35) and (rvol[idx] >= 1.05) and (r14[idx] < 58.0)
        )
    else:
        raise ValueError(f"Unknown VIKA_LONG_V1 key: {key}")

    utc_hour = int((int(ts[idx]) // 3_600_000) % 24)
    raw_sig = "LONG" if fired else None
    final_sig = raw_sig
    reasons = []
    ls_enabled = bool(fcfg.get("retail_ls_enabled", False))
    ls_max_z = float(fcfg.get("retail_ls_max_abs_z", 1.0))
    ls_avail = retail_ls_z is not None
    ls_z_val = float(retail_ls_z) if ls_avail else 0.0
    ls_blocked = False
    d1_lock_on = bool(fcfg.get("btc_1d_ema20_lock", False))
    d1_ok = True if btc_1d_up is None else bool(btc_1d_up)

    if raw_sig == "LONG":
        if btc_switch_on and (not btc_regime.is_4h_up):
            reasons.append("BTC_4H_DOWN")
        if d1_lock_on and (not d1_ok):
            reasons.append("BTC_1D_BELOW_EMA20")
        if ls_enabled and ls_avail and ls_z_val > ls_max_z:
            ls_blocked = True
            reasons.append(f"RETAIL_LS_z={ls_z_val:+.2f}")
        if utc_hour in skip_hours:
            reasons.append(f"UTC_{utc_hour:02d}_CHOP")
        if max_dist4h > 0 and dist4h_pct >= max_dist4h:
            reasons.append(f"4H_STRETCH_{dist4h_pct:+.1f}%")
        if reasons:
            final_sig = None

    return LongCoinEvaluation(
        symbol=symbol,
        key=key,
        bar_ts=int(ts[idx]),
        close_price=float(cl[idx]),
        high_price=float(hi[idx]),
        rvol=round(float(rvol[idx]), 2),
        atr15_pct=round(float(atr_pct[idx]), 3),
        h1_up=h1_up,
        h4_up=h4_up,
        dist4h_ema50_pct=round(dist4h_pct, 2),
        btc_4h_up=btc_regime.is_4h_up,
        utc_hour=utc_hour,
        funding_rate=float(funding_rate),
        funding_bps=round(float(funding_rate) * 10000.0, 2),
        retail_ls_z=round(ls_z_val, 3),
        retail_ls_available=ls_avail,
        retail_ls_blocked=ls_blocked,
        btc_1d_up=d1_ok,
        raw_signal=raw_sig,
        final_signal=final_sig,
        block_reason=", ".join(reasons) if reasons else None,
    )
