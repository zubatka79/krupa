"""
12-School Technical Analysis Engine + Strict 1H/4H/BTC Multi-Timeframe Gate
+ Full Multi-Timeframe Candle & Market Structure Context Builder for the AI Judge.

Strictly evaluates CLOSED 15m, 1H, and 4H candles (zero mid-candle repainting).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ai_trader_v2.chart_renderer import render_15m_chart_png_b64
from bot.bybit_client import BybitClient
from bot.indicators import compute_atr, ema

logger = logging.getLogger(__name__)

ALL_12_SCHOOLS: Tuple[str, ...] = (
    "S01_Supertrend_10_3",
    "S02_Supertrend_7_2",
    "S03_Donchian_Turtle",
    "S04_Donchian_ADX",
    "S05_RS_BTC_Breakout",
    "S06_TTM_Squeeze",
    "S07_Ichimoku_Kumo",
    "S08_EMA_20_50_Cross",
    "S09_Trend_Pullback_RSI",
    "S10_Mean_Rev_BB_RSI",
    "S11_VWAP_Reclaim",
    "S12_Deriv_OI_Funding",
)

# Historical priority weights (S12 Derivatives proved strongest standalone on 15m 10x)
SCHOOL_PRIORITY: Dict[str, float] = {
    "S12_Deriv_OI_Funding": 30.0,
    "S07_Ichimoku_Kumo": 16.0,
    "S06_TTM_Squeeze": 15.0,
    "S05_RS_BTC_Breakout": 15.0,
    "S04_Donchian_ADX": 13.0,
    "S09_Trend_Pullback_RSI": 13.0,
    "S11_VWAP_Reclaim": 13.0,
    "S03_Donchian_Turtle": 12.0,
    "S08_EMA_20_50_Cross": 11.0,
    "S01_Supertrend_10_3": 11.0,
    "S02_Supertrend_7_2": 10.0,
    "S10_Mean_Rev_BB_RSI": 10.0,
}


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    ag = d.clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    al = (-d).clip(lower=0.0).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return (100.0 - 100.0 / (1.0 + ag / al.replace(0.0, np.nan))).fillna(50.0)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    pdm = np.where((up > dn) & (up > 0.0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0.0), dn, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().replace(0.0, np.nan)
    pdi = 100.0 * pd.Series(pdm, index=high.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr
    mdi = 100.0 * pd.Series(mdm, index=high.index).ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr
    dx = (100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)).fillna(0.0)
    return dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().fillna(0.0)


def compute_supertrend(
    hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, atr: np.ndarray, mult: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(cl)
    hl2 = 0.5 * (hi + lo)
    ub = hl2 + mult * atr
    lb = hl2 - mult * atr
    st = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.int8)
    for i in range(1, n):
        if not (ub[i] < ub[i - 1] or cl[i - 1] > ub[i - 1]):
            ub[i] = ub[i - 1]
        if not (lb[i] > lb[i - 1] or cl[i - 1] < lb[i - 1]):
            lb[i] = lb[i - 1]
        if direction[i - 1] == 1:
            if cl[i] < lb[i]:
                direction[i] = -1
                st[i] = ub[i]
            else:
                direction[i] = 1
                st[i] = lb[i]
        else:
            if cl[i] > ub[i]:
                direction[i] = 1
                st[i] = lb[i]
            else:
                direction[i] = -1
                st[i] = ub[i]
    return st, direction


def _candle_tag(o: float, h: float, l: float, c: float, prev_o: float, prev_c: float) -> str:
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    body_r = body / rng
    up_w = (h - max(o, c)) / rng
    dn_w = (min(o, c) - l) / rng
    bull = c >= o
    if body_r <= 0.15:
        return "doji"
    if bull and prev_c < prev_o and c >= prev_o and o <= prev_c and body_r >= 0.50:
        return "bull_engulf"
    if (not bull) and prev_c > prev_o and c <= prev_o and o >= prev_c and body_r >= 0.50:
        return "bear_engulf"
    if dn_w >= 0.55 and body_r <= 0.35:
        return "hammer_pin"
    if up_w >= 0.55 and body_r <= 0.35:
        return "shooting_pin"
    if body_r >= 0.70:
        return "strong_bull" if bull else "strong_bear"
    return "bull" if bull else "bear"


def _fmt_candles(df: pd.DataFrame, count: int, vol_sma: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
    sub = df.tail(count).reset_index(drop=True)
    if vol_sma is not None:
        vs = vol_sma.tail(count).reset_index(drop=True)
    else:
        vs = None
    out: List[Dict[str, Any]] = []
    for idx in range(len(sub)):
        r = sub.iloc[idx]
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        prev_o = float(sub.iloc[idx - 1]["open"]) if idx > 0 else o
        prev_c = float(sub.iloc[idx - 1]["close"]) if idx > 0 else c
        ts_str = datetime.fromtimestamp(int(r["timestamp"]) / 1000.0, tz=timezone.utc).strftime("%m-%d %H:%M")
        chg_pct = ((c - o) / o * 100.0) if o > 0 else 0.0
        item: Dict[str, Any] = {
            "t": ts_str,
            "o": round(o, 6),
            "h": round(h, 6),
            "l": round(l, 6),
            "c": round(c, 6),
            "chg%": round(chg_pct, 2),
            "tag": _candle_tag(o, h, l, c, prev_o, prev_c),
        }
        if vs is not None and float(vs.iloc[idx] or 0) > 0:
            item["rvol"] = round(float(r["volume"]) / float(vs.iloc[idx]), 2)
        out.append(item)
    return out


def _find_swing_levels(df: pd.DataFrame, lookback: int = 48, price: float = 0.0) -> Dict[str, Any]:
    """Identify nearest structural support & resistance from recent pivot highs/lows."""
    sub = df.tail(lookback).reset_index(drop=True)
    hi = sub["high"].to_numpy(float)
    lo = sub["low"].to_numpy(float)
    piv_hi: List[float] = []
    piv_lo: List[float] = []
    for i in range(2, len(sub) - 2):
        if hi[i] >= hi[i - 1] and hi[i] >= hi[i - 2] and hi[i] >= hi[i + 1] and hi[i] >= hi[i + 2]:
            piv_hi.append(float(hi[i]))
        if lo[i] <= lo[i - 1] and lo[i] <= lo[i - 2] and lo[i] <= lo[i + 1] and lo[i] <= lo[i + 2]:
            piv_lo.append(float(lo[i]))
    # Include rolling highs/lows as fallback
    if len(hi) >= 5:
        piv_hi.append(float(np.max(hi[-24:])))
        piv_lo.append(float(np.min(lo[-24:])))

    res_above = sorted([x for x in piv_hi if x > price * 1.001])
    sup_below = sorted([x for x in piv_lo if x < price * 0.999], reverse=True)
    nearest_res = res_above[0] if res_above else None
    nearest_sup = sup_below[0] if sup_below else None
    return {
        "nearest_resistance": round(nearest_res, 6) if nearest_res else None,
        "dist_to_res_pct": round((nearest_res - price) / price * 100.0, 2) if (nearest_res and price > 0) else None,
        "nearest_support": round(nearest_sup, 6) if nearest_sup else None,
        "dist_to_sup_pct": round((price - nearest_sup) / price * 100.0, 2) if (nearest_sup and price > 0) else None,
        "top_resistances": [round(x, 6) for x in res_above[:3]],
        "top_supports": [round(x, 6) for x in sup_below[:3]],
    }


def fetch_orderbook_metrics(client: BybitClient, symbol: str) -> Dict[str, float]:
    """Fetch live orderbook spread (bps), ±0.3% depth in USDT, and bid/ask imbalance."""
    try:
        res = client._public_get(
            "/v5/market/orderbook",
            params={"category": "linear", "symbol": symbol, "limit": 50},
            retries=2,
        )
        bids = [(float(p), float(q)) for p, q in res.get("b", [])]
        asks = [(float(p), float(q)) for p, q in res.get("a", [])]
        if not bids or not asks:
            return {"spread_bps": 999.0, "depth_03_usdt": 0.0, "bid_ask_ratio": 1.0, "best_bid": 0.0, "best_ask": 0.0}
        bb, ba = bids[0][0], asks[0][0]
        mid = 0.5 * (bb + ba)
        spread_bps = ((ba - bb) / mid) * 10000.0 if mid > 0 else 999.0
        bid_03 = sum(p * q for p, q in bids if p >= mid * 0.997)
        ask_03 = sum(p * q for p, q in asks if p <= mid * 1.003)
        depth_03 = min(bid_03, ask_03)
        ratio = bid_03 / max(ask_03, 1.0)
        return {
            "spread_bps": round(spread_bps, 2),
            "depth_03_usdt": round(depth_03, 0),
            "bid_ask_ratio": round(ratio, 2),
            "best_bid": bb,
            "best_ask": ba,
        }
    except Exception as exc:
        logger.debug("Orderbook fetch failed for %s: %s", symbol, exc)
        return {"spread_bps": 999.0, "depth_03_usdt": 0.0, "bid_ask_ratio": 1.0, "best_bid": 0.0, "best_ask": 0.0}


@dataclass
class CandidateSetup:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    bar_ts: int
    close_price: float
    atr_14: float
    atr_pct: float
    rvol: float
    adx_15m: float
    rsi_15m: float
    fired_schools: List[str]
    primary_school: str
    priority_score: float
    regime_label: str
    sl_candidates_pct: Dict[str, float]
    default_sl_pct: float
    recommended_tp1_r: float
    recommended_tp2_r: float
    orderbook: Dict[str, float]
    context_payload: Dict[str, Any] = field(default_factory=dict)
    chart_png_b64: Optional[str] = None


@dataclass
class BtcMarketContext:
    allow_alt_long: bool
    allow_alt_short: bool
    h1_trend: str
    h4_trend: str
    bar_move_15m_pct: float
    shock_active: bool
    ret_24bars_pct: float
    summary_dict: Dict[str, Any]


def build_btc_context(client: BybitClient, shock_limit_pct: float = 0.9) -> Optional[BtcMarketContext]:
    """Fetch closed BTCUSDT 15m/1H/4H candles and compute macro directional gates & context."""
    df15 = client.get_klines("BTCUSDT", interval="15", days=4, use_cache=True, closed_only=True)
    df1h = client.get_klines("BTCUSDT", interval="60", days=10, use_cache=True, closed_only=True)
    df4h = client.get_klines("BTCUSDT", interval="240", days=35, use_cache=True, closed_only=True)
    if len(df15) < 40 or len(df1h) < 60 or len(df4h) < 60:
        logger.warning("Insufficient BTC klines (15m=%d, 1h=%d, 4h=%d)", len(df15), len(df1h), len(df4h))
        return None

    e20_1h = float(ema(df1h["close"], 20).iloc[-1])
    e50_1h = float(ema(df1h["close"], 50).iloc[-1])
    c1h = float(df1h["close"].iloc[-1])
    h1_bull = (c1h > e50_1h) and (e20_1h > e50_1h)
    h1_bear = (c1h < e50_1h) and (e20_1h < e50_1h)
    h1_trend = "BULL" if h1_bull else ("BEAR" if h1_bear else "NEUTRAL")

    e20_4h = float(ema(df4h["close"], 20).iloc[-1])
    e50_4h = float(ema(df4h["close"], 50).iloc[-1])
    c4h = float(df4h["close"].iloc[-1])
    h4_bull = (c4h > e50_4h) and (e20_4h > e50_4h)
    h4_bear = (c4h < e50_4h) and (e20_4h < e50_4h)
    h4_trend = "BULL" if h4_bull else ("BEAR" if h4_bear else "NEUTRAL")

    # Soft BTC gate: forbid alt LONG when BTC 1H+4H both BEAR; forbid alt SHORT when both BULL
    allow_alt_long = not (h1_bear and h4_bear)
    allow_alt_short = not (h1_bull and h4_bull)

    last_o = float(df15["open"].iloc[-1])
    last_c = float(df15["close"].iloc[-1])
    bar_move_pct = abs(last_c - last_o) / max(last_o, 1e-12) * 100.0
    shock_active = bar_move_pct >= shock_limit_pct
    ret_24 = (last_c / max(float(df15["close"].iloc[-25]), 1e-12) - 1.0) * 100.0

    summary = {
        "btc_price": round(last_c, 2),
        "btc_1h_trend": h1_trend,
        "btc_4h_trend": h4_trend,
        "btc_15m_move_pct": round(((last_c - last_o) / max(last_o, 1e-12)) * 100.0, 2),
        "btc_6h_ret_pct": round(ret_24, 2),
        "btc_shock": shock_active,
        "btc_recent_15m": _fmt_candles(df15, 8),
        "btc_recent_4h": _fmt_candles(df4h, 6),
    }
    return BtcMarketContext(
        allow_alt_long=allow_alt_long,
        allow_alt_short=allow_alt_short,
        h1_trend=h1_trend,
        h4_trend=h4_trend,
        bar_move_15m_pct=bar_move_pct,
        shock_active=shock_active,
        ret_24bars_pct=ret_24,
        summary_dict=summary,
    )


def evaluate_symbol_12_schools(
    client: BybitClient,
    symbol: str,
    btc_ctx: BtcMarketContext,
    config: Dict[str, Any],
) -> Optional[CandidateSetup]:
    """
    Evaluate ALL 12 TA schools on the latest strictly CLOSED 15m candle for `symbol`,
    gated by strictly closed 1H + 4H trend and BTC 1H/4H gate.
    Returns a rich CandidateSetup (with full 15m/1H/4H candle picture & SL/TP anchors) or None.
    """
    df15 = client.get_klines(symbol, interval="15", days=4, use_cache=True, closed_only=True)
    df1h = client.get_klines(symbol, interval="60", days=10, use_cache=True, closed_only=True)
    df4h = client.get_klines(symbol, interval="240", days=35, use_cache=True, closed_only=True)
    if len(df15) < 220 or len(df1h) < 60 or len(df4h) < 60:
        return None

    # 1. Strict 1H + 4H closed-candle directional gate
    e20_1h = float(ema(df1h["close"], 20).iloc[-1])
    e50_1h = float(ema(df1h["close"], 50).iloc[-1])
    c1h = float(df1h["close"].iloc[-1])
    adx_1h = float(compute_adx(df1h["high"], df1h["low"], df1h["close"], 14).iloc[-1])
    h1_long = (c1h > e50_1h) and (e20_1h > e50_1h)
    h1_short = (c1h < e50_1h) and (e20_1h < e50_1h)

    e20_4h = float(ema(df4h["close"], 20).iloc[-1])
    e50_4h = float(ema(df4h["close"], 50).iloc[-1])
    c4h = float(df4h["close"].iloc[-1])
    adx_4h = float(compute_adx(df4h["high"], df4h["low"], df4h["close"], 14).iloc[-1])
    h4_long = (c4h > e50_4h) and (e20_4h > e50_4h)
    h4_short = (c4h < e50_4h) and (e20_4h < e50_4h)

    btc_filter = bool(config.get("btc_filter_enabled", True))
    long_req_4h = bool(config.get("long_require_4h", True))
    short_req_4h = bool(config.get("short_require_4h", False))
    allow_long = h1_long and (h4_long if long_req_4h else True) and (btc_ctx.allow_alt_long if btc_filter else True)
    allow_short = h1_short and (h4_short if short_req_4h else True) and (btc_ctx.allow_alt_short if btc_filter else True)
    if not allow_long and not allow_short:
        return None

    # 2. Compute 15m indicators for all 12 schools
    op = df15["open"].to_numpy(float)
    hi = df15["high"].to_numpy(float)
    lo = df15["low"].to_numpy(float)
    cl = df15["close"].to_numpy(float)
    vol = df15["volume"].astype(float)
    bar_ts = int(df15["timestamp"].iloc[-1])
    close_now = float(cl[-1])
    if close_now <= 0:
        return None

    atr14_s = compute_atr(df15["high"], df15["low"], df15["close"], 14).fillna(0.0)
    atr14 = atr14_s.to_numpy(float)
    atr20 = compute_atr(df15["high"], df15["low"], df15["close"], 20).fillna(0.0).to_numpy(float)
    atr10 = compute_atr(df15["high"], df15["low"], df15["close"], 10).fillna(0.0).to_numpy(float)
    atr7 = compute_atr(df15["high"], df15["low"], df15["close"], 7).fillna(0.0).to_numpy(float)

    e20 = ema(df15["close"], 20).to_numpy(float)
    e50 = ema(df15["close"], 50).to_numpy(float)
    e100 = ema(df15["close"], 100).to_numpy(float)
    e200 = ema(df15["close"], 200).to_numpy(float)
    rsi14 = compute_rsi(df15["close"], 14).to_numpy(float)
    adx14 = compute_adx(df15["high"], df15["low"], df15["close"], 14).to_numpy(float)

    vol_sma20 = vol.rolling(20, min_periods=10).mean()
    rvol = float(vol.iloc[-1] / vol_sma20.iloc[-1]) if float(vol_sma20.iloc[-1] or 0) > 0 else 1.0

    hi20 = float(df15["high"].iloc[-21:-1].max())
    lo20 = float(df15["low"].iloc[-21:-1].min())
    hi16 = float(df15["high"].iloc[-17:-1].max())
    lo16 = float(df15["low"].iloc[-17:-1].min())
    hi10 = float(df15["high"].iloc[-11:-1].max())
    lo10 = float(df15["low"].iloc[-11:-1].min())
    hi8 = float(df15["high"].iloc[-9:-1].max())
    lo8 = float(df15["low"].iloc[-9:-1].min())

    sma20_s = df15["close"].rolling(20, min_periods=20).mean().fillna(0.0)
    sd20_s = df15["close"].rolling(20, min_periods=20).std().fillna(0.0)
    sma20 = sma20_s.to_numpy(float)
    sd20 = sd20_s.to_numpy(float)
    bb_u20, bb_l20 = sma20 + 2.0 * sd20, sma20 - 2.0 * sd20
    bb_u22, bb_l22 = sma20 + 2.2 * sd20, sma20 - 2.2 * sd20
    kc_u, kc_l = e20 + 1.5 * atr20, e20 - 1.5 * atr20
    sqz = (bb_u20 < kc_u) & (bb_l20 > kc_l)
    bb_w = np.where(cl > 0, 4.0 * sd20 / np.maximum(cl, 1e-12) * 100.0, 0.0)
    bb_w_prev = float(bb_w[-2])
    bb_w_q50 = float(pd.Series(bb_w[:-1]).tail(100).quantile(0.50))

    tenkan = (0.5 * (df15["high"].rolling(9).max() + df15["low"].rolling(9).min())).fillna(0.0).to_numpy(float)
    kijun = (0.5 * (df15["high"].rolling(26).max() + df15["low"].rolling(26).min())).fillna(0.0).to_numpy(float)
    sen_a = (0.5 * (pd.Series(tenkan) + pd.Series(kijun))).shift(26).fillna(0.0).to_numpy(float)
    sen_b = (0.5 * (df15["high"].rolling(52).max() + df15["low"].rolling(52).min())).shift(26).fillna(0.0).to_numpy(float)
    c_top = np.maximum(sen_a, sen_b)
    c_bot = np.minimum(sen_a, sen_b)

    tp = (df15["high"] + df15["low"] + df15["close"]) / 3.0
    vwap_s = (tp * vol).rolling(96, min_periods=48).sum() / vol.rolling(96, min_periods=48).sum().replace(0.0, np.nan)
    vwap = vwap_s.bfill().fillna(df15["close"]).to_numpy(float)
    vwap_sd = (df15["close"] - vwap).rolling(96, min_periods=48).std().fillna(0.0).to_numpy(float)
    v_lo = vwap - 2.0 * vwap_sd
    v_hi = vwap + 2.0 * vwap_sd

    st1_line, st1_dir = compute_supertrend(hi, lo, cl, atr10, 3.0)
    st2_line, st2_dir = compute_supertrend(hi, lo, cl, atr7, 2.0)

    # Relative strength vs BTC over last 24 bars (6 hours)
    coin_r24 = (close_now / max(float(cl[-25]), 1e-12) - 1.0) * 100.0
    rs24 = coin_r24 - btc_ctx.ret_24bars_pct

    # Live OI 24h change & Funding Rate quantiles (for S12 and AI context)
    oi_chg_24h = 0.0
    df_oi = client.get_open_interest(symbol, interval_time="1h", limit=30)
    if df_oi is not None and len(df_oi) >= 25:
        oi_now = float(df_oi["openInterest"].iloc[-1])
        oi_prev = float(df_oi["openInterest"].iloc[-25])
        if oi_prev > 0:
            oi_chg_24h = (oi_now - oi_prev) / oi_prev * 100.0

    fr_now = 0.0
    fr_q15 = 0.0
    fr_q85 = 0.0
    df_fr = client.get_funding_history(symbol, limit=90)
    if df_fr is not None and len(df_fr) >= 10:
        fr_now = float(df_fr["fundingRate"].iloc[-1])
        fr_q15 = float(df_fr["fundingRate"].quantile(0.15))
        fr_q85 = float(df_fr["fundingRate"].quantile(0.85))

    # 3. Evaluate ALL 12 Schools on the latest closed bar (-1)
    enabled = set(config.get("schools_enabled", list(ALL_12_SCHOOLS)))
    fired_long: List[str] = []
    fired_short: List[str] = []

    if allow_long:
        if "S01_Supertrend_10_3" in enabled and st1_dir[-1] == 1 and st1_dir[-2] == -1 and cl[-1] > e200[-1]:
            fired_long.append("S01_Supertrend_10_3")
        if "S02_Supertrend_7_2" in enabled and st2_dir[-1] == 1 and st2_dir[-2] == -1 and cl[-1] > e100[-1]:
            fired_long.append("S02_Supertrend_7_2")
        if "S03_Donchian_Turtle" in enabled and hi20 > 0 and cl[-1] > hi20 and cl[-1] > e100[-1]:
            fired_long.append("S03_Donchian_Turtle")
        if "S04_Donchian_ADX" in enabled and hi20 > 0 and cl[-1] > hi20 and adx14[-1] > 22.0:
            fired_long.append("S04_Donchian_ADX")
        if "S05_RS_BTC_Breakout" in enabled and bb_w_q50 > 0 and bb_w_prev < bb_w_q50 and rs24 > 0.8 and cl[-1] > hi16 and cl[-1] > e100[-1]:
            fired_long.append("S05_RS_BTC_Breakout")
        if "S06_TTM_Squeeze" in enabled and bool(sqz[-2]) and (not bool(sqz[-1])) and cl[-1] > e20[-1] and cl[-1] > e50[-1]:
            fired_long.append("S06_TTM_Squeeze")
        if "S07_Ichimoku_Kumo" in enabled and c_top[-1] > 0 and cl[-1] > c_top[-1] and tenkan[-1] > kijun[-1] and sen_a[-1] > sen_b[-1] and (cl[-2] <= c_top[-2] or tenkan[-2] <= kijun[-2]):
            fired_long.append("S07_Ichimoku_Kumo")
        if "S08_EMA_20_50_Cross" in enabled and e20[-1] > e50[-1] and e20[-2] <= e50[-2] and cl[-1] > e200[-1]:
            fired_long.append("S08_EMA_20_50_Cross")
        if "S09_Trend_Pullback_RSI" in enabled and e50[-1] > e200[-1] and rsi14[-2] < 40.0 and rsi14[-1] > rsi14[-2]:
            fired_long.append("S09_Trend_Pullback_RSI")
        if "S10_Mean_Rev_BB_RSI" in enabled and lo[-1] <= bb_l22[-1] and rsi14[-1] < 30.0:
            fired_long.append("S10_Mean_Rev_BB_RSI")
        if "S11_VWAP_Reclaim" in enabled and vwap_sd[-1] > 0 and (lo[-1] < v_lo[-1] or cl[-2] < v_lo[-2]) and cl[-1] > v_lo[-1] and cl[-1] < vwap[-1]:
            fired_long.append("S11_VWAP_Reclaim")
        if "S12_Deriv_OI_Funding" in enabled and oi_chg_24h >= 3.0 and (fr_now <= -0.0001 or (fr_now <= fr_q15 and fr_now < 0.00005)) and hi8 > 0 and cl[-1] > hi8:
            fired_long.append("S12_Deriv_OI_Funding")

    if allow_short:
        if "S01_Supertrend_10_3" in enabled and st1_dir[-1] == -1 and st1_dir[-2] == 1 and cl[-1] < e200[-1]:
            fired_short.append("S01_Supertrend_10_3")
        if "S02_Supertrend_7_2" in enabled and st2_dir[-1] == -1 and st2_dir[-2] == 1 and cl[-1] < e100[-1]:
            fired_short.append("S02_Supertrend_7_2")
        if "S03_Donchian_Turtle" in enabled and lo20 > 0 and cl[-1] < lo20 and cl[-1] < e100[-1]:
            fired_short.append("S03_Donchian_Turtle")
        if "S04_Donchian_ADX" in enabled and lo20 > 0 and cl[-1] < lo20 and adx14[-1] > 22.0:
            fired_short.append("S04_Donchian_ADX")
        if "S05_RS_BTC_Breakout" in enabled and bb_w_q50 > 0 and bb_w_prev < bb_w_q50 and rs24 < -0.8 and cl[-1] < lo16 and cl[-1] < e100[-1]:
            fired_short.append("S05_RS_BTC_Breakout")
        if "S06_TTM_Squeeze" in enabled and bool(sqz[-2]) and (not bool(sqz[-1])) and cl[-1] < e20[-1] and cl[-1] < e50[-1]:
            fired_short.append("S06_TTM_Squeeze")
        if "S07_Ichimoku_Kumo" in enabled and c_bot[-1] > 0 and cl[-1] < c_bot[-1] and tenkan[-1] < kijun[-1] and sen_a[-1] < sen_b[-1] and (cl[-2] >= c_bot[-2] or tenkan[-2] >= kijun[-2]):
            fired_short.append("S07_Ichimoku_Kumo")
        if "S08_EMA_20_50_Cross" in enabled and e20[-1] < e50[-1] and e20[-2] >= e50[-2] and cl[-1] < e200[-1]:
            fired_short.append("S08_EMA_20_50_Cross")
        if "S09_Trend_Pullback_RSI" in enabled and e50[-1] < e200[-1] and rsi14[-2] > 60.0 and rsi14[-1] < rsi14[-2]:
            fired_short.append("S09_Trend_Pullback_RSI")
        if "S10_Mean_Rev_BB_RSI" in enabled and hi[-1] >= bb_u22[-1] and rsi14[-1] > 70.0:
            fired_short.append("S10_Mean_Rev_BB_RSI")
        if "S11_VWAP_Reclaim" in enabled and vwap_sd[-1] > 0 and (hi[-1] > v_hi[-1] or cl[-2] > v_hi[-2]) and cl[-1] < v_hi[-1] and cl[-1] > vwap[-1]:
            fired_short.append("S11_VWAP_Reclaim")
        if "S12_Deriv_OI_Funding" in enabled and oi_chg_24h >= 3.0 and fr_now >= fr_q85 and fr_now >= 0.0001 and lo8 > 0 and cl[-1] < lo8:
            fired_short.append("S12_Deriv_OI_Funding")

    if not fired_long and not fired_short:
        return None

    direction = "LONG" if len(fired_long) >= len(fired_short) else "SHORT"
    fired = fired_long if direction == "LONG" else fired_short

    # Variant 2 consensus pre-filter + mandatory volume check for S12 too:
    # - S12_Deriv_OI_Funding requires rvol >= min_rvol_s12 (default 1.2)
    # - Non-S12 requires >= min_schools_non_s12 (2+) schools AND rvol >= min_rvol_non_s12 (1.2)
    min_rvol_s12 = float(config.get("min_rvol_s12", 1.2))
    min_schools_non_s12 = int(config.get("min_schools_non_s12", 2))
    min_rvol_non_s12 = float(config.get("min_rvol_non_s12", 1.2))
    if "S12_Deriv_OI_Funding" in fired:
        if rvol < min_rvol_s12:
            return None
    else:
        if len(fired) < min_schools_non_s12 or rvol < min_rvol_non_s12:
            return None

    # 4. Mandatory SL corridor check (trading-core.md: [1.2%, 1.6%] and >= 1.1 * ATR(14))
    atr_now = float(atr14[-1])
    atr_pct = (atr_now / close_now) * 100.0
    sl_min = float(config.get("sl_min_pct", 1.2))
    sl_max = float(config.get("sl_max_pct", 1.6))
    atr_floor_mult = float(config.get("sl_atr_floor_mult", 1.1))
    atr_floor_pct = atr_floor_mult * atr_pct
    if atr_floor_pct > sl_max:
        # Coin is too volatile right now for safe 10x [1.2%-1.6%] SL -> skip!
        return None

    if direction == "LONG":
        raw_swing_sl = ((close_now - (lo10 - 0.15 * atr_now)) / close_now) * 100.0
        raw_kijun_sl = ((close_now - kijun[-1]) / close_now) * 100.0 if kijun[-1] > 0 else raw_swing_sl
        raw_st_sl = ((close_now - st2_line[-1]) / close_now) * 100.0 if st2_line[-1] > 0 else raw_swing_sl
    else:
        raw_swing_sl = (((hi10 + 0.15 * atr_now) - close_now) / close_now) * 100.0
        raw_kijun_sl = ((kijun[-1] - close_now) / close_now) * 100.0 if kijun[-1] > 0 else raw_swing_sl
        raw_st_sl = ((st2_line[-1] - close_now) / close_now) * 100.0 if st2_line[-1] > 0 else raw_swing_sl

    def _clamp_sl(val: float) -> float:
        return round(min(sl_max, max(sl_min, atr_floor_pct, val)), 3)

    sl_candidates = {
        "swing_10_sl_pct": _clamp_sl(raw_swing_sl),
        "raw_swing_10_sl_pct": round(raw_swing_sl, 3),
        "atr_floor_sl_pct": _clamp_sl(atr_floor_pct),
        "kijun_sl_pct": _clamp_sl(raw_kijun_sl) if 0.3 < raw_kijun_sl < 3.0 else _clamp_sl(atr_floor_pct),
        "supertrend_sl_pct": _clamp_sl(raw_st_sl) if 0.3 < raw_st_sl < 3.0 else _clamp_sl(atr_floor_pct),
    }
    default_sl_pct = sl_candidates["swing_10_sl_pct"]

    # 5. Live orderbook check
    ob = fetch_orderbook_metrics(client, symbol)
    if ob["depth_03_usdt"] < float(config.get("min_orderbook_depth_03_usdt", 50000.0)):
        return None
    if ob["spread_bps"] > float(config.get("max_spread_bps", 10.0)):
        return None

    # 6. Classify higher-TF market regime & compute priority
    if adx_4h >= 25.0 and adx_1h >= 22.0:
        regime = "STRONG_TREND"
    elif bool(sqz[-2]) or (bb_w_q50 > 0 and bb_w_prev < bb_w_q50 * 0.85):
        regime = "SQUEEZE_BREAKOUT"
    elif adx_1h >= 18.0:
        regime = "TREND"
    else:
        regime = "RANGE_OR_CHOP"

    fired_sorted = sorted(fired, key=lambda s: SCHOOL_PRIORITY.get(s, 10.0), reverse=True)
    primary_school = fired_sorted[0]
    priority = sum(SCHOOL_PRIORITY.get(s, 10.0) for s in fired_sorted)
    if len(fired_sorted) >= 2:
        priority += 12.0 * (len(fired_sorted) - 1)  # Confluence bonus
    if rvol >= 1.5:
        priority += 8.0
    if regime == "STRONG_TREND":
        priority += 6.0

    # Structural S/R levels on 1H and 4H
    sr_1h = _find_swing_levels(df1h, lookback=60, price=close_now)
    sr_4h = _find_swing_levels(df4h, lookback=60, price=close_now)

    # Clearance to nearest 1H/4H obstacle in R multiples
    if direction == "LONG":
        dists = [d for d in (sr_1h["dist_to_res_pct"], sr_4h["dist_to_res_pct"]) if d is not None]
    else:
        dists = [d for d in (sr_1h["dist_to_sup_pct"], sr_4h["dist_to_sup_pct"]) if d is not None]
    nearest_wall_pct = min(dists) if dists else 9.99
    wall_r = round(nearest_wall_pct / max(default_sl_pct, 0.1), 2)

    rec_tp1_r = 1.8
    rec_tp2_r = 3.5 if regime in ("STRONG_TREND", "SQUEEZE_BREAKOUT") else 3.0

    # 7. Build complete multi-TF candle & TA picture for the AI Judge
    context_payload: Dict[str, Any] = {
        "symbol": symbol,
        "direction": direction,
        "closed_bar_utc": datetime.fromtimestamp(bar_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "price": round(close_now, 6),
        "fired_schools": fired_sorted,
        "primary_school": primary_school,
        "confluence_count": len(fired_sorted),
        "market_regime": regime,
        "higher_tf": {
            "1H": {
                "trend": "BULL" if h1_long else ("BEAR" if h1_short else "NEUTRAL"),
                "close": round(c1h, 6),
                "ema20": round(e20_1h, 6),
                "ema50": round(e50_1h, 6),
                "adx14": round(adx_1h, 1),
            },
            "4H": {
                "trend": "BULL" if h4_long else ("BEAR" if h4_short else "NEUTRAL"),
                "close": round(c4h, 6),
                "ema20": round(e20_4h, 6),
                "ema50": round(e50_4h, 6),
                "adx14": round(adx_4h, 1),
            },
        },
        "indicators_15m": {
            "atr14": round(atr_now, 6),
            "atr_pct": round(atr_pct, 3),
            "rsi14": round(float(rsi14[-1]), 1),
            "adx14": round(float(adx14[-1]), 1),
            "rvol": round(rvol, 2),
            "ema20": round(float(e20[-1]), 6),
            "ema50": round(float(e50[-1]), 6),
            "ema200": round(float(e200[-1]), 6),
            "vwap_24h": round(float(vwap[-1]), 6),
            "kijun": round(float(kijun[-1]), 6),
            "supertrend_7_2": round(float(st2_line[-1]), 6),
            "rs_vs_btc_6h_pct": round(rs24, 2),
            "squeeze_released": bool(sqz[-2]) and (not bool(sqz[-1])),
        },
        "derivatives": {
            "oi_change_24h_pct": round(oi_chg_24h, 2),
            "funding_rate_pct": round(fr_now * 100.0, 4),
            "funding_q15_pct": round(fr_q15 * 100.0, 4),
            "funding_q85_pct": round(fr_q85 * 100.0, 4),
        },
        "orderbook": ob,
        "structure_levels": {
            "1H": sr_1h,
            "4H": sr_4h,
            "nearest_obstacle_pct": round(nearest_wall_pct, 2),
            "nearest_obstacle_in_R": wall_r,
        },
        "sl_tp_options": {
            "allowed_sl_corridor_pct": [sl_min, sl_max],
            "atr_floor_sl_pct": round(atr_floor_pct, 3),
            "structural_sl_candidates_pct": sl_candidates,
            "default_sl_pct": default_sl_pct,
            "recommended_tp1_r": rec_tp1_r,
            "recommended_tp2_r": rec_tp2_r,
            "tp1_close_pct": float(config.get("tp1_close_pct", 55.0)),
            "smart_be_stages": {
                "stage1_close_r": float(config.get("be_stage1_close_r", 1.3)),
                "stage1_sl_r": float(config.get("be_stage1_sl_r", -0.5)),
                "stage2_tp1_sl_r": float(config.get("be_stage2_tp1_sl_r", -0.35)),
                "stage3_close_r": float(config.get("be_stage3_close_r", 2.3)),
                "stage3_sl_r": float(config.get("be_stage3_sl_r", 0.3)),
                "trailing_atr_mult": float(config.get("trailing_atr_mult", 4.0)),
            },
        },
        "candles_15m_last20": _fmt_candles(df15, 20, vol_sma20),
        "candles_1h_last12": _fmt_candles(df1h, 12),
        "candles_4h_last8": _fmt_candles(df4h, 8),
        "btc_context": btc_ctx.summary_dict,
    }

    chart_b64: Optional[str] = None
    if bool(config.get("chart_png_enabled", True)):
        chart_b64 = render_15m_chart_png_b64(
            symbol=symbol,
            direction=direction,
            df15=df15,
            ema20=e20,
            ema50=e50,
            vwap=vwap,
            sr_1h=sr_1h,
            sr_4h=sr_4h,
            sl_pct=default_sl_pct,
            tp1_r=rec_tp1_r,
            candles_count=int(config.get("chart_candles_count", 72)),
        )

    return CandidateSetup(
        symbol=symbol,
        direction=direction,
        bar_ts=bar_ts,
        close_price=close_now,
        atr_14=atr_now,
        atr_pct=atr_pct,
        rvol=rvol,
        adx_15m=float(adx14[-1]),
        rsi_15m=float(rsi14[-1]),
        fired_schools=fired_sorted,
        primary_school=primary_school,
        priority_score=round(priority, 2),
        regime_label=regime,
        sl_candidates_pct=sl_candidates,
        default_sl_pct=default_sl_pct,
        recommended_tp1_r=rec_tp1_r,
        recommended_tp2_r=rec_tp2_r,
        orderbook=ob,
        context_payload=context_payload,
        chart_png_b64=chart_b64,
    )
