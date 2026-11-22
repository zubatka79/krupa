"""Variant 1 Strategy: trend_pullback_v2 (High Win-Rate Trend Pullback with strict risk sizing)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from bot.bybit_client import InstrumentSpec, round_step_down, round_tick
from bot.indicators import (
    compute_adx,
    compute_atr,
    compute_daily_vwap,
    compute_rsi,
    ema,
)


@dataclass
class V2Signal:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    timestamp: int
    close_price: float
    quality: float
    rsi: float
    atr_pct: float
    h1_adx: float
    h4_adx: float


@dataclass
class V2PositionPlan:
    symbol: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    be_trigger_price: Optional[float]
    be_stop_price: Optional[float]
    sl_pct: float
    tp_pct: float
    leverage: int
    margin_usdt: float
    notional_usdt: float
    qty: float
    entry_is_maker: bool
    risk_usdt: float
    risk_pct: float


def prepare_htf_trend_v2(df_htf: pd.DataFrame, interval_ms: int, prefix: str) -> pd.DataFrame:
    """Compute strictly causal EMA(50), EMA(200), EMA(50) slope, and ADX(14) on 1H or 4H."""
    out = pd.DataFrame(index=df_htf.index)
    out["close_ts"] = (df_htf["timestamp"] + interval_ms).astype("int64")

    ema50 = ema(df_htf["close"], 50)
    ema200 = ema(df_htf["close"], 200)
    adx_val, pdi, mdi = compute_adx(df_htf["high"], df_htf["low"], df_htf["close"], 14)

    slope_bars = 3 if prefix == "h1" else 2
    ema50_up = ema50 > ema50.shift(slope_bars)
    ema50_down = ema50 < ema50.shift(slope_bars)

    # Strict trend alignment: price & EMA50 on the right side of EMA200, EMA50 sloping with trend, ADX >= 18
    out[f"{prefix}_trend_long"] = (
        (df_htf["close"] > ema50)
        & (ema50 > ema200)
        & ema50_up
        & (adx_val >= 18.0)
    )
    out[f"{prefix}_trend_short"] = (
        (df_htf["close"] < ema50)
        & (ema50 < ema200)
        & ema50_down
        & (adx_val >= 18.0)
    )
    out[f"{prefix}_adx"] = adx_val.values
    out[f"{prefix}_ema50_dist_pct"] = ((df_htf["close"] - ema50) / ema50.replace(0.0, np.nan) * 100.0).fillna(0.0).values
    return out.sort_values("close_ts").reset_index(drop=True)


def enrich_v2_dataframe(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    entry_interval_ms: int = 15 * 60_000,
    context_interval_ms: int = 60 * 60_000,
    trend_interval_ms: int = 240 * 60_000,
    btc_shock_pause_pct: float = 0.8,
) -> pd.DataFrame:
    """
    Vectorized, strictly causal feature & signal computation for `trend_pullback_v2`.
    Zero lookahead: 1H and 4H features merge strictly on completed bar close timestamps.
    """
    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_interval_ms).astype("int64")

    # 15M Indicators
    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    df["vwap"] = compute_daily_vwap(df)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)

    c_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / c_range).fillna(0.0)
    df["is_bull_candle"] = (df["close"] > df["open"]) & (df["body_ratio"] >= 0.35)
    df["is_bear_candle"] = (df["close"] < df["open"]) & (df["body_ratio"] >= 0.35)

    # Protective anti-pump/dump abnormal volatility filter + 3-bar (45m) cooldown
    bar_range = df["high"] - df["low"]
    range_3 = df["high"].rolling(3, min_periods=3).max() - df["low"].rolling(3, min_periods=3).min()
    roc_3_abs = (df["close"] - df["close"].shift(3)).abs() / df["close"].shift(3).replace(0.0, np.nan)
    atr_frac = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)

    is_abnormal_vol = (
        (bar_range > 2.2 * df["atr"])
        | (range_3 > 3.8 * df["atr"])
        | (roc_3_abs > 2.2 * atr_frac)
    ).fillna(False)
    df["is_abnormal_vol"] = is_abnormal_vol
    # Mandatory 3 closed 15M bars (45 minutes) pause after any abnormal volatility spike
    df["vol_cooldown"] = (
        is_abnormal_vol
        | is_abnormal_vol.shift(1, fill_value=False)
        | is_abnormal_vol.shift(2, fill_value=False)
        | is_abnormal_vol.shift(3, fill_value=False)
    )

    # Local 32-bar (8h) swing wave for 0.382-0.618 Fibonacci retracement zone
    wave_high = df["high"].shift(1).rolling(32, min_periods=16).max()
    wave_low = df["low"].shift(1).rolling(32, min_periods=16).min()
    wave_range = wave_high - wave_low
    valid_wave = wave_range >= (2.0 * df["atr"])

    min_low_2 = pd.concat([df["low"], df["low"].shift(1)], axis=1).min(axis=1)
    max_high_2 = pd.concat([df["high"], df["high"].shift(1)], axis=1).max(axis=1)

    # Pullback zones (VWAP, EMA21/50, or 0.382-0.618 Fib)
    long_vwap = (min_low_2 <= df["vwap"] * 1.0025) & (df["close"] >= df["vwap"] * 0.995)
    long_ema = (min_low_2 <= df["ema21"] * 1.002) & (df["close"] >= df["ema50"] * 0.990)
    long_fib = (
        valid_wave
        & (min_low_2 <= wave_high - 0.35 * wave_range)
        & (min_low_2 >= wave_high - 0.65 * wave_range)
    )
    long_zone_hits = long_vwap.astype(int) + long_ema.astype(int) + long_fib.astype(int)
    long_not_chasing = df["close"] <= df["ema21"] * 1.012

    short_vwap = (max_high_2 >= df["vwap"] * 0.9975) & (df["close"] <= df["vwap"] * 1.005)
    short_ema = (max_high_2 >= df["ema21"] * 0.998) & (df["close"] <= df["ema50"] * 1.010)
    short_fib = (
        valid_wave
        & (max_high_2 >= wave_low + 0.35 * wave_range)
        & (max_high_2 <= wave_low + 0.65 * wave_range)
    )
    short_zone_hits = short_vwap.astype(int) + short_ema.astype(int) + short_fib.astype(int)
    short_not_chasing = df["close"] >= df["ema21"] * 0.988

    # RSI(14) cooldown & reversal turn
    rsi_min_prev3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).min(axis=1)
    rsi_max_prev3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).max(axis=1)

    rsi_long_ok = (rsi_min_prev3 < 42.0) & (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"] < 58.0)
    rsi_short_ok = (rsi_max_prev3 > 58.0) & (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"] > 42.0)

    # Align BTC 15M shock filter (strictly closed BTC candle)
    btc = df_btc_15m[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc["btc_bar_move_pct"] = ((btc["close"] - btc["open"]).abs() / btc["open"].replace(0.0, np.nan)) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_bar_move_pct"]], on="timestamp", direction="backward")
    df["btc_bar_move_pct"] = df["btc_bar_move_pct"].fillna(0.0)
    btc_ok = df["btc_bar_move_pct"] <= btc_shock_pause_pct

    # Align strictly closed 1H and 4H trend filters
    h1_t = prepare_htf_trend_v2(df_1h, context_interval_ms, "h1")
    h4_t = prepare_htf_trend_v2(df_4h, trend_interval_ms, "h4")
    df = pd.merge_asof(df, h1_t, on="close_ts", direction="backward")
    df = pd.merge_asof(df, h4_t, on="close_ts", direction="backward")

    for col in ("h1_trend_long", "h1_trend_short", "h4_trend_long", "h4_trend_short"):
        df[col] = df[col].astype("boolean").fillna(False).astype(bool)
    for col in ("h1_adx", "h4_adx"):
        df[col] = df[col].fillna(0.0)

    # Sanity bound on 15M ATR% so 2.8%-3.5% SL is well outside routine 15M noise
    atr_sane = (df["atr_pct"] >= 0.12) & (df["atr_pct"] <= 1.35)

    # Final strict binary entry signals
    df["sig_long"] = (
        df["h1_trend_long"]
        & df["h4_trend_long"]
        & (long_zone_hits >= 1)
        & long_not_chasing
        & rsi_long_ok
        & df["is_bull_candle"]
        & (df["close"] > df["close"].shift(1))
        & (~df["vol_cooldown"])
        & btc_ok
        & atr_sane
    )
    df["sig_short"] = (
        df["h1_trend_short"]
        & df["h4_trend_short"]
        & (short_zone_hits >= 1)
        & short_not_chasing
        & rsi_short_ok
        & df["is_bear_candle"]
        & (df["close"] < df["close"].shift(1))
        & (~df["vol_cooldown"])
        & btc_ok
        & atr_sane
    )

    # Clean quality score for ranking concurrent setups across the 25-coin universe
    trend_q = np.clip((df["h1_adx"] + df["h4_adx"]) / 2.0, 0.0, 50.0)
    rsi_depth_long = np.clip(45.0 - rsi_min_prev3.fillna(45.0), 0.0, 25.0)
    rsi_depth_short = np.clip(rsi_max_prev3.fillna(55.0) - 55.0, 0.0, 25.0)

    df["quality_long"] = trend_q + long_zone_hits * 12.0 + rsi_depth_long + df["body_ratio"] * 15.0
    df["quality_short"] = trend_q + short_zone_hits * 12.0 + rsi_depth_short + df["body_ratio"] * 15.0

    return df


def build_v2_position_plan(
    symbol: str,
    direction: str,
    entry_price: float,
    equity: float,
    tp_pct: float,
    sl_pct: float,
    entry_is_maker: bool,
    smart_be: bool = False,
    be_trigger_pct: float = 0.85,
    be_offset_pct: float = 0.15,
    max_risk_pct: float = 2.5,
    maker_fee_pct: float = 0.02,
    taker_fee_pct: float = 0.055,
    min_leverage: int = 3,
    target_max_leverage: int = 5,
    hard_max_leverage: int = 10,
    spec: Optional[InstrumentSpec] = None,
) -> Optional[V2PositionPlan]:
    """
    Strict risk-based position sizing:
        notional = (equity * max_risk_pct/100) / (sl_pct/100 + round_trip_fee)
    Automatic leverage within [3x, 5x] (hard cap 10x).
    """
    if equity <= 0 or entry_price <= 0 or sl_pct <= 0 or tp_pct <= 0:
        return None

    entry_fee_rate = (maker_fee_pct if entry_is_maker else taker_fee_pct) / 100.0
    sl_fee_rate = taker_fee_pct / 100.0
    round_trip_sl_fee = entry_fee_rate + sl_fee_rate

    sl_frac = sl_pct / 100.0
    tp_frac = tp_pct / 100.0
    risk_usdt = equity * (max_risk_pct / 100.0)

    notional = risk_usdt / (sl_frac + round_trip_sl_fee)

    # Automatic leverage inside 3x-5x (hard cap 10x) targeting ~20% margin per position
    raw_lev = notional / max(equity * 0.20, 1e-9)
    sym_max_lev = int(spec.max_leverage) if spec is not None else hard_max_leverage
    eff_max_lev = min(hard_max_leverage, max(min_leverage, sym_max_lev))
    lev_cap = min(target_max_leverage, eff_max_lev)
    leverage = int(max(min_leverage, min(lev_cap, round(raw_lev))))

    if spec is not None:
        qty = round_step_down(notional / entry_price, spec.qty_step)
        if qty < spec.min_order_qty or (qty * entry_price) < max(spec.min_notional, 5.0):
            return None
        notional = qty * entry_price
    else:
        qty = notional / entry_price

    margin_usdt = notional / leverage
    actual_risk_usdt = notional * (sl_frac + round_trip_sl_fee)
    actual_risk_pct = (actual_risk_usdt / equity) * 100.0

    if direction == "LONG":
        sl_price = entry_price * (1.0 - sl_frac)
        tp_price = entry_price * (1.0 + tp_frac)
        be_trigger = entry_price * (1.0 + be_trigger_pct / 100.0) if smart_be else None
        be_stop = entry_price * (1.0 + be_offset_pct / 100.0) if smart_be else None
    else:
        sl_price = entry_price * (1.0 + sl_frac)
        tp_price = entry_price * (1.0 - tp_frac)
        be_trigger = entry_price * (1.0 - be_trigger_pct / 100.0) if smart_be else None
        be_stop = entry_price * (1.0 - be_offset_pct / 100.0) if smart_be else None

    if spec is not None and spec.tick_size > 0:
        sl_price = round_tick(sl_price, spec.tick_size)
        tp_price = round_tick(tp_price, spec.tick_size)
        if be_trigger is not None:
            be_trigger = round_tick(be_trigger, spec.tick_size)
        if be_stop is not None:
            be_stop = round_tick(be_stop, spec.tick_size)

    return V2PositionPlan(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        be_trigger_price=be_trigger,
        be_stop_price=be_stop,
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        leverage=leverage,
        margin_usdt=margin_usdt,
        notional_usdt=notional,
        qty=qty,
        entry_is_maker=entry_is_maker,
        risk_usdt=actual_risk_usdt,
        risk_pct=actual_risk_pct,
    )
