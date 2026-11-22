"""Technical indicators, causal divergences, volume profile, and multi-timeframe alignment."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's Moving Average (RMA) used in RSI, ATR, and ADX."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    only_gains = (avg_loss == 0.0) & (avg_gain > 0.0)
    rsi = rsi.mask(only_gains, 100.0)
    return rsi.fillna(50.0)


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    tp_diff = tp.diff()
    pos_mf = np.where(tp_diff > 0.0, rmf, 0.0)
    neg_mf = np.where(tp_diff < 0.0, rmf, 0.0)
    pos_sum = pd.Series(pos_mf, index=close.index).rolling(period, min_periods=period).sum()
    neg_sum = pd.Series(neg_mf, index=close.index).rolling(period, min_periods=period).sum()
    mfr = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    return mfi.fillna(50.0)


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return rma(tr, period)


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=close.index,
    )
    atr_val = compute_atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100.0 * rma(plus_dm, period) / atr_val
    minus_di = 100.0 * rma(minus_dm, period) / atr_val
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx_val = rma(dx.fillna(0.0), period)
    return adx_val.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)


def compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """UTC-midnight anchored daily VWAP."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    day_key = (df["timestamp"] // 86_400_000).astype("int64")
    pv = (tp * df["volume"]).groupby(day_key).cumsum()
    cum_vol = df["volume"].groupby(day_key).cumsum().replace(0.0, np.nan)
    return (pv / cum_vol).ffill().fillna(df["close"])


def compute_rvol(volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative volume against prior `period` closed bars (excluding current bar)."""
    baseline = volume.shift(1).rolling(period, min_periods=max(5, period // 2)).mean()
    return (volume / baseline.replace(0.0, np.nan)).fillna(1.0)


def detect_causal_divergences(
    high: np.ndarray,
    low: np.ndarray,
    rsi_arr: np.ndarray,
    mfi_arr: np.ndarray,
    left: int = 3,
    right: int = 2,
    min_sep: int = 5,
    max_sep: int = 36,
    active_window: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Strictly causal pivot-based RSI and MFI divergence detector.
    A pivot at index `p` is ONLY confirmed at index `i = p + right` (zero lookahead).
    Signal stays active for `active_window` closed bars after confirmation.
    """
    n = len(low)
    bull_both = np.zeros(n, dtype=bool)
    bull_single = np.zeros(n, dtype=bool)
    bear_both = np.zeros(n, dtype=bool)
    bear_single = np.zeros(n, dtype=bool)

    last_pl_idx = -1
    last_ph_idx = -1
    bull_both_cd = 0
    bull_single_cd = 0
    bear_both_cd = 0
    bear_single_cd = 0

    for i in range(left + right, n):
        p = i - right

        # Check confirmed pivot low at p using window [p - left, p + right] == [p - left, i]
        win_low = low[p - left : i + 1]
        if low[p] <= win_low.min() and np.isfinite(rsi_arr[p]) and np.isfinite(mfi_arr[p]):
            if last_pl_idx >= 0 and min_sep <= (p - last_pl_idx) <= max_sep:
                # Bullish divergence: lower (or equal/swept) price low, higher oscillator low
                price_ll = low[p] <= low[last_pl_idx] * 1.0015
                rsi_hl = rsi_arr[p] > rsi_arr[last_pl_idx] + 1.5 and rsi_arr[last_pl_idx] < 45.0
                mfi_hl = mfi_arr[p] > mfi_arr[last_pl_idx] + 2.0 and mfi_arr[last_pl_idx] < 45.0
                if price_ll and rsi_hl and mfi_hl:
                    bull_both_cd = active_window
                elif price_ll and (rsi_hl or mfi_hl):
                    bull_single_cd = max(bull_single_cd, active_window)
            last_pl_idx = p

        # Check confirmed pivot high at p
        win_high = high[p - left : i + 1]
        if high[p] >= win_high.max() and np.isfinite(rsi_arr[p]) and np.isfinite(mfi_arr[p]):
            if last_ph_idx >= 0 and min_sep <= (p - last_ph_idx) <= max_sep:
                price_hh = high[p] >= high[last_ph_idx] * 0.9985
                rsi_lh = rsi_arr[p] < rsi_arr[last_ph_idx] - 1.5 and rsi_arr[last_ph_idx] > 55.0
                mfi_lh = mfi_arr[p] < mfi_arr[last_ph_idx] - 2.0 and mfi_arr[last_ph_idx] > 55.0
                if price_hh and rsi_lh and mfi_lh:
                    bear_both_cd = active_window
                elif price_hh and (rsi_lh or mfi_lh):
                    bear_single_cd = max(bear_single_cd, active_window)
            last_ph_idx = p

        if bull_both_cd > 0:
            bull_both[i] = True
            bull_both_cd -= 1
        if bull_single_cd > 0:
            bull_single[i] = True
            bull_single_cd -= 1
        if bear_both_cd > 0:
            bear_both[i] = True
            bear_both_cd -= 1
        if bear_single_cd > 0:
            bear_single[i] = True
            bear_single_cd -= 1

    return {
        "div_bull_both": bull_both,
        "div_bull_single": bull_single,
        "div_bear_both": bear_both,
        "div_bear_single": bear_single,
    }


def check_volume_profile_clearance(
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    idx: int,
    direction: str,
    entry_price: float,
    tp1_price: float,
    tp2_price: float,
    lookback: int = 96,
    n_bins: int = 24,
    shelf_multiplier: float = 1.65,
) -> Tuple[bool, bool]:
    """
    Build a causal horizontal volume profile from the last `lookback` closed candles
    ending at `idx` and check if a dense opposing volume shelf blocks TP1 or TP2.
    Returns (clear_to_tp1, clear_to_tp2).
    """
    start = max(0, idx - lookback + 1)
    h_win = highs[start : idx + 1]
    l_win = lows[start : idx + 1]
    v_win = volumes[start : idx + 1]

    if len(h_win) < 10:
        return True, True

    p_min = float(np.min(l_win))
    p_max = float(np.max(h_win))
    if p_max <= p_min:
        return True, True

    edges = np.linspace(p_min, p_max, n_bins + 1)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = (p_max - p_min) / n_bins

    # Fast approximate volume-by-price distribution across candle [low, high] span
    lo_bins = np.clip(((l_win - p_min) / bin_width).astype(int), 0, n_bins - 1)
    hi_bins = np.clip(((h_win - p_min) / bin_width).astype(int), 0, n_bins - 1)
    span_counts = (hi_bins - lo_bins + 1).astype(np.float64)
    vol_per_bin = v_win / span_counts

    # Difference array trick for O(lookback + n_bins) exact range-add!
    diff = np.zeros(n_bins + 1, dtype=np.float64)
    np.add.at(diff, lo_bins, vol_per_bin)
    np.add.at(diff, hi_bins + 1, -vol_per_bin)
    profile = np.cumsum(diff[:-1])

    mean_vol = float(np.mean(profile))
    if mean_vol <= 0:
        return True, True

    dense_threshold = max(mean_vol * shelf_multiplier, float(np.percentile(profile, 82)))
    dense_mask = profile >= dense_threshold

    # Exclude the immediate entry bin (support/origin zone) so only opposing shelves ahead count
    buffer = entry_price * 0.0015
    if direction == "LONG":
        tp1_path = (bin_centers > entry_price + buffer) & (bin_centers <= tp1_price)
        tp2_path = (bin_centers > tp1_price) & (bin_centers <= tp2_price)
    else:
        tp1_path = (bin_centers < entry_price - buffer) & (bin_centers >= tp1_price)
        tp2_path = (bin_centers < tp1_price) & (bin_centers >= tp2_price)

    clear_tp1 = not bool(np.any(dense_mask & tp1_path))
    clear_tp2 = clear_tp1 and not bool(np.any(dense_mask & tp2_path))
    return clear_tp1, clear_tp2


def prepare_higher_tf_features(df_htf: pd.DataFrame, interval_ms: int, prefix: str) -> pd.DataFrame:
    """Compute EMA20, EMA50, ADX14, +DI, -DI on a higher timeframe indexed by bar close time."""
    out = pd.DataFrame(index=df_htf.index)
    out["close_ts"] = (df_htf["timestamp"] + interval_ms).astype("int64")
    ema20 = ema(df_htf["close"], 20)
    ema50 = ema(df_htf["close"], 50)
    adx_val, pdi, mdi = compute_adx(df_htf["high"], df_htf["low"], df_htf["close"], 14)

    out[f"{prefix}_bull"] = (df_htf["close"] > ema20) & (ema20 >= ema50 * 0.998)
    out[f"{prefix}_bear"] = (df_htf["close"] < ema20) & (ema20 <= ema50 * 1.002)
    out[f"{prefix}_adx"] = adx_val.values
    out[f"{prefix}_pdi"] = pdi.values
    out[f"{prefix}_mdi"] = mdi.values
    return out.sort_values("close_ts").reset_index(drop=True)


def enrich_entry_dataframe(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    entry_interval_ms: int = 15 * 60_000,
    context_interval_ms: int = 60 * 60_000,
    trend_interval_ms: int = 240 * 60_000,
) -> pd.DataFrame:
    """
    Compute all 15M indicators, causal divergences, liquidity sweeps, VWAP,
    relative strength vs BTC, and strictly closed 1H/4H context on df_15m.
    """
    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_interval_ms).astype("int64")

    # Core oscillators & volatility
    df["rsi"] = compute_rsi(df["close"], 14)
    df["mfi"] = compute_mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    adx_15, pdi_15, mdi_15 = compute_adx(df["high"], df["low"], df["close"], 14)
    df["adx_15m"] = adx_15
    df["vwap"] = compute_daily_vwap(df)
    df["rvol"] = compute_rvol(df["volume"], 20)

    # Candle anatomy
    c_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / c_range).fillna(0.0)
    df["is_bull_candle"] = df["close"] > df["open"]
    df["is_bear_candle"] = df["close"] < df["open"]

    # Local swing levels for liquidity sweeps & structural SL (strictly shifted by 1 for sweeps)
    prev_swing_low = df["low"].shift(1).rolling(20, min_periods=10).min()
    prev_swing_high = df["high"].shift(1).rolling(20, min_periods=10).max()
    df["swing_low_10"] = df["low"].rolling(10, min_periods=5).min()
    df["swing_high_10"] = df["high"].rolling(10, min_periods=5).max()

    # Liquidity sweep (current or previous closed bar swept local extreme & reclaimed)
    swept_low_now = (df["low"] < prev_swing_low) & (df["close"] > prev_swing_low)
    swept_low_prev = (df["low"].shift(1) < prev_swing_low.shift(1)) & (df["close"] > prev_swing_low.shift(1))
    df["bull_sweep"] = (swept_low_now | swept_low_prev) & df["is_bull_candle"]

    swept_high_now = (df["high"] > prev_swing_high) & (df["close"] < prev_swing_high)
    swept_high_prev = (df["high"].shift(1) > prev_swing_high.shift(1)) & (df["close"] < prev_swing_high.shift(1))
    df["bear_sweep"] = (swept_high_now | swept_high_prev) & df["is_bear_candle"]

    # Daily VWAP retest / reclaim hold
    vwap = df["vwap"]
    df["bull_vwap_retest"] = (
        (df["close"] > vwap)
        & (
            (df["low"] <= vwap * 1.002)
            | ((df["close"].shift(1) <= vwap.shift(1)) & (df["close"] > vwap))
        )
        & df["is_bull_candle"]
    )
    df["bear_vwap_retest"] = (
        (df["close"] < vwap)
        & (
            (df["high"] >= vwap * 0.998)
            | ((df["close"].shift(1) >= vwap.shift(1)) & (df["close"] < vwap))
        )
        & df["is_bear_candle"]
    )
    df["bull_vwap_near"] = (df["close"] > vwap) & ((df["close"] - vwap) / vwap <= 0.005)
    df["bear_vwap_near"] = (df["close"] < vwap) & ((vwap - df["close"]) / vwap <= 0.005)

    # Causal RSI & MFI divergences
    divs = detect_causal_divergences(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["rsi"].to_numpy(dtype=np.float64),
        df["mfi"].to_numpy(dtype=np.float64),
    )
    for k, arr in divs.items():
        df[k] = arr

    # Momentum hook helper (oversold/overbought turn in RSI + MFI)
    df["bull_osc_turn"] = (
        (df["rsi"] > df["rsi"].shift(1))
        & (df["mfi"] > df["mfi"].shift(1))
        & (df["rsi"].shift(1) < 42.0)
        & (df["mfi"].shift(1) < 42.0)
    )
    df["bear_osc_turn"] = (
        (df["rsi"] < df["rsi"].shift(1))
        & (df["mfi"] < df["mfi"].shift(1))
        & (df["rsi"].shift(1) > 58.0)
        & (df["mfi"].shift(1) > 58.0)
    )

    # Align BTC 15M series for BTC shock filter & Relative Strength
    btc = df_btc_15m[["timestamp", "open", "high", "low", "close"]].copy().sort_values("timestamp")
    btc["btc_bar_move_pct"] = ((btc["close"] - btc["open"]).abs() / btc["open"].replace(0.0, np.nan)) * 100.0
    btc["btc_ret_4"] = btc["close"].pct_change(4)
    btc["btc_ret_16"] = btc["close"].pct_change(16)
    btc = btc[["timestamp", "btc_bar_move_pct", "btc_ret_4", "btc_ret_16"]]

    df["coin_ret_4"] = df["close"].pct_change(4)
    df["coin_ret_16"] = df["close"].pct_change(16)
    df = pd.merge_asof(df, btc, on="timestamp", direction="backward")

    df["rs_4"] = (df["coin_ret_4"] - df["btc_ret_4"]).fillna(0.0)
    df["rs_16"] = (df["coin_ret_16"] - df["btc_ret_16"]).fillna(0.0)

    # Align strictly closed 1H and 4H context (using bar close_ts so zero lookahead!)
    h1_feat = prepare_higher_tf_features(df_1h, context_interval_ms, "h1")
    h4_feat = prepare_higher_tf_features(df_4h, trend_interval_ms, "h4")

    df = pd.merge_asof(df, h1_feat, on="close_ts", direction="backward")
    df = pd.merge_asof(df, h4_feat, on="close_ts", direction="backward")

    for col in ("h1_bull", "h1_bear", "h4_bull", "h4_bear"):
        df[col] = df[col].astype("boolean").fillna(False).astype(bool)
    for col in ("h1_adx", "h4_adx", "btc_bar_move_pct"):
        df[col] = df[col].fillna(0.0)

    return df
