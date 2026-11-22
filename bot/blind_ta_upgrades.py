"""100% Blind A/B Ablation Test of 5 Classical TA Upgrades on `fibo_trend_v2`.

Strictly offline using cached Top-100 pure crypto data in data/cache/.
Does NOT download data and does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_tpsl_grid import simulate_combo
from bot.blind_walkforward import load_cached_15m, slice_sym_data
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, compute_rsi, compute_rvol, ema


def detect_causal_rsi_divs_full(
    high: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    left: int = 3,
    right: int = 2,
    min_sep: int = 5,
    max_sep: int = 36,
    reg_veto_window: int = 28,
    hid_active_window: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Strictly causal pivot RSI divergence detector (zero lookahead):
      - reg_bear_veto: classic bearish RSI divergence (HH price, LH RSI) confirmed within last 28 bars -> vetoes LONG
      - reg_bull_veto: classic bullish RSI divergence (LL price, HL RSI) confirmed within last 28 bars -> vetoes SHORT
      - hid_bull_div:  hidden bullish RSI divergence (HL price, LL RSI) for LONG continuation
      - hid_bear_div:  hidden bearish RSI divergence (LH price, HH RSI) for SHORT continuation
    """
    n = len(low)
    reg_bear_veto = np.zeros(n, dtype=bool)
    reg_bull_veto = np.zeros(n, dtype=bool)
    hid_bull_div = np.zeros(n, dtype=bool)
    hid_bear_div = np.zeros(n, dtype=bool)

    pl_indices: List[int] = []
    ph_indices: List[int] = []
    reg_bear_cd = 0
    reg_bull_cd = 0
    hid_bull_cd = 0
    hid_bear_cd = 0

    for i in range(left + right, n):
        p = i - right

        # Confirmed pivot low at p
        if low[p] <= low[p - left : i + 1].min() and np.isfinite(rsi[p]):
            for prev_p in reversed(pl_indices):
                sep = p - prev_p
                if sep > max_sep:
                    break
                if sep < min_sep:
                    continue
                # Classic bullish divergence: price LL, RSI HL -> vetoes SHORT
                if low[p] <= low[prev_p] * 1.001 and rsi[p] > rsi[prev_p] + 2.0 and rsi[prev_p] < 42.0:
                    reg_bull_cd = reg_veto_window
                # Hidden bullish divergence: price HL, RSI LL -> supports LONG continuation
                if low[p] > low[prev_p] * 1.001 and rsi[p] < rsi[prev_p] - 1.5 and rsi[p] < 50.0:
                    hid_bull_cd = hid_active_window
            pl_indices.append(p)
            if len(pl_indices) > 10:
                pl_indices.pop(0)

        # Confirmed pivot high at p
        if high[p] >= high[p - left : i + 1].max() and np.isfinite(rsi[p]):
            for prev_p in reversed(ph_indices):
                sep = p - prev_p
                if sep > max_sep:
                    break
                if sep < min_sep:
                    continue
                # Classic bearish divergence: price HH, RSI LH -> vetoes LONG
                if high[p] >= high[prev_p] * 0.999 and rsi[p] < rsi[prev_p] - 2.0 and rsi[prev_p] > 58.0:
                    reg_bear_cd = reg_veto_window
                # Hidden bearish divergence: price LH, RSI HH -> supports SHORT continuation
                if high[p] < high[prev_p] * 0.999 and rsi[p] > rsi[prev_p] + 1.5 and rsi[p] > 50.0:
                    hid_bear_cd = hid_active_window
            ph_indices.append(p)
            if len(ph_indices) > 10:
                ph_indices.pop(0)

        # Also check live pullback low/high (bars i-1..i) vs last confirmed pivot for timely hidden div on bounce bar
        cur_low2 = min(low[i], low[i - 1])
        cur_rsi_lo2 = min(rsi[i], rsi[i - 1])
        for prev_p in reversed(pl_indices):
            sep = i - prev_p
            if sep > max_sep:
                break
            if sep >= min_sep and cur_low2 > low[prev_p] * 1.001 and cur_rsi_lo2 < rsi[prev_p] - 1.5 and cur_rsi_lo2 < 50.0:
                hid_bull_div[i] = True
                break

        cur_hi2 = max(high[i], high[i - 1])
        cur_rsi_hi2 = max(rsi[i], rsi[i - 1])
        for prev_p in reversed(ph_indices):
            sep = i - prev_p
            if sep > max_sep:
                break
            if sep >= min_sep and cur_hi2 < high[prev_p] * 0.999 and cur_rsi_hi2 > rsi[prev_p] + 1.5 and cur_rsi_hi2 > 50.0:
                hid_bear_div[i] = True
                break

        if reg_bear_cd > 0:
            reg_bear_veto[i] = True
            reg_bear_cd -= 1
        if reg_bull_cd > 0:
            reg_bull_veto[i] = True
            reg_bull_cd -= 1
        if hid_bull_cd > 0:
            hid_bull_div[i] = True
            hid_bull_cd -= 1
        if hid_bear_cd > 0:
            hid_bear_div[i] = True
            hid_bear_cd -= 1

    return {
        "reg_bear_veto": reg_bear_veto,
        "reg_bull_veto": reg_bull_veto,
        "hid_bull_div": hid_bull_div,
        "hid_bear_div": hid_bear_div,
    }


def prepare_btc_1h_trend(df_btc_15m: pd.DataFrame) -> pd.DataFrame:
    """Strictly closed BTC 1H EMA20/EMA50 trend table for the room_and_btc filter."""
    h1_ms = INTERVAL_TO_MS["60"]
    btc_1h = BybitClient.resample_klines(df_btc_15m, "60")
    e20 = ema(btc_1h["close"], 20)
    e50 = ema(btc_1h["close"], 50)
    out = pd.DataFrame(
        {
            "close_ts": (btc_1h["timestamp"] + h1_ms).astype("int64"),
            "btc_h1_strict_bear": (btc_1h["close"] < e50) & (e20 < e50),
            "btc_h1_strict_bull": (btc_1h["close"] > e50) & (e20 > e50),
        }
    )
    return out.sort_values("close_ts").reset_index(drop=True)


def build_ta_feature_bundle(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    btc_1h_trend: pd.DataFrame,
    timeline_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    Build exact baseline `fibo_trend_v2` (0.618-0.786, ADX>=20) signals PLUS
    individual boolean masks for all 5 classical TA upgrades so any subset can be composed in O(1).
    """
    entry_ms = INTERVAL_TO_MS["15"]
    h1_ms = INTERVAL_TO_MS["60"]
    h4_ms = INTERVAL_TO_MS["240"]

    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    df["rvol"] = compute_rvol(df["volume"], 20)

    c_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_ratio = ((df["close"] - df["open"]).abs() / c_range).fillna(0.0)
    lower_wick = ((pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]) / c_range).fillna(0.0)
    upper_wick = ((df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)) / c_range).fillna(0.0)

    # Common protective filters (identical to baseline)
    bar_range = df["high"] - df["low"]
    range_3 = df["high"].rolling(3, min_periods=3).max() - df["low"].rolling(3, min_periods=3).min()
    roc_3_abs = (df["close"] - df["close"].shift(3)).abs() / df["close"].shift(3).replace(0.0, np.nan)
    atr_frac = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)

    is_abn = ((bar_range > 2.2 * df["atr"]) | (range_3 > 3.8 * df["atr"]) | (roc_3_abs > 2.2 * atr_frac)).fillna(False)
    vol_cd = is_abn | is_abn.shift(1, fill_value=False) | is_abn.shift(2, fill_value=False) | is_abn.shift(3, fill_value=False)

    btc = df_btc_15m[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc["btc_move"] = ((btc["close"] - btc["open"]).abs() / btc["open"].replace(0.0, np.nan)) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_move"]], on="timestamp", direction="backward")
    df = pd.merge_asof(df, btc_1h_trend, on="close_ts", direction="backward")
    common_ok = (~vol_cd) & (df["btc_move"].fillna(0.0) <= 0.8) & (df["atr_pct"] >= 0.12) & (df["atr_pct"] <= 1.50)

    # Strictly closed 1H and 4H trends + ADX >= 20
    df_1h = BybitClient.resample_klines(df_15m, "60")
    df_4h = BybitClient.resample_klines(df_15m, "240")

    h1_out = pd.DataFrame(index=df_1h.index)
    h1_out["close_ts"] = (df_1h["timestamp"] + h1_ms).astype("int64")
    h1_e20 = ema(df_1h["close"], 20)
    h1_e50 = ema(df_1h["close"], 50)
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out["h1_bull"] = (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50) & (h1_adx >= 20.0)
    h1_out["h1_bear"] = (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50) & (h1_adx >= 20.0)
    h1_out["h1_adx"] = h1_adx.values

    h4_out = pd.DataFrame(index=df_4h.index)
    h4_out["close_ts"] = (df_4h["timestamp"] + h4_ms).astype("int64")
    h4_e20 = ema(df_4h["close"], 20)
    h4_e50 = ema(df_4h["close"], 50)
    h4_adx, _, _ = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    h4_out["h4_bull"] = (df_4h["close"] > h4_e50) & (h4_e20 > h4_e50) & (h4_adx >= 20.0)
    h4_out["h4_bear"] = (df_4h["close"] < h4_e50) & (h4_e20 < h4_e50) & (h4_adx >= 20.0)
    h4_out["h4_adx"] = h4_adx.values

    df = pd.merge_asof(df, h1_out.sort_values("close_ts"), on="close_ts", direction="backward")
    df = pd.merge_asof(df, h4_out.sort_values("close_ts"), on="close_ts", direction="backward")

    htf_bull = df["h1_bull"].astype("boolean").fillna(False) & df["h4_bull"].astype("boolean").fillna(False)
    htf_bear = df["h1_bear"].astype("boolean").fillna(False) & df["h4_bear"].astype("boolean").fillna(False)
    adx_avg = (df["h1_adx"].fillna(0.0) + df["h4_adx"].fillna(0.0)) / 2.0

    # RSI bounce + candle direction (identical to baseline)
    rsi_min3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).min(axis=1)
    rsi_max3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).max(axis=1)
    rsi_L = (rsi_min3 < 45.0) & (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"] < 62.0)
    rsi_S = (rsi_max3 > 55.0) & (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"] > 38.0)

    bounce_L = (df["close"] > df["open"]) & (body_ratio >= 0.35) & (df["close"] > df["close"].shift(1))
    bounce_S = (df["close"] < df["open"]) & (body_ratio >= 0.35) & (df["close"] < df["close"].shift(1))

    # Wave 36 Fibonacci zones
    w_hi = df["high"].shift(1).rolling(36, min_periods=36).max()
    w_lo = df["low"].shift(1).rolling(36, min_periods=36).min()
    w_rng = w_hi - w_lo
    w_ok = w_rng >= (2.2 * df["atr"])

    min_low_2 = pd.concat([df["low"], df["low"].shift(1)], axis=1).min(axis=1)
    max_high_2 = pd.concat([df["high"], df["high"].shift(1)], axis=1).max(axis=1)

    # Deep zone 0.618 - 0.786 (EXACT BASELINE)
    z618_L = w_ok & (min_low_2 <= w_hi - 0.60 * w_rng) & (min_low_2 >= w_hi - 0.80 * w_rng) & (df["close"] > w_hi - 0.786 * w_rng)
    z618_S = w_ok & (max_high_2 >= w_lo + 0.60 * w_rng) & (max_high_2 <= w_lo + 0.80 * w_rng) & (df["close"] < w_lo + 0.786 * w_rng)

    # Shallow zone 0.50 - 0.618 (for Upgrade #5: hidden_div_add)
    z500_only_L = w_ok & (min_low_2 <= w_hi - 0.48 * w_rng) & (min_low_2 > w_hi - 0.60 * w_rng)
    z500_only_S = w_ok & (max_high_2 >= w_lo + 0.48 * w_rng) & (max_high_2 < w_lo + 0.60 * w_rng)

    base_L = common_ok & htf_bull & z618_L & rsi_L & bounce_L
    base_S = common_ok & htf_bear & z618_S & rsi_S & bounce_S

    # --- Upgrade 1 & 5: Causal RSI Divergences ---
    divs = detect_causal_rsi_divs_full(
        df["high"].to_numpy(dtype=np.float64),
        df["low"].to_numpy(dtype=np.float64),
        df["rsi"].to_numpy(dtype=np.float64),
    )
    u1_ok_L = ~pd.Series(divs["reg_bear_veto"], index=df.index)
    u1_ok_S = ~pd.Series(divs["reg_bull_veto"], index=df.index)

    # --- Upgrade 2: VSA Volume Confirmation ---
    pullback_vol = df["volume"].shift(1).rolling(4, min_periods=3).mean()
    impulse_vol = df["volume"].shift(5).rolling(12, min_periods=6).mean()
    vsa_dry_pullback = (pullback_vol < impulse_vol * 0.95) & (df["volume"] > df["volume"].shift(1))
    u2_ok = (df["rvol"] >= 1.15) | vsa_dry_pullback

    # --- Upgrade 3: Price Action Pattern (Pin-bar wick >= 35% OR Engulfing previous body) ---
    prev_body_hi = pd.concat([df["open"].shift(1), df["close"].shift(1)], axis=1).max(axis=1)
    prev_body_lo = pd.concat([df["open"].shift(1), df["close"].shift(1)], axis=1).min(axis=1)
    bull_engulf = (df["close"] > df["open"]) & (df["close"] >= prev_body_hi) & (df["open"] <= prev_body_lo * 1.001)
    bear_engulf = (df["close"] < df["open"]) & (df["close"] <= prev_body_lo) & (df["open"] >= prev_body_hi * 0.999)
    u3_ok_L = (lower_wick >= 0.35) | bull_engulf
    u3_ok_S = (upper_wick >= 0.35) | bear_engulf

    # --- Upgrade 4: Room to Fib 0.0 >= 1.3% + BTC 1H trend not opposing ---
    room_L = ((w_hi - df["close"]) / df["close"].replace(0.0, np.nan) * 100.0) >= 1.3
    room_S = ((df["close"] - w_lo) / df["close"].replace(0.0, np.nan) * 100.0) >= 1.3
    btc_ok_L = ~df["btc_h1_strict_bear"].astype("boolean").fillna(False)
    btc_ok_S = ~df["btc_h1_strict_bull"].astype("boolean").fillna(False)
    u4_ok_L = room_L.fillna(False) & btc_ok_L
    u4_ok_S = room_S.fillna(False) & btc_ok_S

    # --- Upgrade 5: Additional entries in 0.50-0.618 with Hidden RSI Divergence ---
    u5_add_L = common_ok & htf_bull & z500_only_L & rsi_L & bounce_L & pd.Series(divs["hid_bull_div"], index=df.index)
    u5_add_S = common_ok & htf_bear & z500_only_S & rsi_S & bounce_S & pd.Series(divs["hid_bear_div"], index=df.index)

    df["base_L"] = base_L
    df["base_S"] = base_S
    df["q"] = (adx_avg + body_ratio * 25.0).fillna(0.0)
    df["u1_L"] = u1_ok_L
    df["u1_S"] = u1_ok_S
    df["u2"] = u2_ok.fillna(False)
    df["u3_L"] = u3_ok_L.fillna(False)
    df["u3_S"] = u3_ok_S.fillna(False)
    df["u4_L"] = u4_ok_L.fillna(False)
    df["u4_S"] = u4_ok_S.fillna(False)
    df["u5_L"] = u5_add_L.fillna(False)
    df["u5_S"] = u5_add_S.fillna(False)

    cols = [
        "timestamp", "high", "low", "close", "q",
        "base_L", "base_S", "u1_L", "u1_S", "u2", "u3_L", "u3_S", "u4_L", "u4_S", "u5_L", "u5_S",
    ]
    aligned = timeline_df.merge(df[cols], on="timestamp", how="left")
    out: Dict[str, np.ndarray] = {
        "high": aligned["high"].to_numpy(dtype=np.float64),
        "low": aligned["low"].to_numpy(dtype=np.float64),
        "close": aligned["close"].to_numpy(dtype=np.float64),
        "q": aligned["q"].fillna(0.0).to_numpy(dtype=np.float64),
    }
    for c in ("base_L", "base_S", "u1_L", "u1_S", "u2", "u3_L", "u3_S", "u4_L", "u4_S", "u5_L", "u5_S"):
        out[c] = aligned[c].astype("boolean").fillna(False).to_numpy(dtype=bool)
    return out


def compose_signals(
    bundle: Dict[str, Dict[str, np.ndarray]],
    active_upgrades: Tuple[str, ...],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compose (L, S, Q) arrays for any subset of upgrades ('div_veto','vol_confirm','price_action','room_and_btc','hidden_div_add')."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    u_set = set(active_upgrades)
    for sym, d in bundle.items():
        sig_L = d["base_L"].copy()
        sig_S = d["base_S"].copy()
        if "hidden_div_add" in u_set:
            sig_L = sig_L | d["u5_L"]
            sig_S = sig_S | d["u5_S"]
        if "div_veto" in u_set:
            sig_L = sig_L & d["u1_L"]
            sig_S = sig_S & d["u1_S"]
        if "vol_confirm" in u_set:
            sig_L = sig_L & d["u2"]
            sig_S = sig_S & d["u2"]
        if "price_action" in u_set:
            sig_L = sig_L & d["u3_L"]
            sig_S = sig_S & d["u3_S"]
        if "room_and_btc" in u_set:
            sig_L = sig_L & d["u4_L"]
            sig_S = sig_S & d["u4_S"]

        out[sym] = {
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "sig_L": sig_L,
            "sig_S": sig_S,
            "sig_Q": d["q"],
        }
    return out


def eval_blind_variant(
    name: str,
    upgrades: Tuple[str, ...],
    train_bundle: Dict[str, Dict[str, np.ndarray]],
    train_tl: np.ndarray,
    test_bundle: Dict[str, Dict[str, np.ndarray]],
    test_tl: np.ndarray,
    active_syms: List[str],
) -> Dict[str, Any]:
    tr_data = compose_signals(train_bundle, upgrades)
    te_data = compose_signals(test_bundle, upgrades)

    # 1. Blind Top-25 ranking strictly on first 4 months (120d)
    ranked: List[Tuple[float, float, int, str]] = []
    for sym in active_syms:
        r_tr = simulate_combo(
            symbols=[sym],
            sym_data={sym: tr_data[sym]},
            timeline=train_tl,
            tp_pct=1.50,
            sl_pct=3.50,
            smart_be=True,
            margin_pct=15.0,
            leverage=10.0,
            max_open=1,
            max_same_dir=1,
            max_daily_stops=4,
            strat_key="sig",
        )
        ranked.append((r_tr["net_pct"], r_tr["win_rate_pct"], r_tr["trades"], sym))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_25 = [x[3] for x in ranked[:25]]

    # 2. Blind 60-day portfolio test on those Top-25 starting at $1,000
    res = simulate_combo(
        symbols=top_25,
        sym_data=te_data,
        timeline=test_tl,
        tp_pct=1.50,
        sl_pct=3.50,
        smart_be=True,
        margin_pct=15.0,
        leverage=10.0,
        max_open=5,
        max_same_dir=3,
        max_daily_stops=4,
        strat_key="sig",
    )
    return {"name": name, "upgrades": list(upgrades), "top25": top_25, **res}


def is_better_than_base(cand: Dict[str, Any], base: Dict[str, Any]) -> Tuple[bool, str]:
    """Determine objectively whether an upgrade improves on baseline."""
    base_end = base["end_usd"]
    base_dd = base["max_dd_pct"]
    base_calmar = base["net_pct"] / max(base_dd, 1.0)
    cand_calmar = cand["net_pct"] / max(cand["max_dd_pct"], 1.0)

    if cand["end_usd"] > base_end:
        return True, "ЛУЧШЕ (выше прибыль)"
    # Noticeably lower drawdown/stops without collapsing profit (keeps >= 88% of base profit & better Return/DD)
    if cand["end_usd"] >= base_end * 0.88 and cand["max_dd_pct"] <= base_dd - 4.0 and cand_calmar > base_calmar:
        return True, "ЛУЧШЕ (ниже просадка и лучше риск/доходность)"
    return False, "ХУЖЕ"


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    symbols_100 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    btc_1h_trend = prepare_btc_1h_trend(df_btc)

    sim_end_ms = int(df_btc["timestamp"].iloc[-1])
    sim_start_ms = sim_end_ms - 180 * 86_400_000
    split_ms = sim_start_ms + 120 * 86_400_000

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]
    timeline = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    train_mask = timeline < split_ms
    test_mask = timeline >= split_ms
    train_tl = timeline[train_mask]
    test_tl = timeline[test_mask]

    print("Building causal TA feature bundles across cached Top-100...")
    bundle: Dict[str, Dict[str, np.ndarray]] = {}
    active_syms: List[str] = []
    for sym in symbols_100:
        if not (cache_dir / f"{sym}_15.csv.gz").exists():
            continue
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 400:
            continue
        bundle[sym] = build_ta_feature_bundle(df_s, df_btc, btc_1h_trend, timeline_df)
        active_syms.append(sym)

    train_bundle = slice_sym_data(bundle, train_mask)
    test_bundle = slice_sym_data(bundle, test_mask)

    # Baseline verification
    base = eval_blind_variant("БАЗА (fibo_trend_v2 эталон)", (), train_bundle, train_tl, test_bundle, test_tl, active_syms)

    # Step 1: Strictly individual A/B tests of all 5 upgrades
    single_tests = [
        ("div_veto", "1. Щит от обычной дивергенции (div_veto)"),
        ("vol_confirm", "2. Подтверждение объёмом VSA (vol_confirm)"),
        ("price_action", "3. Паттерн отбоя Pin-bar / Engulfing (price_action)"),
        ("room_and_btc", "4. Запас хода >= 1.3% + тренд BTC 1H (room_and_btc)"),
        ("hidden_div_add", "5. Добавление 0.50-0.618 со скрытой дивергенцией (hidden_div_add)"),
    ]

    singles: List[Dict[str, Any]] = []
    winners: List[str] = []

    print("\n" + "=" * 104)
    print(
        f"BASELINE CHECK: $1,000 -> ${base['end_usd']:,.2f} ({base['net_pct']:+.1f}%) | "
        f"{base['trades']} tr ({base['trades_per_day']:.2f}/d) | WR {base['win_rate_pct']:.2f}% | "
        f"SL={base['sl_count']} | MaxDD={base['max_dd_pct']:.2f}% (${base['max_dd_usd']:.2f})"
    )
    print("=" * 104)

    for key, label in single_tests:
        res = eval_blind_variant(label, (key,), train_bundle, train_tl, test_bundle, test_tl, active_syms)
        better, verdict = is_better_than_base(res, base)
        res["better"] = better
        res["verdict"] = verdict
        singles.append(res)
        if better:
            winners.append(key)

        print(f"\n>>> {label} => {verdict}")
        print(
            f"  Trades: {res['trades']} ({res['trades_per_day']:.2f}/d) | WR: {res['win_rate_pct']:.2f}% | "
            f"SL: {res['sl_count']} (base: {base['sl_count']}) [TP={res['tp_count']}, BE={res['be_count']}]"
        )
        print(
            f"  Blind 2m: $1,000 -> ${res['end_usd']:,.2f} ({res['net_pct']:+.2f}%) | "
            f"MaxDD: {res['max_dd_pct']:.2f}% (${res['max_dd_usd']:.2f})"
        )

    # Step 2: Combine ONLY winning upgrades (test all pairs + full joint of top 2-3 winners)
    combos_tested: List[Dict[str, Any]] = []
    if len(winners) >= 2:
        # Sort winners by single-test performance and cap to top 3
        single_by_key = {r["upgrades"][0]: r for r in singles}
        winners.sort(key=lambda k: (single_by_key[k]["end_usd"], -single_by_key[k]["max_dd_pct"]), reverse=True)
        top_winners = winners[:3]

        combo_subsets: List[Tuple[str, ...]] = []
        for r_len in range(2, len(top_winners) + 1):
            for comb in combinations(top_winners, r_len):
                combo_subsets.append(comb)

        for comb in combo_subsets:
            c_label = " + ".join(comb)
            res = eval_blind_variant(f"КОМБИНАЦИЯ ({c_label})", comb, train_bundle, train_tl, test_bundle, test_tl, active_syms)
            better, verdict = is_better_than_base(res, base)
            res["better"] = better
            res["verdict"] = verdict
            combos_tested.append(res)

        combos_tested.sort(key=lambda r: (r["end_usd"], -r["max_dd_pct"]), reverse=True)
    elif len(winners) == 1:
        combos_tested = [next(r for r in singles if r["upgrades"] == winners)]

    print("\n" + "=" * 104)
    print(f"STEP 2: COMBINATIONS OF WINNING UPGRADES (winners={winners})")
    print("=" * 104)
    if not combos_tested:
        print("  No individual upgrades beat the baseline; baseline remains champion.")
    else:
        for res in combos_tested:
            print(f"\n>>> {res['name']} => {res['verdict']}")
            print(
                f"  Trades: {res['trades']} ({res['trades_per_day']:.2f}/d) | WR: {res['win_rate_pct']:.2f}% | "
                f"SL: {res['sl_count']} (base: {base['sl_count']}) [TP={res['tp_count']}, BE={res['be_count']}]"
            )
            print(
                f"  Blind 2m: $1,000 -> ${res['end_usd']:,.2f} ({res['net_pct']:+.2f}%) | "
                f"MaxDD: {res['max_dd_pct']:.2f}% (${res['max_dd_usd']:.2f})"
            )

    out = {
        "baseline": base,
        "singles": singles,
        "winning_keys": winners,
        "combinations": combos_tested,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path("data/blind_ta_upgrades_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/blind_ta_upgrades_results.json")


if __name__ == "__main__":
    main()
