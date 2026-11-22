"""100% Blind ADX Ablation Test (ADX off vs >=15 vs >=20) for `fibo_trend_v2` on Top-25 (Max 5 pos)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_scale_test import simulate_scaled_10x
from bot.blind_walkforward import load_cached_15m, slice_sym_data
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, compute_rsi, ema


def build_adx_signals_for_symbol(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    timeline_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Compute strictly causal `fibo_trend_v2` (0.618-0.786) signals for ADX thresholds 0, 15, 20."""
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

    # Common protective filters
    bar_range = df["high"] - df["low"]
    range_3 = df["high"].rolling(3, min_periods=3).max() - df["low"].rolling(3, min_periods=3).min()
    roc_3_abs = (df["close"] - df["close"].shift(3)).abs() / df["close"].shift(3).replace(0.0, np.nan)
    atr_frac = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)

    is_abn = ((bar_range > 2.2 * df["atr"]) | (range_3 > 3.8 * df["atr"]) | (roc_3_abs > 2.2 * atr_frac)).fillna(False)
    vol_cd = is_abn | is_abn.shift(1, fill_value=False) | is_abn.shift(2, fill_value=False) | is_abn.shift(3, fill_value=False)

    btc = df_btc_15m[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc["btc_move"] = ((btc["close"] - btc["open"]).abs() / btc["open"].replace(0.0, np.nan)) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_move"]], on="timestamp", direction="backward")
    common_ok = (~vol_cd) & (df["btc_move"].fillna(0.0) <= 0.8) & (df["atr_pct"] >= 0.12) & (df["atr_pct"] <= 1.50)

    # Strictly closed 1H and 4H trend + ADX values
    df_1h = BybitClient.resample_klines(df_15m, "60")
    df_4h = BybitClient.resample_klines(df_15m, "240")

    h1_out = pd.DataFrame(index=df_1h.index)
    h1_out["close_ts"] = (df_1h["timestamp"] + h1_ms).astype("int64")
    h1_e20 = ema(df_1h["close"], 20)
    h1_e50 = ema(df_1h["close"], 50)
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out["h1_bull"] = (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50)
    h1_out["h1_bear"] = (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50)
    h1_out["h1_adx"] = h1_adx.values

    h4_out = pd.DataFrame(index=df_4h.index)
    h4_out["close_ts"] = (df_4h["timestamp"] + h4_ms).astype("int64")
    h4_e20 = ema(df_4h["close"], 20)
    h4_e50 = ema(df_4h["close"], 50)
    h4_adx, _, _ = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    h4_out["h4_bull"] = (df_4h["close"] > h4_e50) & (h4_e20 > h4_e50)
    h4_out["h4_bear"] = (df_4h["close"] < h4_e50) & (h4_e20 < h4_e50)
    h4_out["h4_adx"] = h4_adx.values

    df = pd.merge_asof(df, h1_out.sort_values("close_ts"), on="close_ts", direction="backward")
    df = pd.merge_asof(df, h4_out.sort_values("close_ts"), on="close_ts", direction="backward")

    htf_bull_raw = df["h1_bull"].astype("boolean").fillna(False) & df["h4_bull"].astype("boolean").fillna(False)
    htf_bear_raw = df["h1_bear"].astype("boolean").fillna(False) & df["h4_bear"].astype("boolean").fillna(False)
    h1_adx_s = df["h1_adx"].fillna(0.0)
    h4_adx_s = df["h4_adx"].fillna(0.0)
    adx_avg = (h1_adx_s + h4_adx_s) / 2.0

    # RSI bounce + deep Fib 0.618-0.786 (wave 36)
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

    base_L = common_ok & htf_bull_raw & z618_L & rsi_L & bounce_L
    base_S = common_ok & htf_bear_raw & z618_S & rsi_S & bounce_S
    q_val = (adx_avg + body_ratio * 25.0).fillna(0.0)

    for adx_min in (0, 15, 20):
        adx_ok = True if adx_min == 0 else ((h1_adx_s >= float(adx_min)) & (h4_adx_s >= float(adx_min)))
        df[f"adx{adx_min}_L"] = base_L & adx_ok
        df[f"adx{adx_min}_S"] = base_S & adx_ok
        df[f"adx{adx_min}_Q"] = q_val

    cols = [
        "timestamp", "high", "low", "close",
        "adx0_L", "adx0_S", "adx0_Q",
        "adx15_L", "adx15_S", "adx15_Q",
        "adx20_L", "adx20_S", "adx20_Q",
    ]
    aligned = timeline_df.merge(df[cols], on="timestamp", how="left")
    out: Dict[str, np.ndarray] = {
        "high": aligned["high"].to_numpy(dtype=np.float64),
        "low": aligned["low"].to_numpy(dtype=np.float64),
        "close": aligned["close"].to_numpy(dtype=np.float64),
    }
    for adx_min in (0, 15, 20):
        out[f"adx{adx_min}_L"] = aligned[f"adx{adx_min}_L"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        out[f"adx{adx_min}_S"] = aligned[f"adx{adx_min}_S"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        out[f"adx{adx_min}_Q"] = aligned[f"adx{adx_min}_Q"].fillna(0.0).to_numpy(dtype=np.float64)
    return out


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

    train_mask = timeline < split_ms
    test_mask = timeline >= split_ms
    train_tl = timeline[train_mask]
    test_tl = timeline[test_mask]

    sym_data: Dict[str, Dict[str, np.ndarray]] = {}
    active_syms: List[str] = []
    for sym in symbols_100:
        if not (cache_dir / f"{sym}_15.csv.gz").exists():
            continue
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 400:
            continue
        sym_data[sym] = build_adx_signals_for_symbol(df_s, df_btc, timeline_df)
        active_syms.append(sym)

    train_data = slice_sym_data(sym_data, train_mask)
    test_data = slice_sym_data(sym_data, test_mask)

    summary: Dict[str, Any] = {}
    print("\n" + "=" * 96)
    print("BLIND ADX ABLATION (Top-25 by first 4m | Last 60d BLIND | Max 5 pos (3/dir) | 4 stops/day | 10x)")
    print("=" * 96)

    for adx_min, label in ((0, "Без ADX (отключён)"), (15, "Мягкий ADX >= 15"), (20, "Текущий ADX >= 20")):
        s_key = f"adx{adx_min}"

        # Blind Top-25 ranking strictly on first 4 months
        ranked: List[Tuple[float, float, int, str]] = []
        for sym in active_syms:
            r_tr = simulate_scaled_10x(
                symbols=[sym],
                sym_data={sym: train_data[sym]},
                timeline=train_tl,
                margin_pct=10.0,
                leverage=10.0,
                max_open=1,
                max_same_dir=1,
                max_daily_stops=4,
                strat_key=s_key,
            )
            ranked.append((r_tr["net_pct"], r_tr["win_rate_pct"], r_tr["trades"], sym))
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_25 = [x[3] for x in ranked[:25]]

        res_10 = simulate_scaled_10x(
            symbols=top_25,
            sym_data=test_data,
            timeline=test_tl,
            margin_pct=10.0,
            leverage=10.0,
            max_open=5,
            max_same_dir=3,
            max_daily_stops=4,
            strat_key=s_key,
        )
        res_15 = simulate_scaled_10x(
            symbols=top_25,
            sym_data=test_data,
            timeline=test_tl,
            margin_pct=15.0,
            leverage=10.0,
            max_open=5,
            max_same_dir=3,
            max_daily_stops=4,
            strat_key=s_key,
        )
        summary[s_key] = {"label": label, "top25": top_25, "margin_10": res_10, "margin_15": res_15}

        print(f"\n>>> {label}")
        print(
            f"  Trades: {res_10['trades']} ({res_10['trades_per_day']:.2f}/d) | Blind WR: {res_10['win_rate_pct']:.2f}% "
            f"(TP={res_10['tp_count']}, BE={res_10['be_count']}, SL={res_10['sl_count']})"
        )
        print(
            f"  Margin 10%: $1,000 -> ${res_10['end_usd']:,.2f} ({res_10['net_pct']:+.2f}%) | "
            f"Max DD: ${res_10['max_dd_usd']:.2f} ({res_10['max_dd_pct']:.2f}%)"
        )
        print(
            f"  Margin 15%: $1,000 -> ${res_15['end_usd']:,.2f} ({res_15['net_pct']:+.2f}%) | "
            f"Max DD: ${res_15['max_dd_usd']:.2f} ({res_15['max_dd_pct']:.2f}%)"
        )

    Path("data/blind_adx_results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/blind_adx_results.json")


if __name__ == "__main__":
    main()
