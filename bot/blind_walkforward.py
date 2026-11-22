"""100% Blind Walk-Forward Exam on cached Top-100 pure crypto coins.

Rules:
- Strictly offline using data/cache/*.csv.gz (zero API downloads).
- For every strategy:
  1. Select Top-12 coins STRICTLY from the first 120 days (4 months) without seeing the last 60 days.
  2. Run those exact 12 coins BLIND on the last 60 days (2 months) starting at $1,000.
- Fixed money rules: 10% margin x 10x leverage (100% notional), max 3 open (2 same dir),
  daily loss limit -10.5%, Maker limit entry (0.02%, 0.065% pullback, 2-bar TTL),
  Maker TP (0.02%), Taker SL/BE (0.055%), 3-bar cooldown.
"""

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

from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, compute_rsi, ema
from bot.strategy_arena import (
    ArenaConfig,
    compute_all_strategy_signals,
    simulate_fixed_10x,
)


def load_cached_15m(symbol: str, cache_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(cache_dir / f"{symbol}_15.csv.gz", compression="gzip")
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = df[col].astype("float64")
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def add_fibo_v2_signals(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    timeline_df: pd.DataFrame,
    base_arrays: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Add strictly causal `fibo_trend_v2` signals (both 0.618-0.786 and 0.50-0.786 zones):
      - Wave 36 on 15M
      - Mandatory 4H trend + 1H trend + ADX >= 20 (anti-chop)
      - Cooled RSI (<45 for LONG / >55 for SHORT) turning in direction + bounce candle
      - Common anti-pump/dump + 3-bar vol cooldown + BTC shock <= 0.8%
    """
    entry_ms = INTERVAL_TO_MS["15"]
    h1_ms = INTERVAL_TO_MS["60"]
    h4_ms = INTERVAL_TO_MS["240"]

    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    df["ema5"] = ema(df["close"], 5)

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
    adx_avg = ((df["h1_adx"].fillna(0.0) + df["h4_adx"].fillna(0.0)) / 2.0)

    # Cooled RSI (<45 LONG, >55 SHORT) turning in direction + bounce candle
    rsi_min3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).min(axis=1)
    rsi_max3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).max(axis=1)

    rsi_bounce_L = (rsi_min3 < 45.0) & (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"] < 62.0)
    rsi_bounce_S = (rsi_max3 > 55.0) & (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"] > 38.0)

    bounce_L = (df["close"] > df["open"]) & (body_ratio >= 0.35) & (df["close"] > df["close"].shift(1))
    bounce_S = (df["close"] < df["open"]) & (body_ratio >= 0.35) & (df["close"] < df["close"].shift(1))

    # Wave 36 Fibonacci retracement zones
    w_hi = df["high"].shift(1).rolling(36, min_periods=36).max()
    w_lo = df["low"].shift(1).rolling(36, min_periods=36).min()
    w_rng = w_hi - w_lo
    w_ok = w_rng >= (2.2 * df["atr"])

    min_low_2 = pd.concat([df["low"], df["low"].shift(1)], axis=1).min(axis=1)
    max_high_2 = pd.concat([df["high"], df["high"].shift(1)], axis=1).max(axis=1)

    # Zone A: Deep Fib 0.618 - 0.786
    z618_L = w_ok & (min_low_2 <= w_hi - 0.60 * w_rng) & (min_low_2 >= w_hi - 0.80 * w_rng) & (df["close"] > w_hi - 0.786 * w_rng)
    z618_S = w_ok & (max_high_2 >= w_lo + 0.60 * w_rng) & (max_high_2 <= w_lo + 0.80 * w_rng) & (df["close"] < w_lo + 0.786 * w_rng)

    # Zone B: Standard Fib 0.50 - 0.786
    z500_L = w_ok & (min_low_2 <= w_hi - 0.48 * w_rng) & (min_low_2 >= w_hi - 0.80 * w_rng) & (df["close"] > w_hi - 0.786 * w_rng)
    z500_S = w_ok & (max_high_2 >= w_lo + 0.48 * w_rng) & (max_high_2 <= w_lo + 0.80 * w_rng) & (df["close"] < w_lo + 0.786 * w_rng)

    df["fv2_618_L"] = common_ok & htf_bull & z618_L & rsi_bounce_L & bounce_L
    df["fv2_618_S"] = common_ok & htf_bear & z618_S & rsi_bounce_S & bounce_S
    df["fv2_618_Q"] = (adx_avg + body_ratio * 25.0).fillna(0.0)

    df["fv2_500_L"] = common_ok & htf_bull & z500_L & rsi_bounce_L & bounce_L
    df["fv2_500_S"] = common_ok & htf_bear & z500_S & rsi_bounce_S & bounce_S
    df["fv2_500_Q"] = (adx_avg + body_ratio * 25.0).fillna(0.0)

    cols = ["timestamp", "fv2_618_L", "fv2_618_S", "fv2_618_Q", "fv2_500_L", "fv2_500_S", "fv2_500_Q"]
    aligned = timeline_df.merge(df[cols], on="timestamp", how="left")

    out = dict(base_arrays)
    for pref in ("fv2_618", "fv2_500"):
        out[f"{pref}_L"] = aligned[f"{pref}_L"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        out[f"{pref}_S"] = aligned[f"{pref}_S"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        out[f"{pref}_Q"] = aligned[f"{pref}_Q"].fillna(0.0).to_numpy(dtype=np.float64)
    return out


def slice_sym_data(sym_data: Dict[str, Dict[str, np.ndarray]], mask: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    return {sym: {k: v[mask] for k, v in d.items()} for sym, d in sym_data.items()}


def run_blind_exam_for_config(
    cfg: ArenaConfig,
    symbols: List[str],
    train_data: Dict[str, Dict[str, np.ndarray]],
    train_tl: np.ndarray,
    test_data: Dict[str, Dict[str, np.ndarray]],
    test_tl: np.ndarray,
    top_k: int = 12,
    min_train_trades: int = 5,
) -> Dict[str, Any]:
    """
    100% Blind Walk-Forward:
      1. Score all 100 coins ONLY on `train_tl` (first 120 days). Pick Top-12 strictly on train PnL.
      2. Trade those exact 12 coins BLIND on `test_tl` (last 60 days) starting at $1,000.
    """
    train_end_ts = int(train_tl[-1]) + 1
    test_end_ts = int(test_tl[-1]) + 1

    # Stage 1: Train-only coin ranking (ZERO peeking at test_tl!)
    train_rows: List[Dict[str, Any]] = []
    for sym in symbols:
        res_tr = simulate_fixed_10x(
            cfg=cfg,
            symbols=[sym],
            sym_data={sym: train_data[sym]},
            timeline=train_tl,
            split_ts=train_end_ts,
            initial_deposit=1000.0,
            max_open=1,
            max_same_dir=1,
            daily_loss_limit_pct=10.5,
            cooldown_bars=3,
        )
        if res_tr["trades"] >= min_train_trades and res_tr["total_net_pct"] > 0.0:
            train_rows.append(
                {
                    "symbol": sym,
                    "train_trades": res_tr["trades"],
                    "train_wr": res_tr["win_rate_pct"],
                    "train_pnl_pct": res_tr["total_net_pct"],
                }
            )

    train_rows.sort(key=lambda r: (r["train_pnl_pct"], r["train_wr"]), reverse=True)
    selected = train_rows[:top_k]
    selected_syms = [r["symbol"] for r in selected]

    # Stage 2: Blind 60-day portfolio test starting at $1,000
    if not selected_syms:
        return {
            "label": cfg.label,
            "selected_count": 0,
            "selected_train": [],
            "blind_end_usd": 1000.0,
            "blind_net_usd": 0.0,
            "blind_net_pct": 0.0,
            "blind_trades": 0,
            "blind_trades_per_day": 0.0,
            "blind_wr_pct": 0.0,
            "blind_max_dd_usd": 0.0,
            "blind_max_dd_pct": 0.0,
            "top5_breakdown": [],
        }

    port = simulate_fixed_10x(
        cfg=cfg,
        symbols=selected_syms,
        sym_data=test_data,
        timeline=test_tl,
        split_ts=test_end_ts,
        initial_deposit=1000.0,
        max_open=3,
        max_same_dir=2,
        daily_loss_limit_pct=10.5,
        cooldown_bars=3,
    )

    # Per-coin blind performance inside the 60-day portfolio (for the Top-5 selected coins)
    trades_list = port["trades_list"]
    by_sym: Dict[str, List[Dict[str, Any]]] = {s: [] for s in selected_syms}
    for t in trades_list:
        by_sym.setdefault(t["symbol"], []).append(t)

    per_coin_blind: List[Dict[str, Any]] = []
    for r in selected:
        sym = r["symbol"]
        c_tr = by_sym.get(sym, [])
        n_c = len(c_tr)
        w_c = sum(1 for x in c_tr if x["net_usd"] > 0)
        pnl_usd_c = sum(x["net_usd"] for x in c_tr)
        eq_c = 100.0
        for x in c_tr:
            eq_c *= 1.0 + x["ret_pct"] / 100.0
        per_coin_blind.append(
            {
                "symbol": sym,
                "train_pnl_pct": r["train_pnl_pct"],
                "train_wr": r["train_wr"],
                "blind_trades": n_c,
                "blind_wr": round((w_c / n_c) * 100.0, 1) if n_c > 0 else 0.0,
                "blind_pnl_usd": round(pnl_usd_c, 2),
                "blind_pnl_pct": round(eq_c - 100.0, 2),
            }
        )

    return {
        "label": cfg.label,
        "selected_count": len(selected_syms),
        "qualified_train_count": len(train_rows),
        "selected_syms": selected_syms,
        "blind_end_usd": port["end_usd"],
        "blind_net_usd": port["total_net_usd"],
        "blind_net_pct": port["total_net_pct"],
        "blind_trades": port["trades"],
        "blind_trades_per_day": round(port["trades"] / 60.0, 2),
        "blind_wr_pct": port["win_rate_pct"],
        "blind_tp": port["tp_count"],
        "blind_be": port["be_count"],
        "blind_sl": port["sl_count"],
        "blind_fees_usd": port["fees_usd"],
        "blind_max_dd_usd": port["max_dd_usd"],
        "blind_max_dd_pct": port["max_dd_pct"],
        "top5_selected_performance": per_coin_blind[:5],
        "all_12_performance": per_coin_blind,
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    arena_meta = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]
    symbols_100 = arena_meta["symbols_100"]

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

    print(f"Preparing cached Top-100 signals (Train: 120d={len(train_tl)} bars | Blind Test: 60d={len(test_tl)} bars)...")
    sym_data: Dict[str, Dict[str, np.ndarray]] = {}
    active_syms: List[str] = []
    for sym in symbols_100:
        p = cache_dir / f"{sym}_15.csv.gz"
        if not p.exists():
            continue
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 400:
            continue
        base_arr = compute_all_strategy_signals(df_s, df_btc, timeline_df)
        full_arr = add_fibo_v2_signals(df_s, df_btc, timeline_df, base_arr)
        sym_data[sym] = full_arr
        active_syms.append(sym)

    train_data = slice_sym_data(sym_data, train_mask)
    test_data = slice_sym_data(sym_data, test_mask)

    # 3 Base participants
    base_configs = [
        ("1_fibo_trend", ArenaConfig("fibo36", "fibo_trend (волна 36)", 2.00, 3.5, False)),
        ("2_liquidity_sweep", ArenaConfig("ls", "liquidity_sweep (24 св.)", 1.25, 3.5, False)),
        ("3_mean_reversion", ArenaConfig("mr", "mean_reversion (BB/VWAP+RSI/MFI)", 2.00, 3.5, True)),
    ]

    results: Dict[str, Any] = {}
    for key, cfg in base_configs:
        results[key] = run_blind_exam_for_config(cfg, active_syms, train_data, train_tl, test_data, test_tl, top_k=12)

    # 4th participant: fibo_trend_v2 grid (0.618-0.786 vs 0.50-0.786 x TP 1.50/2.00 x NoBE/BE)
    fv2_variants: List[Dict[str, Any]] = []
    for z_key, z_name in (("fv2_618", "0.618-0.786"), ("fv2_500", "0.50-0.786")):
        for tp in (1.50, 2.00):
            for smart_be in (False, True):
                cfg = ArenaConfig(z_key, f"fibo_trend_v2 ({z_name}, 4H+ADX>=20+RSI)", tp, 3.5, smart_be)
                res = run_blind_exam_for_config(cfg, active_syms, train_data, train_tl, test_data, test_tl, top_k=12, min_train_trades=3)
                fv2_variants.append(res)

    fv2_variants.sort(key=lambda r: (r["blind_end_usd"], -r["blind_max_dd_pct"]), reverse=True)
    results["4_fibo_trend_v2_best"] = fv2_variants[0]
    results["4_fibo_trend_v2_all"] = fv2_variants

    out_path = Path("data/blind_walkforward_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 96)
    print("100% BLIND WALK-FORWARD EXAM (LAST 60 DAYS ONLY | START = $1,000 | ZERO PEEKING)")
    print("=" * 96)

    for key in ("1_fibo_trend", "2_liquidity_sweep", "3_mean_reversion", "4_fibo_trend_v2_best"):
        r = results[key]
        print(f"\n>>> {r['label']} (Selected Top-{r['selected_count']} from {r['qualified_train_count']} profitable in 4m)")
        print(f"  1) Deposit (60d BLIND): $1,000 -> ${r['blind_end_usd']:,.2f} ({r['blind_net_usd']:+,.2f}$ / {r['blind_net_pct']:+.2f}%)")
        print(f"  2) Trades & Blind WR: {r['blind_trades']} tr ({r['blind_trades_per_day']:.2f}/d) | WR = {r['blind_wr_pct']:.2f}% (TP={r['blind_tp']}, BE={r['blind_be']}, SL={r['blind_sl']})")
        print(f"  3) Max Drawdown: ${r['blind_max_dd_usd']:,.2f} ({r['blind_max_dd_pct']:.2f}%) | Fees: ${r['blind_fees_usd']:.2f}")
        top5_str = ", ".join(
            f"{c['symbol']} ({c['blind_trades']}tr, WR {c['blind_wr']:.0f}%, {c['blind_pnl_usd']:+.0f}$)"
            for c in r["top5_selected_performance"]
        )
        print(f"  4) Top-5 selected coins (blind 60d): {top5_str}")

    print("\n--- ALL 8 fibo_trend_v2 VARIANTS (0.618-0.786 vs 0.50-0.786) ---")
    for v in fv2_variants:
        print(
            f"  {v['label']:<68} | $1000->${v['blind_end_usd']:>7.2f} ({v['blind_net_pct']:>+6.2f}%) | "
            f"{v['blind_trades']:>3} tr | WR {v['blind_wr_pct']:>5.1f}% | DD {v['blind_max_dd_pct']:>5.1f}% | Sel={v['selected_count']}"
        )

    print(f"\nDone in {time.time() - t0:.2f}s -> saved to {out_path}")


if __name__ == "__main__":
    main()
