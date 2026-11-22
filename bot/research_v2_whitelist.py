"""Offline 2-Step Whitelist Stability Filter & TP/SL/BE Grid Research on cached data."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.backtest_v2 import (
    GridParams,
    compute_metrics_v2,
    simulate_grid_combo,
)
from bot.bybit_client import BybitClient
from bot.strategy_v2 import enrich_v2_dataframe

PREV_TOP_10 = [
    "MMTUSDT",
    "XRPUSDT",
    "HYPEUSDT",
    "XMRUSDT",
    "SUIUSDT",
    "PIEVERSEUSDT",
    "SOLUSDT",
    "ENAUSDT",
    "NEARUSDT",
    "ETHUSDT",
]


def load_cached_15m(symbol: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{symbol}_15.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Cache file missing for {symbol}: {path}")
    df = pd.read_csv(path, compression="gzip")
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = df[col].astype("float64")
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def run_two_step_research(
    days: int = 180,
    is_days: int = 120,
    oos_days: int = 60,
    n_sims: int = 10_000,
) -> Dict[str, Any]:
    t0 = time.time()
    cache_dir = Path("data/cache")
    prev_report = json.loads(Path("data/backtest_v2_results.json").read_text(encoding="utf-8"))
    symbols_25 = [u["symbol"] for u in prev_report["meta"]["universe_25"]]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    sim_end_ms = int(df_btc["timestamp"].iloc[-1])
    sim_start_ms = sim_end_ms - days * 86_400_000
    split_ms = sim_start_ms + is_days * 86_400_000

    btc_sim_ts = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]["timestamp"].to_numpy(dtype=np.int64)
    timeline = np.sort(np.unique(btc_sim_ts))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    # Precompute trend_pullback_v2 arrays for all 25 pure crypto coins strictly from cache
    sym_data: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in symbols_25:
        df_15m = load_cached_15m(sym, cache_dir)
        df_1h = BybitClient.resample_klines(df_15m, "60")
        df_4h = BybitClient.resample_klines(df_15m, "240")
        enriched = enrich_v2_dataframe(
            df_15m=df_15m,
            df_btc_15m=df_btc,
            df_1h=df_1h,
            df_4h=df_4h,
            btc_shock_pause_pct=0.8,
        )
        aligned = timeline_df.merge(
            enriched[["timestamp", "high", "low", "close", "sig_long", "sig_short", "quality_long", "quality_short"]],
            on="timestamp",
            how="left",
        )
        sym_data[sym] = {
            "high": aligned["high"].to_numpy(dtype=np.float64),
            "low": aligned["low"].to_numpy(dtype=np.float64),
            "close": aligned["close"].to_numpy(dtype=np.float64),
            "sig_long": aligned["sig_long"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "sig_short": aligned["sig_short"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "q_long": aligned["quality_long"].fillna(0.0).to_numpy(dtype=np.float64),
            "q_short": aligned["quality_short"].fillna(0.0).to_numpy(dtype=np.float64),
        }

    # ==================================================================
    # STEP 1: Per-coin standalone evaluation (TP 1.25% / SL 3.5% / MAKER / NoBE)
    #         Keep ONLY coins with Net PnL > 0 in BOTH IS (120d) AND OOS (60d)
    # ==================================================================
    base_params = GridParams(tp_pct=1.25, sl_pct=3.5, entry_type="maker", smart_be=False)
    step1_rows: List[Dict[str, Any]] = []

    for sym in symbols_25:
        trades = simulate_grid_combo(
            params=base_params,
            timeline=timeline,
            split_ts=split_ms,
            sym_data={sym: sym_data[sym]},
            spec_by_sym={},
            initial_equity=1000.0,
            maker_fee_pct=0.02,
            taker_fee_pct=0.055,
            max_risk_pct=2.5,
            max_open_positions=1,
            max_same_direction=1,
            daily_loss_limit_pct=6.0,
            cooldown_bars=3,
        )
        is_tr = [t for t in trades if t.period == "IS"]
        oos_tr = [t for t in trades if t.period == "OOS"]
        m_is = compute_metrics_v2(is_tr, is_days, n_sims=100, seed=1)
        m_oos = compute_metrics_v2(oos_tr, oos_days, n_sims=100, seed=2)
        m_tot = compute_metrics_v2(trades, days, n_sims=100, seed=3)

        passed = (m_is["net_pnl_pct"] > 0.0) and (m_oos["net_pnl_pct"] > 0.0)
        step1_rows.append(
            {
                "symbol": sym,
                "passed": passed,
                "in_prev_top10": sym in PREV_TOP_10,
                "is_trades": m_is["trades"],
                "is_wr": m_is["win_rate_pct"],
                "is_pnl": m_is["net_pnl_pct"],
                "oos_trades": m_oos["trades"],
                "oos_wr": m_oos["win_rate_pct"],
                "oos_pnl": m_oos["net_pnl_pct"],
                "tot_trades": m_tot["trades"],
                "tot_wr": m_tot["win_rate_pct"],
                "tot_pnl": m_tot["net_pnl_pct"],
            }
        )

    passed_rows = sorted([r for r in step1_rows if r["passed"]], key=lambda r: r["tot_pnl"], reverse=True)
    clean_whitelist = [r["symbol"] for r in passed_rows]
    dropped_prev_top10 = [r for r in step1_rows if r["in_prev_top10"] and not r["passed"]]

    print("\n" + "=" * 96)
    print(f"STEP 1: CLEAN WHITELIST ({len(clean_whitelist)} / 25 COINS PASSED BOTH IS > 0 AND OOS > 0)")
    print("=" * 96)
    for r in passed_rows:
        print(
            f"  [PASS] {r['symbol']:<12} | IS(4m): {r['is_trades']:>3} tr, WR {r['is_wr']:>5.1f}%, PnL {r['is_pnl']:>+6.2f}% | "
            f"OOS(2m): {r['oos_trades']:>3} tr, WR {r['oos_wr']:>5.1f}%, PnL {r['oos_pnl']:>+6.2f}% | TOT: {r['tot_pnl']:>+6.2f}%"
        )

    print("\nDROPPED FROM PREVIOUS TOP-10:")
    for r in dropped_prev_top10:
        reason = []
        if r["is_pnl"] <= 0:
            reason.append(f"IS(4m)={r['is_pnl']:+.2f}%")
        if r["oos_pnl"] <= 0:
            reason.append(f"OOS(2m)={r['oos_pnl']:+.2f}%")
        print(
            f"  [DROP] {r['symbol']:<12} | IS(4m): {r['is_trades']:>3} tr, WR {r['is_wr']:>5.1f}%, PnL {r['is_pnl']:>+6.2f}% | "
            f"OOS(2m): {r['oos_trades']:>3} tr, WR {r['oos_wr']:>5.1f}%, PnL {r['oos_pnl']:>+6.2f}% | Reason: {', '.join(reason)}"
        )

    print("\nALL 25 COINS DIAGNOSTIC:")
    for r in sorted(step1_rows, key=lambda x: (x["passed"], x["tot_pnl"]), reverse=True):
        tag = "PASS" if r["passed"] else "FAIL"
        print(
            f"  [{tag}] {r['symbol']:<13} | IS: {r['is_trades']:>2} tr WR {r['is_wr']:>5.1f}% {r['is_pnl']:>+6.2f}% | "
            f"OOS: {r['oos_trades']:>2} tr WR {r['oos_wr']:>5.1f}% {r['oos_pnl']:>+6.2f}% | TOT: {r['tot_pnl']:>+6.2f}%"
        )

    # ==================================================================
    # STEP 2: Grid search on the clean whitelist (TP x SL x NoBE / BE@65%)
    # ==================================================================
    wl_sym_data = {sym: sym_data[sym] for sym in clean_whitelist}
    grid: List[GridParams] = []
    for tp in (1.25, 1.50, 1.75, 2.00):
        for sl in (2.8, 3.2, 3.5, 4.0):
            for smart_be in (False, True):
                be_trig = round(tp * 0.65, 4)
                grid.append(
                    GridParams(
                        tp_pct=tp,
                        sl_pct=sl,
                        entry_type="maker",
                        smart_be=smart_be,
                        be_trigger_pct=be_trig,
                        be_offset_pct=0.15,
                    )
                )

    grid_results: List[Dict[str, Any]] = []
    for idx_g, gp in enumerate(grid):
        trades = simulate_grid_combo(
            params=gp,
            timeline=timeline,
            split_ts=split_ms,
            sym_data=wl_sym_data,
            spec_by_sym={},
            initial_equity=1000.0,
            maker_fee_pct=0.02,
            taker_fee_pct=0.055,
            max_risk_pct=2.5,
            max_open_positions=3,
            max_same_direction=2,
            daily_loss_limit_pct=6.0,
            cooldown_bars=3,
        )
        is_tr = [t for t in trades if t.period == "IS"]
        oos_tr = [t for t in trades if t.period == "OOS"]

        m_is = compute_metrics_v2(is_tr, is_days, n_sims=n_sims, seed=1000 + idx_g)
        m_oos = compute_metrics_v2(oos_tr, oos_days, n_sims=n_sims, seed=2000 + idx_g)
        m_tot = compute_metrics_v2(trades, days, n_sims=n_sims, seed=3000 + idx_g)

        be_label = f"BE(+0.15% @ 65%={gp.be_trigger_pct:.2f}%)" if gp.smart_be else "NoBE"
        label = f"TP={gp.tp_pct:.2f}% | SL={gp.sl_pct:.1f}% | {be_label}"
        grid_results.append(
            {
                "label": label,
                "params": asdict(gp),
                "in_sample": m_is,
                "out_of_sample": m_oos,
                "total": m_tot,
            }
        )

    # Rank by stability (both IS>0 & OOS>0), total net PnL, OOS net PnL, and lower MC95 DD
    grid_results.sort(
        key=lambda x: (
            1.0 if (x["in_sample"]["net_pnl_pct"] > 0 and x["out_of_sample"]["net_pnl_pct"] > 0) else 0.0,
            x["total"]["net_pnl_pct"],
            x["out_of_sample"]["net_pnl_pct"],
            -x["total"]["mc_p95_dd_pct"],
        ),
        reverse=True,
    )

    top_5 = grid_results[:5]
    print("\n" + "=" * 96)
    print(f"STEP 2: TOP-5 PARAMETER COMBINATIONS ON CLEAN WHITELIST ({len(clean_whitelist)} coins, 10,000 MC sims)")
    print("=" * 96)
    for rank, item in enumerate(top_5, 1):
        mi, mo, mt = item["in_sample"], item["out_of_sample"], item["total"]
        print(
            f"\n#{rank}: {item['label']}\n"
            f"  Trades (6m): {mt['trades']} ({mt['trades_per_day']:.2f}/d) | Overall WR: {mt['win_rate_pct']:.2f}% "
            f"(TP={mt['tp_count']}, BE={mt['be_count']}, SL={mt['sl_count']})\n"
            f"  Net PnL (after fees): IS(4m) {mi['net_pnl_pct']:+.2f}% | OOS(2m) {mo['net_pnl_pct']:+.2f}% | Total(6m) {mt['net_pnl_pct']:+.2f}% (PF={mt['profit_factor']:.2f}, Fees={mt['fees_pct']:.2f}%)\n"
            f"  Max DD: Real {mt['max_dd_pct']:.2f}% | Monte Carlo 95% {mt['mc_p95_dd_pct']:.2f}% (Prob Profit: {mt['mc_prob_profit_pct']:.1f}%)"
        )

    out = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_utc": datetime.fromtimestamp(sim_start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "split_utc": datetime.fromtimestamp(split_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "end_utc": datetime.fromtimestamp(sim_end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "step1_clean_whitelist": clean_whitelist,
        "step1_passed": passed_rows,
        "step1_dropped_prev_top10": dropped_prev_top10,
        "step1_all_25": step1_rows,
        "step2_top_5": top_5,
        "step2_all_32": grid_results,
    }
    Path("data/research_v2_whitelist.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/research_v2_whitelist.json")
    return out


if __name__ == "__main__":
    run_two_step_research()
