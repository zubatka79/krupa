"""Monte Carlo Time Machine: 10,000 simulations for 95th-percentile Max DD & short-stop survival."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _simulate_paths(returns_pct: np.ndarray, n_sims: int, rng: np.random.Generator) -> Dict[str, float]:
    """Run n_sims bootstrap Monte Carlo equity paths and return PnL & drawdown percentiles."""
    n_trades = len(returns_pct)
    if n_trades == 0:
        return {
            "trades": 0,
            "sims": n_sims,
            "median_net_pnl_pct": 0.0,
            "p05_net_pnl_pct": 0.0,
            "p95_net_pnl_pct": 0.0,
            "mean_max_dd_pct": 0.0,
            "median_max_dd_pct": 0.0,
            "p95_max_dd_pct": 0.0,
            "p99_max_dd_pct": 0.0,
            "prob_profit_pct": 0.0,
            "prob_dd_over_20_pct": 0.0,
            "prob_dd_over_30_pct": 0.0,
        }

    # Process in chunks to keep RAM footprint tiny even for 10,000 x 2,000 trades
    chunk_size = 1000
    final_pnls = np.empty(n_sims, dtype=np.float64)
    max_dds = np.empty(n_sims, dtype=np.float64)

    r_frac = returns_pct / 100.0

    for start in range(0, n_sims, chunk_size):
        end = min(start + chunk_size, n_sims)
        m = end - start
        idx = rng.integers(0, n_trades, size=(m, n_trades))
        sampled = r_frac[idx]  # shape (m, n_trades)
        growth = np.maximum(1.0 + sampled, 1e-6)

        # Prepend 1.0 starting equity so peak starts at 1.0
        eq = np.empty((m, n_trades + 1), dtype=np.float64)
        eq[:, 0] = 1.0
        np.cumprod(growth, axis=1, out=eq[:, 1:])

        peaks = np.maximum.accumulate(eq, axis=1)
        dds = (peaks - eq) / peaks * 100.0

        final_pnls[start:end] = (eq[:, -1] - 1.0) * 100.0
        max_dds[start:end] = np.max(dds, axis=1)

    return {
        "trades": int(n_trades),
        "sims": int(n_sims),
        "median_net_pnl_pct": round(float(np.median(final_pnls)), 2),
        "p05_net_pnl_pct": round(float(np.percentile(final_pnls, 5)), 2),
        "p95_net_pnl_pct": round(float(np.percentile(final_pnls, 95)), 2),
        "mean_max_dd_pct": round(float(np.mean(max_dds)), 2),
        "median_max_dd_pct": round(float(np.median(max_dds)), 2),
        "p95_max_dd_pct": round(float(np.percentile(max_dds, 95)), 2),
        "p99_max_dd_pct": round(float(np.percentile(max_dds, 99)), 2),
        "prob_profit_pct": round(float(np.mean(final_pnls > 0.0) * 100.0), 2),
        "prob_dd_over_20_pct": round(float(np.mean(max_dds >= 20.0) * 100.0), 2),
        "prob_dd_over_30_pct": round(float(np.mean(max_dds >= 30.0) * 100.0), 2),
    }


def _short_stop_survival_stress(
    trades: List[Dict[str, Any]],
    n_sims: int,
    rng: np.random.Generator,
    wick_out_prob: float = 0.07,
    slippage_extra_pct: float = 0.04,
) -> Dict[str, Any]:
    """
    Short-Stop Survival Stress Test (1.2%-1.6% stops):
    In each of n_sims simulations:
    - Each winning/BE trade has `wick_out_prob` (7%) chance of getting wicked out at SL
      due to spread spikes / stop-hunts before reaching TP1.
    - Every SL/BE market stop experiences additional random slippage up to `slippage_extra_pct`
      (scaled by leverage * margin_pct -> equity impact).
    """
    n_trades = len(trades)
    if n_trades == 0:
        return {
            "wick_out_prob_pct": round(wick_out_prob * 100.0, 1),
            "extra_stop_slippage_pct": slippage_extra_pct,
            "median_net_pnl_pct": 0.0,
            "p95_max_dd_pct": 0.0,
            "prob_profit_pct": 0.0,
            "survival_rate_dd_under_25_pct": 0.0,
        }

    base_ret = np.array([float(t["equity_return_pct"]) for t in trades], dtype=np.float64)
    is_sl = np.array([t["exit_reason"] == "SL" for t in trades], dtype=bool)
    sl_losses = base_ret[is_sl]
    typical_sl_ret = float(np.mean(sl_losses)) if len(sl_losses) > 0 else -2.25

    # Slippage on tight market stops: roughly leverage * margin_pct(0.15) * extra_slippage_pct
    lev_arr = np.array([float(t.get("leverage", 8)) for t in trades], dtype=np.float64)
    slip_eq_impact = 0.15 * lev_arr * slippage_extra_pct  # in equity %

    chunk_size = 1000
    final_pnls = np.empty(n_sims, dtype=np.float64)
    max_dds = np.empty(n_sims, dtype=np.float64)

    for start in range(0, n_sims, chunk_size):
        end = min(start + chunk_size, n_sims)
        m = end - start
        idx = rng.integers(0, n_trades, size=(m, n_trades))
        sim_ret = base_ret[idx].copy()
        sim_sl_mask = is_sl[idx]
        sim_slip = slip_eq_impact[idx]

        # Random stop-hunt wick-outs flipping winners into SL losses
        wick_flip = (~sim_sl_mask) & (rng.random((m, n_trades)) < wick_out_prob)
        sim_ret[wick_flip] = typical_sl_ret

        # Extra slippage on all market stop-outs (original SL + wicked-out stops)
        all_stops = sim_sl_mask | wick_flip
        rand_slip = rng.uniform(0.0, 1.0, size=(m, n_trades)) * sim_slip
        sim_ret[all_stops] -= rand_slip[all_stops]

        growth = np.maximum(1.0 + sim_ret / 100.0, 1e-6)
        eq = np.empty((m, n_trades + 1), dtype=np.float64)
        eq[:, 0] = 1.0
        np.cumprod(growth, axis=1, out=eq[:, 1:])

        peaks = np.maximum.accumulate(eq, axis=1)
        dds = (peaks - eq) / peaks * 100.0

        final_pnls[start:end] = (eq[:, -1] - 1.0) * 100.0
        max_dds[start:end] = np.max(dds, axis=1)

    return {
        "wick_out_prob_pct": round(wick_out_prob * 100.0, 1),
        "extra_stop_slippage_pct": slippage_extra_pct,
        "typical_sl_return_pct": round(typical_sl_ret, 2),
        "median_net_pnl_pct": round(float(np.median(final_pnls)), 2),
        "p05_net_pnl_pct": round(float(np.percentile(final_pnls, 5)), 2),
        "p95_max_dd_pct": round(float(np.percentile(max_dds, 95)), 2),
        "prob_profit_pct": round(float(np.mean(final_pnls > 0.0) * 100.0), 2),
        "survival_rate_dd_under_25_pct": round(float(np.mean(max_dds < 25.0) * 100.0), 2),
    }


def run_time_machine(
    input_path: str | Path = "data/backtest_results.json",
    output_path: str | Path = "data/time_machine_results.json",
    n_sims: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Backtest results not found: {in_p}. Run bot/backtest.py first.")

    report = json.loads(in_p.read_text(encoding="utf-8"))
    trades: List[Dict[str, Any]] = report.get("trades", [])

    is_trades = [t for t in trades if t.get("period") == "IS"]
    oos_trades = [t for t in trades if t.get("period") == "OOS"]

    rng = np.random.default_rng(seed)

    mc_is = _simulate_paths(np.array([float(t["equity_return_pct"]) for t in is_trades], dtype=np.float64), n_sims, rng)
    mc_oos = _simulate_paths(np.array([float(t["equity_return_pct"]) for t in oos_trades], dtype=np.float64), n_sims, rng)
    mc_total = _simulate_paths(np.array([float(t["equity_return_pct"]) for t in trades], dtype=np.float64), n_sims, rng)

    stress_total = _short_stop_survival_stress(trades, n_sims, rng, wick_out_prob=0.07, slippage_extra_pct=0.04)

    result = {
        "sims": n_sims,
        "seed": seed,
        "in_sample": mc_is,
        "out_of_sample": mc_oos,
        "total": mc_total,
        "short_stop_survival_stress": stress_total,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _print_time_machine_summary(result)
    return result


def _print_time_machine_summary(res: Dict[str, Any]) -> None:
    mc_is = res["in_sample"]
    mc_oos = res["out_of_sample"]
    mc_tot = res["total"]
    st = res["short_stop_survival_stress"]

    print("\n" + "=" * 92)
    print(f"TIME MACHINE MONTE CARLO ({res['sims']:,} SIMULATIONS)")
    print("=" * 92)
    print(f"{'Metric':<32} | {'In-Sample (120d)':>17} | {'Out-of-Sample (60d)':>19} | {'Total (180d)':>15}")
    print("-" * 92)
    rows = [
        ("Median Net PnL (%)", f"{mc_is['median_net_pnl_pct']:+.2f}%", f"{mc_oos['median_net_pnl_pct']:+.2f}%", f"{mc_tot['median_net_pnl_pct']:+.2f}%"),
        ("5th Pct Net PnL (%)", f"{mc_is['p05_net_pnl_pct']:+.2f}%", f"{mc_oos['p05_net_pnl_pct']:+.2f}%", f"{mc_tot['p05_net_pnl_pct']:+.2f}%"),
        ("95th Pct Net PnL (%)", f"{mc_is['p95_net_pnl_pct']:+.2f}%", f"{mc_oos['p95_net_pnl_pct']:+.2f}%", f"{mc_tot['p95_net_pnl_pct']:+.2f}%"),
        ("Median Max Drawdown (%)", f"{mc_is['median_max_dd_pct']:.2f}%", f"{mc_oos['median_max_dd_pct']:.2f}%", f"{mc_tot['median_max_dd_pct']:.2f}%"),
        ("95th Pct Max Drawdown (%)", f"{mc_is['p95_max_dd_pct']:.2f}%", f"{mc_oos['p95_max_dd_pct']:.2f}%", f"{mc_tot['p95_max_dd_pct']:.2f}%"),
        ("99th Pct Max Drawdown (%)", f"{mc_is['p99_max_dd_pct']:.2f}%", f"{mc_oos['p99_max_dd_pct']:.2f}%", f"{mc_tot['p99_max_dd_pct']:.2f}%"),
        ("Prob. of Profit (%)", f"{mc_is['prob_profit_pct']:.2f}%", f"{mc_oos['prob_profit_pct']:.2f}%", f"{mc_tot['prob_profit_pct']:.2f}%"),
        ("Prob. Max DD >= 20% (%)", f"{mc_is['prob_dd_over_20_pct']:.2f}%", f"{mc_oos['prob_dd_over_20_pct']:.2f}%", f"{mc_tot['prob_dd_over_20_pct']:.2f}%"),
    ]
    for label, v1, v2, v3 in rows:
        print(f"{label:<32} | {v1:>17} | {v2:>19} | {v3:>15}")
    print("-" * 92)
    print(
        f"SHORT-STOP SURVIVAL STRESS TEST (Wick-out={st['wick_out_prob_pct']}%, Extra Stop Slippage={st['extra_stop_slippage_pct']}%):\n"
        f"  Median Net PnL: {st['median_net_pnl_pct']:+.2f}% | 5th Pct PnL: {st['p05_net_pnl_pct']:+.2f}% | "
        f"95th Pct Max DD: {st['p95_max_dd_pct']:.2f}% | Prob. Profit: {st['prob_profit_pct']:.2f}% | "
        f"Survival (Max DD < 25%): {st['survival_rate_dd_under_25_pct']:.2f}%"
    )
    print("=" * 92 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="10,000-Run Monte Carlo Stress Tester")
    parser.add_argument("--input", type=str, default="data/backtest_results.json", help="Backtest JSON report path")
    parser.add_argument("--output", type=str, default="data/time_machine_results.json", help="Monte Carlo output path")
    parser.add_argument("--sims", type=int, default=10_000, help="Number of Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_time_machine(
        input_path=args.input,
        output_path=args.output,
        n_sims=args.sims,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
