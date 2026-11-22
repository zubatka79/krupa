"""POLYGRAPH 6-MONTH EXAM (`bot/basket_ada_6m.py`):
Calm 2-3 Coin Basket around `ADAUSDT` under the exact locked 15M Fib 0.50 setup (15% x 10x isolated).

HONESTY PROTOCOL:
  1. Companion selection and basket rules were locked STRICTLY on Days 1-120 (M1-M4)
     with Days 121-180 physically excluded.
  2. Locked Primary Basket (from Days 1-120):
     `ADAUSDT + TRUMPUSDT` (2 coins, max_open=2, max_same_dir=2) — matched 15M ATR
     (0.558% vs 0.581%), low 1H correlation (0.54), both 4/4 green standalone, lowest DD (14.8%).
  3. Reference Baskets logged alongside (not cherry-picked):
     - 3-coin 4/4 club: `ADAUSDT + 1000PEPEUSDT + TRUMPUSDT` (max_open=3, max_same_dir=2)
     - Blue-chip 2-coin: `ADAUSDT + XRPUSDT` (max_open=2, max_same_dir=2)
  4. Months 5 & 6 (Days 121-180) are executed ONCE blindly with zero tweaks.

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
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

from bot.blind_walkforward import load_cached_15m
from bot.single_coin_fibo_6m import prep_single_coin_180d


def align_coin_for_basket(df_prep: pd.DataFrame, tl_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    al = tl_df.merge(
        df_prep[
            [
                "timestamp", "high", "low", "close", "whi_20", "wlo_20",
                "iup_20", "idn_20", "atr", "atr_pct", "vol_cd",
                "h1_bull", "h1_bear", "h1_adx",
            ]
        ],
        on="timestamp",
        how="left",
    )
    return {
        "high": al["high"].to_numpy(dtype=np.float64),
        "low": al["low"].to_numpy(dtype=np.float64),
        "close": al["close"].to_numpy(dtype=np.float64),
        "whi": al["whi_20"].to_numpy(dtype=np.float64),
        "wlo": al["wlo_20"].to_numpy(dtype=np.float64),
        "iup": al["iup_20"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "idn": al["idn_20"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "atr": al["atr"].fillna(0.0).to_numpy(dtype=np.float64),
        "atr_pct": al["atr_pct"].fillna(99.0).to_numpy(dtype=np.float64),
        "vol_cd": al["vol_cd"].astype("boolean").fillna(True).to_numpy(dtype=bool),
        "h1_bull": al["h1_bull"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "h1_bear": al["h1_bear"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "h1_adx": al["h1_adx"].fillna(0.0).to_numpy(dtype=np.float64),
    }


def simulate_6m_basket(
    name: str,
    is_locked_primary: bool,
    syms: List[str],
    bundles: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    month_bounds: List[Tuple[str, int, int]],
    blind_split_ms: int,
    max_open: int,
    max_same_dir: int,
    margin_pct: float = 15.0,
    max_daily_stops: int = 4,
) -> Dict[str, Any]:
    mk, tk, slip = 0.0002, 0.00055, 0.0004
    tp_f, sl_f, be_off_f = 0.0120, 0.0250, 0.0020
    be_tr_f = tp_f * 0.65

    equity, peak, max_dd = 1000.0, 1000.0, 0.0
    blind_peak, blind_max_dd = None, 0.0
    open_pos, pending, cd_until, trades = [], {}, {}, []
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end = {}, {}

    def _close(p, trig_p, reason, is_mk, bar_i, ts):
        nonlocal equity, peak, max_dd, blind_peak, blind_max_dd, day_sl, halted
        is_l = p["dir"] == "LONG"
        ex_p = trig_p if is_mk else (trig_p * (1 - slip) if is_l else trig_p * (1 + slip))
        gross = (ex_p - p["ep"]) * p["qty"] if is_l else (p["ep"] - ex_p) * p["qty"]
        net = max(-p["margin"], gross - p["efee"] - abs(p["qty"] * ex_p) * (mk if is_mk else tk))
        equity = max(1.0, equity + net)
        if reason == "SL":
            day_sl += 1
            if day_sl >= max_daily_stops:
                halted = True
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
        if ts >= blind_split_ms:
            if blind_peak is None or equity > blind_peak:
                blind_peak = equity
            bdd = (blind_peak - equity) / blind_peak * 100.0
            if bdd > blind_max_dd:
                blind_max_dd = bdd
        trades.append({"sym": p["sym"], "reason": reason, "net": float(net), "is_blind": bool(ts >= blind_split_ms)})
        cd_until[p["sym"]] = bar_i + 1

    for i in range(len(timeline)):
        ts = int(timeline[i])
        if ts >= blind_split_ms and blind_peak is None:
            blind_peak = equity
        for mn, ms, me in month_bounds:
            if mn not in m_start and ts >= ms:
                m_start[mn] = equity
            if ts <= me:
                m_end[mn] = equity
        dk = ts // 86_400_000
        if dk != cur_day:
            cur_day, day_sl, halted = dk, 0, False

        if open_pos:
            still = []
            for p in open_pos:
                sd = bundles[p["sym"]]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    still.append(p)
                    continue
                is_l = p["dir"] == "LONG"
                if p["be_on"]:
                    if (lo <= p["be_p"]) if is_l else (hi >= p["be_p"]):
                        _close(p, p["be_p"], "BE", False, i, ts)
                        continue
                    if (hi >= p["tp_p"]) if is_l else (lo <= p["tp_p"]):
                        _close(p, p["tp_p"], "TP", True, i, ts)
                        continue
                else:
                    if (lo <= p["sl_p"]) if is_l else (hi >= p["sl_p"]):
                        _close(p, p["sl_p"], "SL", False, i, ts)
                        continue
                    if (hi >= p["tp_p"]) if is_l else (lo <= p["tp_p"]):
                        _close(p, p["tp_p"], "TP", True, i, ts)
                        continue
                    if (hi >= p["be_tr"]) if is_l else (lo <= p["be_tr"]):
                        p["be_on"] = True
                still.append(p)
            open_pos = still

        if pending:
            new_pend = {}
            for sym, o in pending.items():
                if i > o["exp"] or halted or i < cd_until.get(sym, -1) or any(p["sym"] == sym for p in open_pos):
                    continue
                sd = bundles[sym]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    if i < o["exp"]:
                        new_pend[sym] = o
                    continue
                is_l = o["dir"] == "LONG"
                filled = (lo < o["lp"]) if is_l else (hi > o["lp"])
                if not filled:
                    if i < o["exp"]:
                        new_pend[sym] = o
                    continue
                if len(open_pos) >= max_open or sum(1 for p in open_pos if p["dir"] == o["dir"]) >= max_same_dir:
                    continue
                margin = equity * (margin_pct / 100.0)
                if sum(p["margin"] for p in open_pos) + margin > equity * 0.95:
                    continue
                notional = margin * 10.0
                ep = o["lp"]
                pos = {
                    "sym": sym, "dir": o["dir"], "ep": ep, "qty": notional / ep, "margin": margin,
                    "efee": notional * mk, "be_on": False,
                    "sl_p": ep * (1 - sl_f) if is_l else ep * (1 + sl_f),
                    "tp_p": ep * (1 + tp_f) if is_l else ep * (1 - tp_f),
                    "be_tr": ep * (1 + be_tr_f) if is_l else ep * (1 - be_tr_f),
                    "be_p": ep * (1 + be_off_f) if is_l else ep * (1 - be_off_f),
                }
                if (lo <= pos["sl_p"]) if is_l else (hi >= pos["sl_p"]):
                    _close(pos, pos["sl_p"], "SL", False, i, ts)
                else:
                    open_pos.append(pos)
            pending = new_pend

        if halted or len(open_pos) >= max_open:
            continue
        busy = {p["sym"] for p in open_pos} | set(pending.keys())
        for sym in syms:
            if sym in busy or i < cd_until.get(sym, -1):
                continue
            sd = bundles[sym]
            if sd["vol_cd"][i] or sd["h1_adx"][i] < 18.0 or sd["atr_pct"][i] < 0.10 or sd["atr_pct"][i] > 1.80:
                continue
            rng = sd["whi"][i] - sd["wlo"][i]
            if not np.isfinite(rng) or rng < 2.2 * sd["atr"][i]:
                continue
            cl = sd["close"][i]
            if sd["h1_bull"][i] and sd["iup"][i]:
                if sum(1 for p in open_pos if p["dir"] == "LONG") >= max_same_dir:
                    continue
                lp = sd["whi"][i] - 0.50 * rng
                if lp < cl * 0.9992:
                    pending[sym] = {"dir": "LONG", "lp": float(lp), "exp": i + 4}
            elif sd["h1_bear"][i] and sd["idn"][i]:
                if sum(1 for p in open_pos if p["dir"] == "SHORT") >= max_same_dir:
                    continue
                lp = sd["wlo"][i] + 0.50 * rng
                if lp > cl * 1.0008:
                    pending[sym] = {"dir": "SHORT", "lp": float(lp), "exp": i + 4}

    if open_pos:
        last_ts = int(timeline[-1])
        for p in open_pos:
            cl = bundles[p["sym"]]["close"][-1]
            _close(p, float(cl) if np.isfinite(cl) else p["ep"], "EOD", False, len(timeline) - 1, last_ts)
        for mn, _, me in month_bounds:
            if last_ts <= me:
                m_end[mn] = equity

    mp = {mn: round((m_end.get(mn, 1000.0) - m_start.get(mn, 1000.0)) / max(m_start.get(mn, 1000.0), 1.0) * 100.0, 2) for mn, _, _ in month_bounds}
    n = len(trades)
    sl_n = sum(1 for t in trades if t["reason"] == "SL")
    wr = round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2)
    bl_tr = [t for t in trades if t["is_blind"]]
    bl_wr = round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2)
    bl_sl = sum(1 for t in bl_tr if t["reason"] == "SL")
    gp = sum(t["net"] for t in trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in trades if t["net"] < 0))
    by_sym = {}
    for s in syms:
        st = [t for t in trades if t["sym"] == s]
        sb = [t for t in bl_tr if t["sym"] == s]
        by_sym[s] = {
            "trades": len(st),
            "wr": round(sum(1 for t in st if t["net"] > 0) / max(len(st), 1) * 100.0, 1),
            "net_usd": round(sum(t["net"] for t in st), 2),
            "blind_net_usd": round(sum(t["net"] for t in sb), 2),
            "blind_wr": round(sum(1 for t in sb if t["net"] > 0) / max(len(sb), 1) * 100.0, 1),
        }

    return {
        "name": name,
        "is_locked_primary": is_locked_primary,
        "symbols": syms,
        "max_open": max_open,
        "max_same_dir": max_same_dir,
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n,
        "trades_per_day": round(n / 180.0, 2),
        "wr_pct": wr,
        "sl_count": sl_n,
        "pf": round(gp / max(gl, 1e-9), 2),
        "max_dd_pct": round(max_dd, 2),
        "blind_trades": len(bl_tr),
        "blind_wr_pct": bl_wr,
        "blind_sl": bl_sl,
        "blind_max_dd_pct": round(blind_max_dd, 2),
        "by_symbol": by_sym,
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    day_ms = 86_400_000
    sim_end = int(df_btc["timestamp"].iloc[-1])
    sim_start = sim_end - 180 * day_ms
    blind_split = sim_start + 120 * day_ms

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start) & (df_btc["timestamp"] <= sim_end)].sort_values("timestamp")
    tl = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    tl_df = pd.DataFrame({"timestamp": tl})

    m_bounds = [
        ("M1", sim_start + 0 * day_ms, sim_start + 30 * day_ms - 1),
        ("M2", sim_start + 30 * day_ms, sim_start + 60 * day_ms - 1),
        ("M3", sim_start + 60 * day_ms, sim_start + 90 * day_ms - 1),
        ("M4", sim_start + 90 * day_ms, sim_start + 120 * day_ms - 1),
        ("M5", sim_start + 120 * day_ms, sim_start + 150 * day_ms - 1),
        ("M6", sim_start + 150 * day_ms, sim_end),
    ]

    all_syms = ["ADAUSDT", "TRUMPUSDT", "1000PEPEUSDT", "XRPUSDT"]
    bundles = {s: align_coin_for_basket(prep_single_coin_180d(load_cached_15m(s, cache_dir), waves=(20,)), tl_df) for s in all_syms}

    configs = [
        ("LOCKED PRIMARY (2-coin: ADAUSDT + TRUMPUSDT, max_open=2, max_same_dir=2)", True, ["ADAUSDT", "TRUMPUSDT"], 2, 2),
        ("REF-3COIN (3-coin 4/4 club: ADAUSDT + 1000PEPEUSDT + TRUMPUSDT, max_open=3, max_same_dir=2)", False, ["ADAUSDT", "1000PEPEUSDT", "TRUMPUSDT"], 3, 2),
        ("REF-BLUECHIP (2-coin: ADAUSDT + XRPUSDT, max_open=2, max_same_dir=2)", False, ["ADAUSDT", "XRPUSDT"], 2, 2),
    ]

    results = []
    print("\n" + "=" * 116)
    print("POLYGRAPH 6-MONTH EXAM: ADAUSDT FIB 0.50 BASKET (15% x 10x ISOLATED | BYBIT FEES + 0.04% SLIPPAGE)")
    print("=" * 116)
    for name, prim, syms, mo, msd in configs:
        r = simulate_6m_basket(name, prim, syms, bundles, tl, m_bounds, blind_split, max_open=mo, max_same_dir=msd)
        results.append(r)
        mp = r["monthly_pct"]
        print(f"\n>>> {r['name']}")
        print(
            f"  M1:{mp['M1']:>+6.1f}% / M2:{mp['M2']:>+6.1f}% / M3:{mp['M3']:>+6.1f}% / M4:{mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5:{mp['M5']:>+6.1f}% / M6:{mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого: $1 000 -> ${r['end_usd']:,.2f} | Сделок: {r['trades']} ({r['trades_per_day']:.2f}/день) | "
            f"WR: {r['wr_pct']:.1f}% (Blind WR: {r['blind_wr_pct']:.1f}%) | Стопов: {r['sl_count']} (Blind SL: {r['blind_sl']}) | "
            f"PF: {r['pf']:.2f} | Макс. DD: {r['max_dd_pct']:.1f}% (Blind DD: {r['blind_max_dd_pct']:.1f}%)"
        )
        for s, st in r["by_symbol"].items():
            print(f"    {s:<13}: N={st['trades']:>3}, WR={st['wr']:>4.1f}%, Net=${st['net_usd']:>+8.2f} | Blind WR={st['blind_wr']:>4.1f}%, Blind Net=${st['blind_net_usd']:>+8.2f}")

    out = {
        "results": results,
        "benchmarks": {
            "ada_only": {"end_usd": 3241.07, "wr_pct": 84.3, "max_dd_pct": 13.7, "blind_dd_pct": 12.9},
            "v4": {"end_usd": 4344.67, "wr_pct": 84.5, "max_dd_pct": 20.4},
        },
    }
    Path("data/basket_ada_6m_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/basket_ada_6m_results.json")


if __name__ == "__main__":
    main()
