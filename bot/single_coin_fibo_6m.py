"""POLYGRAPH 6-MONTH EXAM (`bot/single_coin_fibo_6m.py`):
Single-Coin 15M Fibonacci Strategy (`ADAUSDT`) vs V4 Benchmark.

HONESTY PROTOCOL:
  1. Single coin (`ADAUSDT`) and its 15M Fibonacci parameters were selected and locked
     STRICTLY on Days 1-120 (M1-M4) with Days 121-180 physically excluded.
  2. Months 5 & 6 (Days 121-180) are executed ONCE blindly with zero tweaks after seeing results.
  3. Exact Bybit math: $1,000 start compounded, 15% isolated margin x 10x,
     0.02% Maker / 0.055% Taker + 0.04% slippage on SL/BE exits, strict limit fill (`lo < lp`).

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_walkforward import load_cached_15m
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, ema


@dataclass
class SingleCoinFiboSpec:
    name: str
    symbol: str
    is_locked_primary: bool = False
    wave: int = 20
    fib_entry: float = 0.50
    adx_min: float = 18.0
    min_wave_atr: float = 2.2
    tp_pct: float = 1.20
    sl_pct: float = 2.50
    smart_be: bool = True
    be_trigger_ratio: float = 0.65
    be_offset_pct: float = 0.20
    ttl_bars: int = 4
    cd_bars: int = 1
    margin_pct: float = 15.0
    max_daily_stops: int = 4


def prep_single_coin_180d(df_15m: pd.DataFrame, waves: Tuple[int, ...] = (16, 20)) -> pd.DataFrame:
    entry_ms, h1_ms = INTERVAL_TO_MS["15"], INTERVAL_TO_MS["60"]
    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    bar_r = df["high"] - df["low"]
    is_abn = (bar_r > 2.8 * df["atr"]).fillna(False)
    df["vol_cd"] = is_abn | is_abn.shift(1, fill_value=False)

    df_1h = BybitClient.resample_klines(df_15m, "60")
    h1_e20, h1_e50 = ema(df_1h["close"], 20), ema(df_1h["close"], 50)
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out = pd.DataFrame({
        "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
        "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
        "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
        "h1_adx": h1_adx.fillna(0.0).values,
    }).sort_values("close_ts")

    df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
    df["h1_bull"] = df["h1_bull"].astype("boolean").fillna(False).astype(bool)
    df["h1_bear"] = df["h1_bear"].astype("boolean").fillna(False).astype(bool)
    df["h1_adx"] = df["h1_adx"].fillna(0.0)

    hi_v = df["high"].to_numpy(dtype=np.float64)
    lo_v = df["low"].to_numpy(dtype=np.float64)
    n = len(df)
    for w in waves:
        whi = np.full(n, np.nan)
        wlo = np.full(n, np.nan)
        iup = np.zeros(n, dtype=bool)
        idn = np.zeros(n, dtype=bool)
        for i in range(w, n):
            seg_h = hi_v[i - w : i]
            seg_l = lo_v[i - w : i]
            ih = int(np.argmax(seg_h))
            il = int(np.argmin(seg_l))
            whi[i] = seg_h[ih]
            wlo[i] = seg_l[il]
            if ih > il:
                iup[i] = True
            elif il > ih:
                idn[i] = True
        df[f"whi_{w}"] = whi
        df[f"wlo_{w}"] = wlo
        df[f"iup_{w}"] = iup
        df[f"idn_{w}"] = idn

    return df


def simulate_6m_single_coin(
    df: pd.DataFrame,
    spec: SingleCoinFiboSpec,
    sim_start: int,
    sim_end: int,
    month_bounds: List[Tuple[str, int, int]],
    blind_split_ms: int,
) -> Dict[str, Any]:
    w = spec.wave
    ts_all = df["timestamp"].to_numpy(dtype=np.int64)
    mask = (ts_all >= sim_start) & (ts_all <= sim_end)

    ts_arr = ts_all[mask]
    hi_arr = df["high"].to_numpy(dtype=np.float64)[mask]
    lo_arr = df["low"].to_numpy(dtype=np.float64)[mask]
    cl_arr = df["close"].to_numpy(dtype=np.float64)[mask]
    whi = df[f"whi_{w}"].to_numpy(dtype=np.float64)[mask]
    wlo = df[f"wlo_{w}"].to_numpy(dtype=np.float64)[mask]
    iup = df[f"iup_{w}"].to_numpy(dtype=bool)[mask]
    idn = df[f"idn_{w}"].to_numpy(dtype=bool)[mask]
    atr = df["atr"].to_numpy(dtype=np.float64)[mask]
    atr_pct = df["atr_pct"].to_numpy(dtype=np.float64)[mask]
    vol_cd = df["vol_cd"].to_numpy(dtype=bool)[mask]
    trL = df["h1_bull"].to_numpy(dtype=bool)[mask]
    trS = df["h1_bear"].to_numpy(dtype=bool)[mask]
    adx_ok = (df["h1_adx"].to_numpy(dtype=np.float64)[mask]) >= spec.adx_min

    mk, tk, slip = 0.0002, 0.00055, 0.0004
    tp_f, sl_f = spec.tp_pct / 100.0, spec.sl_pct / 100.0
    be_off_f = spec.be_offset_pct / 100.0

    equity, peak, max_dd = 1000.0, 1000.0, 0.0
    blind_peak, blind_max_dd = None, 0.0
    pos, pend = None, None
    cd_until = -1
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end, trades = {}, {}, []

    def _close(p, trig_p, reason, is_mk, bar_i, ts):
        nonlocal equity, peak, max_dd, blind_peak, blind_max_dd, day_sl, halted, cd_until
        is_l = p["dir"] == "LONG"
        ex_p = trig_p if is_mk else (trig_p * (1 - slip) if is_l else trig_p * (1 + slip))
        gross = (ex_p - p["ep"]) * p["qty"] if is_l else (p["ep"] - ex_p) * p["qty"]
        net = max(-p["margin"], gross - p["efee"] - abs(p["qty"] * ex_p) * (mk if is_mk else tk))
        equity = max(1.0, equity + net)
        if reason == "SL":
            day_sl += 1
            if day_sl >= spec.max_daily_stops:
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
        trades.append({"reason": reason, "net": float(net), "is_blind": bool(ts >= blind_split_ms)})
        cd_until = bar_i + spec.cd_bars

    n_bars = len(ts_arr)
    for i in range(n_bars):
        ts = int(ts_arr[i])
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

        hi, lo, cl = hi_arr[i], lo_arr[i], cl_arr[i]

        if pos is not None:
            is_l = pos["dir"] == "LONG"
            if spec.smart_be and pos["be_on"]:
                if (lo <= pos["be_p"]) if is_l else (hi >= pos["be_p"]):
                    _close(pos, pos["be_p"], "BE", False, i, ts)
                    pos = None
                elif (hi >= pos["tp_p"]) if is_l else (lo <= pos["tp_p"]):
                    _close(pos, pos["tp_p"], "TP", True, i, ts)
                    pos = None
            else:
                if (lo <= pos["sl_p"]) if is_l else (hi >= pos["sl_p"]):
                    _close(pos, pos["sl_p"], "SL", False, i, ts)
                    pos = None
                elif (hi >= pos["tp_p"]) if is_l else (lo <= pos["tp_p"]):
                    _close(pos, pos["tp_p"], "TP", True, i, ts)
                    pos = None
                elif spec.smart_be and ((hi >= pos["be_tr"]) if is_l else (lo <= pos["be_tr"])):
                    pos["be_on"] = True

        if pos is None and pend is not None:
            if i > pend["exp"] or halted or i < cd_until:
                pend = None
            else:
                is_l = pend["dir"] == "LONG"
                # Strict Maker fill: price must strictly cross resting limit order
                filled = (lo < pend["lp"]) if is_l else (hi > pend["lp"])
                if filled:
                    margin = equity * (spec.margin_pct / 100.0)
                    notional = margin * 10.0
                    ep = pend["lp"]
                    p_new = {
                        "dir": pend["dir"], "ep": ep, "qty": notional / ep, "margin": margin,
                        "efee": notional * mk, "be_on": False,
                        "sl_p": ep * (1 - sl_f) if is_l else ep * (1 + sl_f),
                        "tp_p": ep * (1 + tp_f) if is_l else ep * (1 - tp_f),
                        "be_tr": ep * (1 + tp_f * spec.be_trigger_ratio) if is_l else ep * (1 - tp_f * spec.be_trigger_ratio),
                        "be_p": ep * (1 + be_off_f) if is_l else ep * (1 - be_off_f),
                    }
                    pend = None
                    if (lo <= p_new["sl_p"]) if is_l else (hi >= p_new["sl_p"]):
                        _close(p_new, p_new["sl_p"], "SL", False, i, ts)
                    else:
                        pos = p_new
                elif i >= pend["exp"]:
                    pend = None

        if pos is None and pend is None and (not halted) and i >= cd_until:
            if vol_cd[i] or (not adx_ok[i]) or atr_pct[i] < 0.10 or atr_pct[i] > 1.80:
                continue
            rng = whi[i] - wlo[i]
            if not np.isfinite(rng) or rng < spec.min_wave_atr * atr[i]:
                continue
            if trL[i] and iup[i]:
                lp = whi[i] - spec.fib_entry * rng
                if lp < cl * 0.9992:
                    pend = {"dir": "LONG", "lp": float(lp), "exp": i + spec.ttl_bars}
            elif trS[i] and idn[i]:
                lp = wlo[i] + spec.fib_entry * rng
                if lp > cl * 1.0008:
                    pend = {"dir": "SHORT", "lp": float(lp), "exp": i + spec.ttl_bars}

    if pos is not None:
        last_ts = int(ts_arr[-1])
        _close(pos, float(cl_arr[-1]), "EOD", False, n_bars - 1, last_ts)
        for mn, _, me in month_bounds:
            if last_ts <= me:
                m_end[mn] = equity

    mp = {mn: round((m_end.get(mn, 1000.0) - m_start.get(mn, 1000.0)) / max(m_start.get(mn, 1000.0), 1.0) * 100.0, 2) for mn, _, _ in month_bounds}
    n = len(trades)
    sl_n = sum(1 for t in trades if t["reason"] == "SL")
    tp_n = sum(1 for t in trades if t["reason"] == "TP")
    be_n = sum(1 for t in trades if t["reason"] == "BE")
    wr = round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2)
    bl_tr = [t for t in trades if t["is_blind"]]
    bl_wr = round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2)
    bl_sl = sum(1 for t in bl_tr if t["reason"] == "SL")
    gp = sum(t["net"] for t in trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in trades if t["net"] < 0))

    return {
        "name": spec.name,
        "symbol": spec.symbol,
        "is_locked_primary": spec.is_locked_primary,
        "spec": asdict(spec),
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n,
        "trades_per_day": round(n / 180.0, 2),
        "wr_pct": wr,
        "sl_count": sl_n,
        "tp_count": tp_n,
        "be_count": be_n,
        "pf": round(gp / max(gl, 1e-9), 2),
        "max_dd_pct": round(max_dd, 2),
        "blind_trades": len(bl_tr),
        "blind_tpd": round(len(bl_tr) / 60.0, 2),
        "blind_wr_pct": bl_wr,
        "blind_sl": bl_sl,
        "blind_max_dd_pct": round(blind_max_dd, 2),
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    day_ms = 86_400_000
    sim_end = int(df_btc["timestamp"].iloc[-1])
    sim_start = sim_end - 180 * day_ms
    blind_split = sim_start + 120 * day_ms

    m_bounds = [
        ("M1", sim_start + 0 * day_ms, sim_start + 30 * day_ms - 1),
        ("M2", sim_start + 30 * day_ms, sim_start + 60 * day_ms - 1),
        ("M3", sim_start + 60 * day_ms, sim_start + 90 * day_ms - 1),
        ("M4", sim_start + 90 * day_ms, sim_start + 120 * day_ms - 1),
        ("M5", sim_start + 120 * day_ms, sim_start + 150 * day_ms - 1),
        ("M6", sim_start + 150 * day_ms, sim_end),
    ]

    df_ada = prep_single_coin_180d(load_cached_15m("ADAUSDT", cache_dir), waves=(16, 20))

    specs = [
        SingleCoinFiboSpec(
            name="LOCKED PRIMARY (ADAUSDT: 15M Wave 20, Fib 0.50 Limit, 1H Trend + ADX>=18, TP1.20/SL2.50/BE+0.20)",
            symbol="ADAUSDT",
            is_locked_primary=True,
            wave=20,
            fib_entry=0.50,
            adx_min=18.0,
            min_wave_atr=2.2,
            tp_pct=1.20,
            sl_pct=2.50,
            be_offset_pct=0.20,
        ),
        SingleCoinFiboSpec(
            name="REF HIGH-PACE (~3/d) (ADAUSDT: 15M Wave 16, Fib 0.382 Limit, 1H Trend + ADX>=15, TP1.00/SL2.20/BE+0.18)",
            symbol="ADAUSDT",
            is_locked_primary=False,
            wave=16,
            fib_entry=0.382,
            adx_min=15.0,
            min_wave_atr=2.0,
            tp_pct=1.00,
            sl_pct=2.20,
            be_offset_pct=0.18,
        ),
    ]

    results = []
    print("\n" + "=" * 116)
    print("POLYGRAPH 6-MONTH EXAM: SINGLE COIN `ADAUSDT` 15M FIBO (15% x 10x ISOLATED | BYBIT FEES + 0.04% SLIPPAGE)")
    print("=" * 116)
    for sp in specs:
        r = simulate_6m_single_coin(df_ada, sp, sim_start, sim_end, m_bounds, blind_split)
        results.append(r)
        mp = r["monthly_pct"]
        print(f"\n>>> {r['name']}")
        print(
            f"  M1:{mp['M1']:>+6.1f}% / M2:{mp['M2']:>+6.1f}% / M3:{mp['M3']:>+6.1f}% / M4:{mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5:{mp['M5']:>+6.1f}% / M6:{mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого: $1 000 -> ${r['end_usd']:,.2f} | Сделок: {r['trades']} ({r['trades_per_day']:.2f}/день; слепых {r['blind_tpd']:.2f}/день) | "
            f"WR: {r['wr_pct']:.1f}% (Blind WR: {r['blind_wr_pct']:.1f}%) | Стопов: {r['sl_count']} (Blind SL: {r['blind_sl']}) | "
            f"PF: {r['pf']:.2f} | Макс. DD: {r['max_dd_pct']:.1f}% (Blind DD: {r['blind_max_dd_pct']:.1f}%)"
        )

    out = {
        "results": results,
        "v4_benchmark": {
            "end_usd": 4344.67,
            "wr_pct": 84.5,
            "max_dd_pct": 20.4,
            "monthly_pct": {"M1": 54.5, "M2": 6.8, "M3": 47.0, "M4": 30.5, "M5": 34.7, "M6": 2.0},
        },
    }
    Path("data/single_coin_fibo_6m_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/single_coin_fibo_6m_results.json")


if __name__ == "__main__":
    main()
