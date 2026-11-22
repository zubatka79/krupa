"""POLYGRAPH 6-MONTH EXAM (`bot/ada_tune_15x_6m.py`):
Tuned `ADAUSDT` 15M Fibo 0.50 Profile at 15x Isolated Leverage.

HONESTY PROTOCOL:
  1. TP sweep (1.2%..3.0% step 0.1%), SL sweep (2.5%..1.2% step 0.1%), trend-reversal
     partial close (50%..100%), and agent upgrades were evaluated STRICTLY on Days 1-120
     (M1-M4) with Days 121-180 physically excluded.
  2. Locked Primary Champion (from Days 1-120):
     - `ADAUSDT`, 15M Wave 20, Fib 0.50 Maker Limit (TTL 4), 1H EMA20/EMA50 + H1 ADX>=18
     - Margin 15%, Leverage 15x isolated
     - TP 1.20% (Maker) / SL 2.40% (Taker + 0.04% slip)
     - Trend-Reversal Exit: 100% close on full 1H EMA20/EMA50 flip against position
     - Agent Upgrades: Slippage-resilient Smart BE (+0.80% -> +0.25%), max_daily_stops=3
  3. Months 5 & 6 (Days 121-180) are executed ONCE blindly with zero tweaks.

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
class Ada15xExamSpec:
    name: str
    is_locked_primary: bool = False
    leverage: float = 15.0
    margin_pct: float = 15.0
    tp_pct: float = 1.20
    sl_pct: float = 2.40
    smart_be: bool = True
    be_trigger_pct: float = 0.80   # if <= 0, uses 0.65 * tp_pct
    be_offset_pct: float = 0.25
    rev_mode: str = "h1_flip"      # "none", "h1_flip"
    rev_close_pct: float = 1.00
    max_daily_stops: int = 3
    h1_adx_min: float = 18.0


def prep_ada_180d(df_15m: pd.DataFrame) -> pd.DataFrame:
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
    n, w = len(df), 20
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
    df["whi"], df["wlo"], df["iup"], df["idn"] = whi, wlo, iup, idn
    return df


def simulate_6m_ada_15x(
    df: pd.DataFrame,
    spec: Ada15xExamSpec,
    sim_start: int,
    sim_end: int,
    month_bounds: List[Tuple[str, int, int]],
    blind_split_ms: int,
) -> Dict[str, Any]:
    ts_all = df["timestamp"].to_numpy(dtype=np.int64)
    m = (ts_all >= sim_start) & (ts_all <= sim_end)
    ts_arr = ts_all[m]
    hi_arr = df["high"].to_numpy(dtype=np.float64)[m]
    lo_arr = df["low"].to_numpy(dtype=np.float64)[m]
    cl_arr = df["close"].to_numpy(dtype=np.float64)[m]
    whi = df["whi"].to_numpy(dtype=np.float64)[m]
    wlo = df["wlo"].to_numpy(dtype=np.float64)[m]
    iup = df["iup"].to_numpy(dtype=bool)[m]
    idn = df["idn"].to_numpy(dtype=bool)[m]
    atr = df["atr"].to_numpy(dtype=np.float64)[m]
    atr_pct = df["atr_pct"].to_numpy(dtype=np.float64)[m]
    vol_cd = df["vol_cd"].to_numpy(dtype=bool)[m]
    h1_bull = df["h1_bull"].to_numpy(dtype=bool)[m]
    h1_bear = df["h1_bear"].to_numpy(dtype=bool)[m]
    h1_adx = df["h1_adx"].to_numpy(dtype=np.float64)[m]

    mk, tk, slip = 0.0002, 0.00055, 0.0004
    tp_f, sl_f = spec.tp_pct / 100.0, spec.sl_pct / 100.0
    be_tr_f = (spec.be_trigger_pct / 100.0) if spec.be_trigger_pct > 0 else (tp_f * 0.65)
    be_off_f = spec.be_offset_pct / 100.0

    equity, peak, max_dd = 1000.0, 1000.0, 0.0
    blind_peak, blind_max_dd = None, 0.0
    pos, pend = None, None
    cd_until = -1
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end, trades = {}, {}, []
    rev_triggers = 0

    def _realize_part(p, trig_p, frac, is_mk, ts):
        nonlocal equity, peak, max_dd, blind_peak, blind_max_dd
        is_l = p["dir"] == "LONG"
        ex_p = trig_p if is_mk else (trig_p * (1 - slip) if is_l else trig_p * (1 + slip))
        q_c = p["qty"] * frac
        m_c = p["margin"] * frac
        ef_c = p["efee"] * frac
        gross = (ex_p - p["ep"]) * q_c if is_l else (p["ep"] - ex_p) * q_c
        net = max(-m_c, gross - ef_c - abs(q_c * ex_p) * (mk if is_mk else tk))
        equity = max(1.0, equity + net)
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
        p["qty"] -= q_c
        p["margin"] -= m_c
        p["efee"] -= ef_c
        p["cum_net"] += net

    def _finish(p, trig_p, reason, is_mk, bar_i, ts):
        nonlocal day_sl, halted, cd_until
        _realize_part(p, trig_p, 1.0, is_mk, ts)
        if reason == "SL" and p["cum_net"] < 0:
            day_sl += 1
            if day_sl >= spec.max_daily_stops:
                halted = True
        trades.append({"reason": reason, "net": float(p["cum_net"]), "is_blind": bool(ts >= blind_split_ms)})
        cd_until = bar_i + 1

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
            closed = False
            cur_sl = pos["be_p"] if (spec.smart_be and pos["be_on"]) else pos["sl_p"]
            cur_sl_reason = "BE" if (spec.smart_be and pos["be_on"]) else "SL"
            if (lo <= cur_sl) if is_l else (hi >= cur_sl):
                _finish(pos, cur_sl, cur_sl_reason, False, i, ts)
                pos = None
                closed = True
            elif (hi >= pos["tp_p"]) if is_l else (lo <= pos["tp_p"]):
                _finish(pos, pos["tp_p"], "TP", True, i, ts)
                pos = None
                closed = True
            elif spec.smart_be and (not pos["be_on"]):
                if (hi >= pos["be_tr"]) if is_l else (lo <= pos["be_tr"]):
                    pos["be_on"] = True

            if (not closed) and pos is not None and spec.rev_mode == "h1_flip" and spec.rev_close_pct > 0 and (not pos["rev_done"]):
                rev_hit = bool(h1_bear[i]) if is_l else bool(h1_bull[i])
                if rev_hit:
                    rev_triggers += 1
                    if spec.rev_close_pct >= 0.999:
                        _finish(pos, float(cl), "REV_EXIT", False, i, ts)
                        pos = None
                    else:
                        _realize_part(pos, float(cl), spec.rev_close_pct, False, ts)
                        pos["rev_done"] = True

        if pos is None and pend is not None:
            if i > pend["exp"] or halted or i < cd_until:
                pend = None
            else:
                is_l = pend["dir"] == "LONG"
                filled = (lo < pend["lp"]) if is_l else (hi > pend["lp"])
                if filled:
                    margin = equity * (spec.margin_pct / 100.0)
                    notional = margin * spec.leverage
                    ep = pend["lp"]
                    p_new = {
                        "dir": pend["dir"], "ep": ep, "qty": notional / ep, "margin": margin,
                        "efee": notional * mk, "be_on": False, "rev_done": False, "cum_net": 0.0,
                        "sl_p": ep * (1 - sl_f) if is_l else ep * (1 + sl_f),
                        "tp_p": ep * (1 + tp_f) if is_l else ep * (1 - tp_f),
                        "be_tr": ep * (1 + be_tr_f) if is_l else ep * (1 - be_tr_f),
                        "be_p": ep * (1 + be_off_f) if is_l else ep * (1 - be_off_f),
                    }
                    pend = None
                    if (lo <= p_new["sl_p"]) if is_l else (hi >= p_new["sl_p"]):
                        _finish(p_new, p_new["sl_p"], "SL", False, i, ts)
                    else:
                        pos = p_new
                elif i >= pend["exp"]:
                    pend = None

        if pos is None and pend is None and (not halted) and i >= cd_until:
            if vol_cd[i] or h1_adx[i] < spec.h1_adx_min or atr_pct[i] < 0.10 or atr_pct[i] > 1.80:
                continue
            rng = whi[i] - wlo[i]
            if not np.isfinite(rng) or rng < 2.2 * atr[i]:
                continue
            if h1_bull[i] and iup[i]:
                lp = whi[i] - 0.50 * rng
                if lp < cl * 0.9992:
                    pend = {"dir": "LONG", "lp": float(lp), "exp": i + 4}
            elif h1_bear[i] and idn[i]:
                lp = wlo[i] + 0.50 * rng
                if lp > cl * 1.0008:
                    pend = {"dir": "SHORT", "lp": float(lp), "exp": i + 4}

    if pos is not None:
        last_ts = int(ts_arr[-1])
        _finish(pos, float(cl_arr[-1]), "EOD", False, n_bars - 1, last_ts)
        for mn, _, me in month_bounds:
            if last_ts <= me:
                m_end[mn] = equity

    mp = {mn: round((m_end.get(mn, 1000.0) - m_start.get(mn, 1000.0)) / max(m_start.get(mn, 1000.0), 1.0) * 100.0, 2) for mn, _, _ in month_bounds}
    n = len(trades)
    sl_n = sum(1 for t in trades if t["reason"] == "SL" and t["net"] < 0)
    wr = round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2)
    bl_tr = [t for t in trades if t["is_blind"]]
    bl_wr = round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2)
    bl_sl = sum(1 for t in bl_tr if t["reason"] == "SL" and t["net"] < 0)
    gp = sum(t["net"] for t in trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in trades if t["net"] < 0))

    return {
        "name": spec.name,
        "is_locked_primary": spec.is_locked_primary,
        "spec": asdict(spec),
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n,
        "trades_per_day": round(n / 180.0, 2),
        "wr_pct": wr,
        "sl_count": sl_n,
        "rev_triggers": rev_triggers,
        "pf": round(gp / max(gl, 1e-9), 2),
        "max_dd_pct": round(max_dd, 2),
        "blind_trades": len(bl_tr),
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

    df_ada = prep_ada_180d(load_cached_15m("ADAUSDT", cache_dir))

    specs = [
        Ada15xExamSpec(
            name="LOCKED PRIMARY 15x (TP 1.20% / SL 2.40% / BE@0.80->+0.25% / H1-Flip 100% / dSL=3)",
            is_locked_primary=True,
            leverage=15.0,
            tp_pct=1.20,
            sl_pct=2.40,
            be_trigger_pct=0.80,
            be_offset_pct=0.25,
            rev_mode="h1_flip",
            rev_close_pct=1.00,
            max_daily_stops=3,
        ),
        Ada15xExamSpec(
            name="REF-BASE 15x (Untuned: TP 1.20% / SL 2.50% / BE@0.78->+0.20% / No RevExit / dSL=4)",
            is_locked_primary=False,
            leverage=15.0,
            tp_pct=1.20,
            sl_pct=2.50,
            be_trigger_pct=0.78,
            be_offset_pct=0.20,
            rev_mode="none",
            rev_close_pct=0.0,
            max_daily_stops=4,
        ),
        Ada15xExamSpec(
            name="REF-RUNNER 15x (High-TP: TP 3.00% / SL 2.40% / PropBE@1.95->+0.22% / H1-Flip 80% / dSL=3)",
            is_locked_primary=False,
            leverage=15.0,
            tp_pct=3.00,
            sl_pct=2.40,
            be_trigger_pct=0.0,
            be_offset_pct=0.22,
            rev_mode="h1_flip",
            rev_close_pct=0.80,
            max_daily_stops=3,
        ),
    ]

    results = []
    print("\n" + "=" * 116)
    print("POLYGRAPH 6-MONTH EXAM: `ADAUSDT` 15M FIBO 0.50 AT 15x ISOLATED (15% MARGIN | BYBIT FEES + 0.04% SLIPPAGE)")
    print("=" * 116)
    for sp in specs:
        r = simulate_6m_ada_15x(df_ada, sp, sim_start, sim_end, m_bounds, blind_split)
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
            f"RevExits: {r['rev_triggers']} | PF: {r['pf']:.2f} | Макс. DD: {r['max_dd_pct']:.1f}% (Blind DD: {r['blind_max_dd_pct']:.1f}%)"
        )

    out = {
        "results": results,
        "ada_10x_benchmark": {
            "end_usd": 3241.07,
            "wr_pct": 84.3,
            "max_dd_pct": 13.7,
            "blind_dd_pct": 12.9,
            "monthly_pct": {"M1": 38.6, "M2": 41.1, "M3": 15.7, "M4": 12.5, "M5": 7.6, "M6": 18.4},
        },
    }
    Path("data/ada_tune_15x_6m_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/ada_tune_15x_6m_results.json")


if __name__ == "__main__":
    main()
