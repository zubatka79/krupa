"""POLYGRAPH 6-MONTH EXAM (`bot/ada_15x_zero_slip_fee01_6m.py`):
`ADAUSDT` 15M Fibo 0.50 at 15x Isolated under Real-Account Execution Model:
  - Slippage: 0.00% (0.0)
  - Maker fee: 0.02% (0.0002, standard Bybit Futures VIP0 Maker) -> Limit Fib entries & Limit TP exits
  - Taker fee: 0.10% (0.0010, per user's Bybit screenshot 0.304946/304.946) -> SL, Smart BE/BE2, H1-Flip/EOD market exits

HONESTY PROTOCOL:
  1. All gentle sweeps and agent ideas were evaluated STRICTLY on Days 1-120 (M1-M4)
     with Days 121-180 physically excluded.
  2. Locked Primary (from Days 1-120):
     - wave=20, Fib=0.50, min_wave_atr=2.2, TTL=4, 1H EMA 20/50, H1 ADX>=18
     - TP 1.20% (Maker) / SL 2.40% (Taker) / 2-Step Smart BE (+0.80%->+0.30%, +0.95%->+0.50%)
     - 100% exit on full 1H EMA flip, max_daily_stops=3, 15% margin x 15x isolated
     - Agent Upgrade: volatility spike-cooldown threshold relaxed 2.8 -> 3.2 x ATR (flat plateau 3.2..3.6,
       stops blocking healthy breakout impulses while still guarding liquidation wicks).
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
class AdaFee01ExamSpec:
    name: str
    is_locked_primary: bool = False
    wave: int = 20
    min_wave_atr: float = 2.2
    fib_entry: float = 0.50
    ttl_bars: int = 4
    h1_adx_min: float = 18.0
    abn_mult: float = 3.2
    tp_pct: float = 1.20
    sl_pct: float = 2.40
    be_trigger_pct: float = 0.80
    be_offset_pct: float = 0.30
    be2_trigger_pct: float = 0.95
    be2_offset_pct: float = 0.50
    rev_h1_flip: bool = True
    max_daily_stops: int = 3
    leverage: float = 15.0
    margin_pct: float = 15.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0010
    slippage: float = 0.0


def prep_ada_180d(df_15m: pd.DataFrame, abn_mult: float = 3.2) -> pd.DataFrame:
    entry_ms, h1_ms = INTERVAL_TO_MS["15"], INTERVAL_TO_MS["60"]
    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    bar_r = df["high"] - df["low"]
    is_abn = (bar_r > abn_mult * df["atr"]).fillna(False)
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
    whi, wlo = np.full(n, np.nan), np.full(n, np.nan)
    iup, idn = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    for i in range(w, n):
        sh, sl = hi_v[i - w : i], lo_v[i - w : i]
        ih, il = int(np.argmax(sh)), int(np.argmin(sl))
        whi[i], wlo[i] = sh[ih], sl[il]
        if ih > il:
            iup[i] = True
        elif il > ih:
            idn[i] = True
    df["whi"], df["wlo"], df["iup"], df["idn"] = whi, wlo, iup, idn
    return df


def simulate_6m_fee01(
    df: pd.DataFrame,
    sp: AdaFee01ExamSpec,
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

    mk, tk, slip = sp.maker_fee, sp.taker_fee, sp.slippage
    tp_f, sl_f = sp.tp_pct / 100.0, sp.sl_pct / 100.0
    be_tr_f, be_off_f = sp.be_trigger_pct / 100.0, sp.be_offset_pct / 100.0
    be2_tr_f, be2_off_f = sp.be2_trigger_pct / 100.0, sp.be2_offset_pct / 100.0

    equity, peak, max_dd = 1000.0, 1000.0, 0.0
    blind_peak, blind_max_dd = None, 0.0
    pos, pend = None, None
    cd_until = -1
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end, trades = {}, {}, []
    total_maker_fees, total_taker_fees = 0.0, 0.0

    def _finish(p, trig_p, reason, is_mk, bar_i, ts):
        nonlocal equity, peak, max_dd, blind_peak, blind_max_dd, day_sl, halted, cd_until
        nonlocal total_maker_fees, total_taker_fees
        is_l = p["dir"] == "LONG"
        ex_p = trig_p if (is_mk or slip == 0.0) else (trig_p * (1 - slip) if is_l else trig_p * (1 + slip))
        gross = (ex_p - p["ep"]) * p["qty"] if is_l else (p["ep"] - ex_p) * p["qty"]
        ex_fee = abs(p["qty"] * ex_p) * (mk if is_mk else tk)
        total_maker_fees += p["efee"] + (ex_fee if is_mk else 0.0)
        total_taker_fees += (0.0 if is_mk else ex_fee)
        net = max(-p["margin"], gross - p["efee"] - ex_fee)
        equity = max(1.0, equity + net)
        if reason == "SL":
            day_sl += 1
            if day_sl >= sp.max_daily_stops:
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
            cur_sl = pos["be2_p"] if pos["be2_on"] else (pos["be_p"] if pos["be_on"] else pos["sl_p"])
            cur_r = "BE2" if pos["be2_on"] else ("BE" if pos["be_on"] else "SL")
            closed = False
            if (lo <= cur_sl) if is_l else (hi >= cur_sl):
                _finish(pos, cur_sl, cur_r, False, i, ts)
                pos = None
                closed = True
            elif (hi >= pos["tp_p"]) if is_l else (lo <= pos["tp_p"]):
                _finish(pos, pos["tp_p"], "TP", True, i, ts)
                pos = None
                closed = True
            else:
                if (not pos["be_on"]) and ((hi >= pos["be_tr"]) if is_l else (lo <= pos["be_tr"])):
                    pos["be_on"] = True
                if be2_tr_f > 0 and (not pos["be2_on"]) and ((hi >= pos["be2_tr"]) if is_l else (lo <= pos["be2_tr"])):
                    pos["be2_on"] = True

            if (not closed) and pos is not None and sp.rev_h1_flip:
                if (h1_bear[i] if is_l else h1_bull[i]):
                    _finish(pos, float(cl), "REV", False, i, ts)
                    pos = None

        if pos is None and pend is not None:
            if i > pend["exp"] or halted or i < cd_until:
                pend = None
            else:
                is_l = pend["dir"] == "LONG"
                filled = (lo < pend["lp"]) if is_l else (hi > pend["lp"])
                if filled:
                    margin = equity * (sp.margin_pct / 100.0)
                    notional = margin * sp.leverage
                    ep = pend["lp"]
                    p_new = {
                        "dir": pend["dir"], "ep": ep, "qty": notional / ep, "margin": margin,
                        "efee": notional * mk, "be_on": False, "be2_on": False,
                        "sl_p": ep * (1 - sl_f) if is_l else ep * (1 + sl_f),
                        "tp_p": ep * (1 + tp_f) if is_l else ep * (1 - tp_f),
                        "be_tr": ep * (1 + be_tr_f) if is_l else ep * (1 - be_tr_f),
                        "be_p": ep * (1 + be_off_f) if is_l else ep * (1 - be_off_f),
                        "be2_tr": ep * (1 + be2_tr_f) if is_l else ep * (1 - be2_tr_f),
                        "be2_p": ep * (1 + be2_off_f) if is_l else ep * (1 - be2_off_f),
                    }
                    pend = None
                    if (lo <= p_new["sl_p"]) if is_l else (hi >= p_new["sl_p"]):
                        _finish(p_new, p_new["sl_p"], "SL", False, i, ts)
                    else:
                        pos = p_new
                elif i >= pend["exp"]:
                    pend = None

        if pos is None and pend is None and (not halted) and i >= cd_until:
            if vol_cd[i] or h1_adx[i] < sp.h1_adx_min or atr_pct[i] < 0.10 or atr_pct[i] > 1.80:
                continue
            rng = whi[i] - wlo[i]
            if not np.isfinite(rng) or rng < sp.min_wave_atr * atr[i]:
                continue
            if h1_bull[i] and iup[i]:
                lp = whi[i] - sp.fib_entry * rng
                if lp < cl * 0.9992:
                    pend = {"dir": "LONG", "lp": float(lp), "exp": i + sp.ttl_bars}
            elif h1_bear[i] and idn[i]:
                lp = wlo[i] + sp.fib_entry * rng
                if lp > cl * 1.0008:
                    pend = {"dir": "SHORT", "lp": float(lp), "exp": i + sp.ttl_bars}

    if pos is not None:
        last_ts = int(ts_arr[-1])
        _finish(pos, float(cl_arr[-1]), "EOD", False, n_bars - 1, last_ts)
        for mn, _, me in month_bounds:
            if last_ts <= me:
                m_end[mn] = equity

    mp = {mn: round((m_end.get(mn, 1000.0) - m_start.get(mn, 1000.0)) / max(m_start.get(mn, 1000.0), 1.0) * 100.0, 2) for mn, _, _ in month_bounds}
    n = len(trades)
    sl_n = sum(1 for t in trades if t["reason"] == "SL")
    tp_n = sum(1 for t in trades if t["reason"] == "TP")
    be_n = sum(1 for t in trades if t["reason"] in ("BE", "BE2"))
    wr = round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2)
    bl_tr = [t for t in trades if t["is_blind"]]
    bl_wr = round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2)
    bl_sl = sum(1 for t in bl_tr if t["reason"] == "SL")
    gp = sum(t["net"] for t in trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in trades if t["net"] < 0))

    return {
        "name": sp.name,
        "is_locked_primary": sp.is_locked_primary,
        "spec": asdict(sp),
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
        "blind_wr_pct": bl_wr,
        "blind_sl": bl_sl,
        "blind_max_dd_pct": round(blind_max_dd, 2),
        "fees_paid_usd": {"maker": round(total_maker_fees, 2), "taker": round(total_taker_fees, 2)},
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

    df_ada_raw = load_cached_15m("ADAUSDT", cache_dir)
    df_32 = prep_ada_180d(df_ada_raw, abn_mult=3.2)
    df_28 = prep_ada_180d(df_ada_raw, abn_mult=2.8)

    runs = [
        (df_32, AdaFee01ExamSpec(
            name="LOCKED PRIMARY (abn=3.2xATR, Mk 0.02% / Tk 0.10% / Slip 0%)",
            is_locked_primary=True, abn_mult=3.2, maker_fee=0.0002, taker_fee=0.0010, slippage=0.0,
        )),
        (df_28, AdaFee01ExamSpec(
            name="REF-PREV-NEWFEE (abn=2.8xATR, Mk 0.02% / Tk 0.10% / Slip 0%)",
            is_locked_primary=False, abn_mult=2.8, maker_fee=0.0002, taker_fee=0.0010, slippage=0.0,
        )),
        (df_32, AdaFee01ExamSpec(
            name="REF-STRESS-FLAT010 (abn=3.2xATR, Mk 0.10% / Tk 0.10% / Slip 0%)",
            is_locked_primary=False, abn_mult=3.2, maker_fee=0.0010, taker_fee=0.0010, slippage=0.0,
        )),
    ]

    results = []
    print("\n" + "=" * 116)
    print("POLYGRAPH 6-MONTH EXAM: `ada_15x_zero_slip_fee01_6m.py` (ADAUSDT 15x | SLIP 0% | TAKER 0.10% | MAKER 0.02%)")
    print("=" * 116)
    for df_use, sp in runs:
        r = simulate_6m_fee01(df_use, sp, sim_start, sim_end, m_bounds, blind_split)
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
        print(
            f"  Exits: TP(Maker)={r['tp_count']}, BE/BE2(Taker)={r['be_count']}, SL(Taker)={r['sl_count']} | "
            f"Fees paid: Maker=${r['fees_paid_usd']['maker']:,.2f}, Taker=${r['fees_paid_usd']['taker']:,.2f}"
        )

    Path("data/ada_15x_zero_slip_fee01_6m_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/ada_15x_zero_slip_fee01_6m_results.json")


if __name__ == "__main__":
    main()
