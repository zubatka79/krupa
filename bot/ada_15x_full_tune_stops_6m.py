"""POLYGRAPH 6-MONTH EXAM (`bot/ada_15x_full_tune_stops_6m.py`):
Head-to-Head Comparison of Variant A (`$9,578.45` Locked Profile) vs Variant B (Stop-Reduction Rebuild)
for `ADAUSDT` 15M Fibo 0.50 at 15x Isolated (Slippage 0%, Maker 0.02%, Taker 0.10%),
plus Full 6-Month Stop-Loss Root-Cause Forensics (M1-M4 vs Blind M5-M6).

HONESTY PROTOCOL:
  1. All stop-loss forensics, parameter checks, and selection of Variant B were locked
     STRICTLY on Days 1-120 (M1-M4) with Days 121-180 physically excluded.
  2. Variant B changes locked from Days 1-120 (every change improved BOTH M1-M2 and M3-M4):
     - US Macro Shield: block entries -30m..+15m around scheduled US FOMC/CPI/NFP/PPI/Fed Chair
       events; at -30m close underwater/flat open trades (< +0.25% uPnL) at market.
     - BTC 1H Adverse Shock Veto: skip LONG limit if BTC 1H <= -1.5% (SHORT if BTC 1H >= +1.5%).
     - Stop-Loss breathing room under 0% slippage: 2.40% -> 2.50% (cuts borderline wick SLs).
  3. Months 5 & 6 (Days 121-180) are executed ONCE blindly with zero tweaks.

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_walkforward import load_cached_15m
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.forensics_heal_6m import US_MACRO_180D
from bot.indicators import compute_adx, compute_atr, ema


@dataclass
class AdaStopExamSpec:
    name: str
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
    skip_macro_win: bool = False
    protect_macro_30m: bool = False
    btc_1h_veto: float = 0.0
    leverage: float = 15.0
    margin_pct: float = 15.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0010
    slippage: float = 0.0


def prep_ada_full_180d(df_ada: pd.DataFrame, df_btc: pd.DataFrame) -> pd.DataFrame:
    entry_ms, h1_ms = INTERVAL_TO_MS["15"], INTERVAL_TO_MS["60"]
    df = df_ada.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    bar_r = df["high"] - df["low"]
    is_abn = (bar_r > 3.2 * df["atr"]).fillna(False)
    df["vol_cd"] = is_abn | is_abn.shift(1, fill_value=False)
    df["utc_hour"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.hour

    btc = df_btc[["timestamp", "close"]].copy().sort_values("timestamp")
    btc["btc_r1h"] = btc["close"].pct_change(4) * 100.0
    btc["btc_r4h"] = btc["close"].pct_change(16) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_r1h", "btc_r4h"]], on="timestamp", direction="backward")
    df["btc_r1h"] = df["btc_r1h"].fillna(0.0)
    df["btc_r4h"] = df["btc_r4h"].fillna(0.0)

    cts = df["close_ts"].to_numpy(dtype=np.int64)
    m_wick = np.zeros(len(df), dtype=bool)
    m_prot = np.zeros(len(df), dtype=bool)
    m_wide = np.zeros(len(df), dtype=bool)
    for iso in US_MACRO_180D:
        ev = int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
        m_wick |= (cts >= ev - 30 * 60_000) & (cts <= ev + 15 * 60_000)
        m_prot |= (cts >= ev - 35 * 60_000) & (cts <= ev - 20 * 60_000)
        m_wide |= (cts >= ev - 60 * 60_000) & (cts <= ev + 120 * 60_000)
    df["macro_wick"] = m_wick
    df["macro_prot"] = m_prot
    df["macro_wide"] = m_wide

    df_1h = BybitClient.resample_klines(df_ada, "60")
    h1_e20, h1_e50 = ema(df_1h["close"], 20), ema(df_1h["close"], 50)
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out = pd.DataFrame({
        "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
        "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
        "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
        "h1_gt50": (df_1h["close"] > h1_e50),
        "h1_lt50": (df_1h["close"] < h1_e50),
        "h1_adx": h1_adx.fillna(0.0).values,
    }).sort_values("close_ts")

    df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
    for c in ("h1_bull", "h1_bear", "h1_gt50", "h1_lt50"):
        df[c] = df[c].astype("boolean").fillna(False).astype(bool)
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


def summarize_sl_forensics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    sl = [t for t in trades if t["reason"] == "SL"]
    n_sl = len(sl)
    if n_sl == 0:
        return {"sl_count": 0}
    return {
        "sl_count": n_sl,
        "btc_adverse_move": sum(1 for t in sl if t["btc_adv_exit"]),
        "btc_adverse_pct": round(sum(1 for t in sl if t["btc_adv_exit"]) / n_sl * 100.0, 1),
        "h1_e50_broken": sum(1 for t in sl if t["h1_lost_at_sl"]),
        "h1_e50_broken_pct": round(sum(1 for t in sl if t["h1_lost_at_sl"]) / n_sl * 100.0, 1),
        "us_macro_window": sum(1 for t in sl if t["macro_hit"]),
        "us_macro_pct": round(sum(1 for t in sl if t["macro_hit"]) / n_sl * 100.0, 1),
        "post_ny_20_24_utc": sum(1 for t in sl if t["ehour"] in (20, 21, 22, 23)),
        "post_ny_pct": round(sum(1 for t in sl if t["ehour"] in (20, 21, 22, 23)) / n_sl * 100.0, 1),
        "near_be_mfe_ge_05": sum(1 for t in sl if t["mfe"] >= 0.50),
        "high_vol_atr_ge_09": sum(1 for t in sl if t["eatr"] >= 0.90),
    }


def simulate_6m_variant(
    df: pd.DataFrame,
    sp: AdaStopExamSpec,
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
    utc_h = df["utc_hour"].to_numpy(dtype=np.int32)[m]
    btc_r1h = df["btc_r1h"].to_numpy(dtype=np.float64)[m]
    btc_r4h = df["btc_r4h"].to_numpy(dtype=np.float64)[m]
    macro_w = df["macro_wick"].to_numpy(dtype=bool)[m]
    macro_p = df["macro_prot"].to_numpy(dtype=bool)[m]
    macro_wide = df["macro_wide"].to_numpy(dtype=bool)[m]
    h1_bull = df["h1_bull"].to_numpy(dtype=bool)[m]
    h1_bear = df["h1_bear"].to_numpy(dtype=bool)[m]
    h1_gt50 = df["h1_gt50"].to_numpy(dtype=bool)[m]
    h1_lt50 = df["h1_lt50"].to_numpy(dtype=bool)[m]
    h1_adx = df["h1_adx"].to_numpy(dtype=np.float64)[m]

    mk, tk = sp.maker_fee, sp.taker_fee
    tp_f, sl_f = sp.tp_pct / 100.0, sp.sl_pct / 100.0
    be_tr_f, be_off_f = sp.be_trigger_pct / 100.0, sp.be_offset_pct / 100.0
    be2_tr_f, be2_off_f = sp.be2_trigger_pct / 100.0, sp.be2_offset_pct / 100.0

    equity, peak, max_dd = 1000.0, 1000.0, 0.0
    blind_peak, blind_max_dd = None, 0.0
    pos, pend = None, None
    cd_until = -1
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end, trades = {}, {}, []

    def _finish(p, trig_p, reason, is_mk, bar_i, ts):
        nonlocal equity, peak, max_dd, blind_peak, blind_max_dd, day_sl, halted, cd_until
        is_l = p["dir"] == "LONG"
        gross = (trig_p - p["ep"]) * p["qty"] if is_l else (p["ep"] - trig_p) * p["qty"]
        net = max(-p["margin"], gross - p["efee"] - abs(p["qty"] * trig_p) * (mk if is_mk else tk))
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
        trades.append({
            "reason": reason,
            "net": float(net),
            "is_blind": bool(ts >= blind_split_ms),
            "ehour": p["eh"],
            "eatr": p["eatr"],
            "mfe": p["mfe"],
            "btc_adv_exit": bool((btc_r1h[bar_i] <= -0.8 or btc_r4h[bar_i] <= -1.8) if is_l else (btc_r1h[bar_i] >= 0.8 or btc_r4h[bar_i] >= 1.8)),
            "h1_lost_at_sl": bool((not h1_gt50[bar_i]) if is_l else (not h1_lt50[bar_i])),
            "macro_hit": bool(p["emac"] or macro_wide[bar_i]),
        })
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
            fav = ((hi - pos["ep"]) / pos["ep"] * 100.0) if is_l else ((pos["ep"] - lo) / pos["ep"] * 100.0)
            if fav > pos["mfe"]:
                pos["mfe"] = fav
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

            if (not closed) and pos is not None:
                upnl = ((cl - pos["ep"]) / pos["ep"] * 100.0) if is_l else ((pos["ep"] - cl) / pos["ep"] * 100.0)
                if sp.rev_h1_flip and (h1_bear[i] if is_l else h1_bull[i]):
                    _finish(pos, float(cl), "REV", False, i, ts)
                    pos = None
                elif sp.protect_macro_30m and macro_p[i] and upnl < 0.25:
                    _finish(pos, float(cl), "MACRO_PROT", False, i, ts)
                    pos = None

        if pos is None and pend is not None:
            if i > pend["exp"] or halted or i < cd_until or (sp.skip_macro_win and macro_w[i]):
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
                        "efee": notional * mk, "be_on": False, "be2_on": False, "mfe": 0.0,
                        "eh": int(utc_h[i]), "eatr": float(atr_pct[i]), "emac": bool(macro_wide[i]),
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
            if sp.skip_macro_win and macro_w[i]:
                continue
            rng = whi[i] - wlo[i]
            if not np.isfinite(rng) or rng < sp.min_wave_atr * atr[i]:
                continue
            if h1_bull[i] and iup[i]:
                if sp.btc_1h_veto > 0 and btc_r1h[i] <= -sp.btc_1h_veto:
                    continue
                lp = whi[i] - sp.fib_entry * rng
                if lp < cl * 0.9992:
                    pend = {"dir": "LONG", "lp": float(lp), "exp": i + sp.ttl_bars}
            elif h1_bear[i] and idn[i]:
                if sp.btc_1h_veto > 0 and btc_r1h[i] >= sp.btc_1h_veto:
                    continue
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
    dev_tr = [t for t in trades if not t["is_blind"]]
    bl_tr = [t for t in trades if t["is_blind"]]
    sl_n = sum(1 for t in trades if t["reason"] == "SL")
    bl_sl = sum(1 for t in bl_tr if t["reason"] == "SL")
    wr = round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2)
    bl_wr = round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2)
    gp = sum(t["net"] for t in trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in trades if t["net"] < 0))

    return {
        "name": sp.name,
        "spec": asdict(sp),
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n,
        "trades_per_day": round(n / 180.0, 2),
        "wr_pct": wr,
        "blind_wr_pct": bl_wr,
        "sl_count": sl_n,
        "blind_sl": bl_sl,
        "macro_prot_exits": sum(1 for t in trades if t["reason"] == "MACRO_PROT"),
        "pf": round(gp / max(gl, 1e-9), 2),
        "max_dd_pct": round(max_dd, 2),
        "blind_max_dd_pct": round(blind_max_dd, 2),
        "sl_forensics_6m": summarize_sl_forensics(trades),
        "sl_forensics_m1_m4": summarize_sl_forensics(dev_tr),
        "sl_forensics_m5_m6_blind": summarize_sl_forensics(bl_tr),
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    df_ada = load_cached_15m("ADAUSDT", cache_dir)
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

    df_full = prep_ada_full_180d(df_ada, df_btc)

    specs = [
        AdaStopExamSpec(
            name="ВАРИАНТ А (Текущий $9,578.45: SL 2.40%, без макро-щита и BTC-вето)",
            sl_pct=2.40,
            skip_macro_win=False,
            protect_macro_30m=False,
            btc_1h_veto=0.0,
        ),
        AdaStopExamSpec(
            name="ВАРИАНТ Б (Stop-Shield: US Macro -30m..+15m & Protect + BTC 1H Veto 1.5% + SL 2.50%)",
            sl_pct=2.50,
            skip_macro_win=True,
            protect_macro_30m=True,
            btc_1h_veto=1.5,
        ),
    ]

    results = []
    print("\n" + "=" * 116)
    print("POLYGRAPH 6-MONTH EXAM: VARIANT A vs VARIANT B (`ada_15x_full_tune_stops_6m.py` | 15x | SLIP 0% | MK 0.02% / TK 0.10%)")
    print("=" * 116)
    for sp in specs:
        r = simulate_6m_variant(df_full, sp, sim_start, sim_end, m_bounds, blind_split)
        results.append(r)
        mp = r["monthly_pct"]
        f_all, f_dev, f_bl = r["sl_forensics_6m"], r["sl_forensics_m1_m4"], r["sl_forensics_m5_m6_blind"]
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
            f"  SL Forensics M1-M4 ({f_dev['sl_count']} SL): BTC_adv={f_dev['btc_adverse_move']} ({f_dev['btc_adverse_pct']}%), "
            f"H1_E50_brk={f_dev['h1_e50_broken']} ({f_dev['h1_e50_broken_pct']}%), USMacro={f_dev['us_macro_window']} ({f_dev['us_macro_pct']}%), "
            f"20-24UTC={f_dev['post_ny_20_24_utc']} ({f_dev['post_ny_pct']}%), NearBE={f_dev['near_be_mfe_ge_05']}"
        )
        print(
            f"  SL Forensics BLIND M5-M6 ({f_bl['sl_count']} SL): BTC_adv={f_bl['btc_adverse_move']} ({f_bl['btc_adverse_pct']}%), "
            f"H1_E50_brk={f_bl['h1_e50_broken']} ({f_bl['h1_e50_broken_pct']}%), USMacro={f_bl['us_macro_window']} ({f_bl['us_macro_pct']}%), "
            f"20-24UTC={f_bl['post_ny_20_24_utc']} ({f_bl['post_ny_pct']}%), NearBE={f_bl['near_be_mfe_ge_05']}"
        )

    Path("data/ada_15x_full_tune_stops_6m_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/ada_15x_full_tune_stops_6m_results.json")


if __name__ == "__main__":
    main()
