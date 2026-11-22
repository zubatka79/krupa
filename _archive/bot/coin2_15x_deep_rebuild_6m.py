"""POLYGRAPH 6-MONTH EXAM (`bot/coin2_15x_deep_rebuild_6m.py`):
Track 1 (Rebuilt `LTCUSDT`) + Track 2 (Best Alternative Coin `ONDOUSDT`) at 15x Isolated Leverage.

HONESTY PROTOCOL:
  1. Both Track 1 (`LTCUSDT` full rebuild) and Track 2 (42-coin ATR-normalized search -> `ONDOUSDT`)
     were selected and parameter-locked STRICTLY on Days 1-120 (M1-M4) with Days 121-180 excluded.
  2. Execution: $1,000 start compounded, 15% margin x 15x isolated, max 1 open pos, max_daily_stops=3,
     0% slippage, Maker 0.02% (limit entries & limit TPs) / Taker 0.10% (market exits),
     US Macro Shield (-30m..+15m block & -30m close uPnL<+0.25%), BTC 1H Veto 1.5%, BTC 4H trend filter,
     2-Step Smart BE, and 100% H1 Flip exit.
  3. Months 5 & 6 (Days 121-180) are executed ONCE blindly with zero modifications.

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
class RebuildExamSpec:
    track: str
    symbol: str
    wave: int
    min_wave_atr: float
    fib_entry: float
    ttl_bars: int
    abn_mult: float
    h1_adx_min: float
    tp_pct: float
    sl_pct: float
    be1_tr: float
    be1_off: float
    be2_tr: float
    be2_off: float
    no_counter_btc_4h: bool = True
    btc_1h_veto: float = 1.5
    rev_h1_flip: bool = True
    macro_shield: bool = True
    macro_protect_upnl_pct: float = 0.25
    max_daily_stops: int = 3
    leverage: float = 15.0
    margin_pct: float = 15.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0010
    slippage: float = 0.0


def prep_180d(df_coin: pd.DataFrame, df_btc: pd.DataFrame, sp: RebuildExamSpec) -> pd.DataFrame:
    entry_ms, h1_ms, h4_ms = INTERVAL_TO_MS["15"], INTERVAL_TO_MS["60"], INTERVAL_TO_MS["240"]
    df = df_coin.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    bar_r = df["high"] - df["low"]
    is_abn = (bar_r > sp.abn_mult * df["atr"]).fillna(False)
    df["vol_cd"] = is_abn | is_abn.shift(1, fill_value=False)

    btc = df_btc[["timestamp", "close"]].copy().sort_values("timestamp")
    btc["btc_r1h"] = btc["close"].pct_change(4) * 100.0
    btc["btc_r4h"] = btc["close"].pct_change(16) * 100.0
    btc_4h = BybitClient.resample_klines(df_btc, "240")
    b4_e20, b4_e50 = ema(btc_4h["close"], 20), ema(btc_4h["close"], 50)
    b4_out = pd.DataFrame({
        "close_ts": (btc_4h["timestamp"] + h4_ms).astype("int64"),
        "b4_bull": (btc_4h["close"] > b4_e50) & (b4_e20 > b4_e50),
        "b4_bear": (btc_4h["close"] < b4_e50) & (b4_e20 < b4_e50),
    }).sort_values("close_ts")

    df = pd.merge_asof(df, btc[["timestamp", "btc_r1h", "btc_r4h"]], on="timestamp", direction="backward")
    df = pd.merge_asof(df, b4_out, on="close_ts", direction="backward")
    df["btc_r1h"] = df["btc_r1h"].fillna(0.0)
    df["btc_r4h"] = df["btc_r4h"].fillna(0.0)
    df["b4_bull"] = df["b4_bull"].astype("boolean").fillna(False).astype(bool)
    df["b4_bear"] = df["b4_bear"].astype("boolean").fillna(False).astype(bool)

    cts = df["close_ts"].to_numpy(dtype=np.int64)
    m_wick = np.zeros(len(df), dtype=bool)
    m_prot = np.zeros(len(df), dtype=bool)
    for iso in US_MACRO_180D:
        ev = int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
        m_wick |= (cts >= ev - 30 * 60_000) & (cts <= ev + 15 * 60_000)
        m_prot |= (cts >= ev - 35 * 60_000) & (cts <= ev - 20 * 60_000)
    df["macro_wick"] = m_wick
    df["macro_prot"] = m_prot

    df_1h = BybitClient.resample_klines(df_coin, "60")
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
    n, w = len(df), sp.wave
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


def sim_6m(
    df: pd.DataFrame,
    sp: RebuildExamSpec,
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
    btc_r1h = df["btc_r1h"].to_numpy(dtype=np.float64)[m]
    b4_bull = df["b4_bull"].to_numpy(dtype=bool)[m]
    b4_bear = df["b4_bear"].to_numpy(dtype=bool)[m]
    macro_w = df["macro_wick"].to_numpy(dtype=bool)[m]
    macro_p = df["macro_prot"].to_numpy(dtype=bool)[m]

    mk, tk = sp.maker_fee, sp.taker_fee
    tp_f, sl_f = sp.tp_pct / 100.0, sp.sl_pct / 100.0
    be1_tr_f, be1_off_f = sp.be1_tr / 100.0, sp.be1_off / 100.0
    be2_tr_f, be2_off_f = sp.be2_tr / 100.0, sp.be2_off / 100.0

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
            cur_sl = pos["be2_p"] if pos["be2_on"] else (pos["be1_p"] if pos["be1_on"] else pos["sl_p"])
            cur_r = "BE2" if pos["be2_on"] else ("BE" if pos["be1_on"] else "SL")
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
                if (not pos["be1_on"]) and ((hi >= pos["be1_tr"]) if is_l else (lo <= pos["be1_tr"])):
                    pos["be1_on"] = True
                if be2_tr_f > 0 and (not pos["be2_on"]) and ((hi >= pos["be2_tr"]) if is_l else (lo <= pos["be2_tr"])):
                    pos["be2_on"] = True

            if (not closed) and pos is not None:
                upnl = ((cl - pos["ep"]) / pos["ep"] * 100.0) if is_l else ((pos["ep"] - cl) / pos["ep"] * 100.0)
                if sp.macro_shield and macro_p[i] and upnl < sp.macro_protect_upnl_pct:
                    _finish(pos, float(cl), "MACRO_PROT", False, i, ts)
                    pos = None
                elif sp.rev_h1_flip and (h1_bear[i] if is_l else h1_bull[i]):
                    _finish(pos, float(cl), "H1_FLIP", False, i, ts)
                    pos = None

        if pos is None and pend is not None:
            if i > pend["exp"] or halted or i < cd_until or (sp.macro_shield and macro_w[i]):
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
                        "efee": notional * mk, "be1_on": False, "be2_on": False,
                        "sl_p": ep * (1 - sl_f) if is_l else ep * (1 + sl_f),
                        "tp_p": ep * (1 + tp_f) if is_l else ep * (1 - tp_f),
                        "be1_tr": ep * (1 + be1_tr_f) if is_l else ep * (1 - be1_tr_f),
                        "be1_p": ep * (1 + be1_off_f) if is_l else ep * (1 - be1_off_f),
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
            if vol_cd[i] or (sp.macro_shield and macro_w[i]) or h1_adx[i] < sp.h1_adx_min or atr_pct[i] < 0.10 or atr_pct[i] > 1.80:
                continue
            rng = whi[i] - wlo[i]
            if not np.isfinite(rng) or rng < sp.min_wave_atr * atr[i]:
                continue
            if h1_bull[i] and iup[i]:
                if sp.btc_1h_veto > 0 and btc_r1h[i] <= -sp.btc_1h_veto:
                    continue
                if sp.no_counter_btc_4h and b4_bear[i]:
                    continue
                lp = whi[i] - sp.fib_entry * rng
                if lp < cl * 0.9992:
                    pend = {"dir": "LONG", "lp": float(lp), "exp": i + sp.ttl_bars}
            elif h1_bear[i] and idn[i]:
                if sp.btc_1h_veto > 0 and btc_r1h[i] >= sp.btc_1h_veto:
                    continue
                if sp.no_counter_btc_4h and b4_bull[i]:
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
    bl_tr = [t for t in trades if t["is_blind"]]
    sl_n = sum(1 for t in trades if t["reason"] == "SL")
    bl_sl = sum(1 for t in bl_tr if t["reason"] == "SL")
    wr = round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2)
    bl_wr = round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2)
    gp = sum(t["net"] for t in trades if t["net"] > 0)
    gl = abs(sum(t["net"] for t in trades if t["net"] < 0))

    return {
        "track": sp.track,
        "symbol": sp.symbol,
        "spec": asdict(sp),
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n,
        "trades_per_day": round(n / 180.0, 2),
        "wr_pct": wr,
        "blind_wr_pct": bl_wr,
        "sl_count": sl_n,
        "blind_sl": bl_sl,
        "pf": round(gp / max(gl, 1e-9), 2),
        "max_dd_pct": round(max_dd, 2),
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

    specs = [
        RebuildExamSpec(
            track="TRACK 1: REBUILT LTCUSDT",
            symbol="LTCUSDT",
            wave=12, min_wave_atr=2.2, fib_entry=0.618, ttl_bars=4, abn_mult=3.0,
            h1_adx_min=16.0, tp_pct=1.40, sl_pct=2.50,
            be1_tr=0.94, be1_off=0.35, be2_tr=1.12, be2_off=0.59,
            no_counter_btc_4h=True, btc_1h_veto=1.5,
        ),
        RebuildExamSpec(
            track="TRACK 2: BEST ALT COIN ONDOUSDT",
            symbol="ONDOUSDT",
            wave=20, min_wave_atr=2.6, fib_entry=0.618, ttl_bars=4, abn_mult=3.2,
            h1_adx_min=20.0, tp_pct=1.00, sl_pct=1.80,
            be1_tr=0.67, be1_off=0.25, be2_tr=0.80, be2_off=0.42,
            no_counter_btc_4h=True, btc_1h_veto=1.5,
        ),
    ]

    results = []
    print("\n" + "=" * 116)
    print("POLYGRAPH 6-MONTH EXAM: `coin2_15x_deep_rebuild_6m.py` (15x | SLIP 0% | MK 0.02% / TK 0.10% | US MACRO SHIELD)")
    print("=" * 116)
    for sp in specs:
        df_c = load_cached_15m(sp.symbol, cache_dir)
        df_full = prep_180d(df_c, df_btc, sp)
        r = sim_6m(df_full, sp, sim_start, sim_end, m_bounds, blind_split)
        results.append(r)
        mp = r["monthly_pct"]
        print(f"\n>>> {r['track']} ({r['symbol']})")
        print(
            f"  M1:{mp['M1']:>+6.1f}% / M2:{mp['M2']:>+6.1f}% / M3:{mp['M3']:>+6.1f}% / M4:{mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5:{mp['M5']:>+6.1f}% / M6:{mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого: $1 000 -> ${r['end_usd']:,.2f} | Сделок: {r['trades']} ({r['trades_per_day']:.2f}/день) | "
            f"WR: {r['wr_pct']:.1f}% (Blind WR: {r['blind_wr_pct']:.1f}%) | Стопов: {r['sl_count']} (слепых {r['blind_sl']}) | "
            f"PF: {r['pf']:.2f} | Макс. DD: {r['max_dd_pct']:.1f}% (слепой {r['blind_max_dd_pct']:.1f}%)"
        )

    Path("data/coin2_15x_deep_rebuild_6m_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/coin2_15x_deep_rebuild_6m_results.json")


if __name__ == "__main__":
    main()
