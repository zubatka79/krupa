"""POLYGRAPH 6-MONTH EXAM (`bot/coin2_serious_hunt_6m.py`):
Serious multi-paradigm hunt across all non-excluded cached coins (excluding ADA, ONDO, LTC, 1000PEPE, ZRO).

HONESTY PROTOCOL:
  1. Multi-family / multi-paradigm search (Trend Pullback, Positive R:R Trail, SFP Sweep, Keltner/ATR Mean-Reversion)
     and full parameter/plateau locking were performed STRICTLY on Days 1-120 (M1-M4) with Days 121-180 excluded.
  2. Locked Primary Candidate (#1): `WLFIUSDT` — 15M Zero-Lookahead Liquidity-Sweep Mean-Reversion
     (15M corr vs ADA = 0.39; 18/18 = 100% 4/4-green neighbors across two 3x3 plateaus on Days 1-120;
      Days 1-120: $7,397.19, PF 2.17, WR 86.3%, 24 SL, DD 16.7%, L/S +$3,603/+$2,794).
  3. Locked Backup Candidate (#2): `RENDERUSDT` — 15M Mean-Reversion + Time-Stop (9/9 plateau on Days 1-120).
  4. Months 5 & 6 (Days 121-180) are executed ONCE strictly blindly with zero modifications, plus full SL forensics.

Execution: $1,000 start compounded, 15% margin x 15x isolated, max 1 open pos, max_daily_stops=3,
0% slippage, Maker 0.02% / Taker 0.10%, US Macro Shield (-30m..+15m & close uPnL<+0.25% at -30m).
Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
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


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_g = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_l = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_g / avg_l.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


@dataclass
class MeanRevExamSpec:
    track: str
    symbol: str
    ema_len: int
    k_band: float
    h1_adx_max: float
    rsi_lim: float
    btc_1h_veto: float
    tp_mode: str            # "entry_ema" (fixed at signal bar EMA -> 100% zero lookahead)
    fallback_tp_pct: float
    sl_pct: float
    ttl_bars: int = 3
    be1_ratio: float = 0.65
    be1_lock_ratio: float = 0.22
    be2_ratio: float = 0.82
    be2_lock_ratio: float = 0.42
    time_stop_bars: int = 0
    use_vol_cd: bool = False
    abn_mult: float = 3.2
    macro_shield: bool = True
    macro_protect_upnl_pct: float = 0.25
    max_daily_stops: int = 3
    leverage: float = 15.0
    margin_pct: float = 15.0
    maker_fee: float = 0.0002
    taker_fee: float = 0.0010
    slippage: float = 0.0


def prep_180d(df_coin: pd.DataFrame, df_btc: pd.DataFrame, sp: MeanRevExamSpec) -> pd.DataFrame:
    entry_ms, h1_ms, h4_ms = INTERVAL_TO_MS["15"], INTERVAL_TO_MS["60"], INTERVAL_TO_MS["240"]
    df = df_coin.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    df["ema_mid"] = ema(df["close"], sp.ema_len)
    df["rsi14"] = compute_rsi(df["close"], 14)

    bar_r = df["high"] - df["low"]
    is_abn = (bar_r > sp.abn_mult * df["atr"]).fillna(False)
    df["vol_cd"] = is_abn | is_abn.shift(1, fill_value=False)

    btc = df_btc[["timestamp", "close"]].copy().sort_values("timestamp")
    btc["btc_r15m"] = btc["close"].pct_change(1) * 100.0
    btc["btc_r1h"] = btc["close"].pct_change(4) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_r15m", "btc_r1h"]], on="timestamp", direction="backward")
    df["btc_r15m"] = df["btc_r15m"].fillna(0.0)
    df["btc_r1h"] = df["btc_r1h"].fillna(0.0)

    cts = df["close_ts"].to_numpy(dtype=np.int64)
    m_wick = np.zeros(len(df), dtype=bool)
    m_prot = np.zeros(len(df), dtype=bool)
    min_mdist = np.full(len(df), 99999.0)
    for iso in US_MACRO_180D:
        ev = int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
        m_wick |= (cts >= ev - 30 * 60_000) & (cts <= ev + 15 * 60_000)
        m_prot |= (cts >= ev - 35 * 60_000) & (cts <= ev - 20 * 60_000)
        min_mdist = np.minimum(min_mdist, np.abs(cts - ev) / 60_000.0)
    df["macro_wick"] = m_wick
    df["macro_prot"] = m_prot
    df["macro_dist_min"] = min_mdist

    df_1h = BybitClient.resample_klines(df_coin, "60")
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out = pd.DataFrame({
        "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
        "h1_adx": h1_adx.fillna(0.0).values,
    }).sort_values("close_ts")
    df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
    df["h1_adx"] = df["h1_adx"].fillna(0.0)

    df_4h = BybitClient.resample_klines(df_coin, "240")
    h4_e20, h4_e50 = ema(df_4h["close"], 20), ema(df_4h["close"], 50)
    h4_out = pd.DataFrame({
        "close_ts": (df_4h["timestamp"] + h4_ms).astype("int64"),
        "h4_bull": (df_4h["close"] > h4_e50) & (h4_e20 > h4_e50),
        "h4_bear": (df_4h["close"] < h4_e50) & (h4_e20 < h4_e50),
    }).sort_values("close_ts")
    df = pd.merge_asof(df, h4_out, on="close_ts", direction="backward")
    df["h4_bull"] = df["h4_bull"].astype("boolean").fillna(False).astype(bool)
    df["h4_bear"] = df["h4_bear"].astype("boolean").fillna(False).astype(bool)
    return df


def sim_6m(
    df: pd.DataFrame,
    sp: MeanRevExamSpec,
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
    atr = df["atr"].to_numpy(dtype=np.float64)[m]
    atr_pct = df["atr_pct"].to_numpy(dtype=np.float64)[m]
    ema_mid = df["ema_mid"].to_numpy(dtype=np.float64)[m]
    rsi14 = df["rsi14"].to_numpy(dtype=np.float64)[m]
    vol_cd = df["vol_cd"].to_numpy(dtype=bool)[m]
    h1_adx = df["h1_adx"].to_numpy(dtype=np.float64)[m]
    h4_bull = df["h4_bull"].to_numpy(dtype=bool)[m]
    h4_bear = df["h4_bear"].to_numpy(dtype=bool)[m]
    btc_r15m = df["btc_r15m"].to_numpy(dtype=np.float64)[m]
    btc_r1h = df["btc_r1h"].to_numpy(dtype=np.float64)[m]
    macro_w = df["macro_wick"].to_numpy(dtype=bool)[m]
    macro_p = df["macro_prot"].to_numpy(dtype=bool)[m]
    macro_dist = df["macro_dist_min"].to_numpy(dtype=np.float64)[m]

    mk, tk = sp.maker_fee, sp.taker_fee
    tp_f, sl_f = sp.fallback_tp_pct / 100.0, sp.sl_pct / 100.0

    equity, peak, max_dd = 1000.0, 1000.0, 0.0
    blind_peak, blind_max_dd = None, 0.0
    pos, pend = None, None
    cd_until = -1
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end, trades = {}, {}, []
    n_bars = len(ts_arr)

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
            "dir": p["dir"],
            "is_blind": bool(ts >= blind_split_ms),
            "bars_held": bar_i - p["eb"],
            "mfe_ratio_tp": round(p["mfe_pct"] / max(p["eff_tp_pct"], 0.1), 3),
            "min_macro_dist_hold": round(p["min_mdist"], 1),
            "passed_mprot_kept": bool(p["passed_mprot"]),
            "btc_1h_opp": bool((btc_r1h[bar_i] <= -0.8) if is_l else (btc_r1h[bar_i] >= 0.8)),
            "btc_15m_opp": bool((btc_r15m[bar_i] <= -0.5) if is_l else (btc_r15m[bar_i] >= 0.5)),
            "against_4h_trend": bool(h4_bear[bar_i] if is_l else h4_bull[bar_i]),
            "h1_adx_breakout": bool(h1_adx[bar_i] >= 35.0),
            "entry_atr_pct": round(p["e_atrp"], 3),
        })
        cd_until = bar_i + 1

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
            bmfe = ((hi - pos["ep"]) / pos["ep"] * 100.0) if is_l else ((pos["ep"] - lo) / pos["ep"] * 100.0)
            if bmfe > pos["mfe_pct"]:
                pos["mfe_pct"] = bmfe
            if macro_dist[i] < pos["min_mdist"]:
                pos["min_mdist"] = float(macro_dist[i])

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
                if (not pos["be2_on"]) and ((hi >= pos["be2_tr"]) if is_l else (lo <= pos["be2_tr"])):
                    pos["be2_on"] = True

            if (not closed) and pos is not None:
                upnl = ((cl - pos["ep"]) / pos["ep"] * 100.0) if is_l else ((pos["ep"] - cl) / pos["ep"] * 100.0)
                if sp.macro_shield and macro_p[i]:
                    if upnl < sp.macro_protect_upnl_pct:
                        _finish(pos, float(cl), "MACRO_PROT", False, i, ts)
                        pos = None
                    else:
                        pos["passed_mprot"] = True
                elif sp.time_stop_bars > 0 and (i - pos["eb"]) >= sp.time_stop_bars and upnl < 0.0:
                    _finish(pos, float(cl), "TIME_STOP", False, i, ts)
                    pos = None

        if pos is None and pend is not None:
            if i > pend["exp"] or halted or i < cd_until or (sp.macro_shield and macro_w[i]):
                pend = None
            else:
                is_l = pend["dir"] == "LONG"
                lp = pend["lp"]
                filled = (lo < lp) if is_l else (hi > lp)
                if filled:
                    margin = equity * (sp.margin_pct / 100.0)
                    notional = margin * sp.leverage
                    sig_em = pend["sig_em"]
                    if sp.tp_mode == "entry_ema" and ((sig_em >= lp * 1.006) if is_l else (sig_em <= lp * 0.994)):
                        tp_p = sig_em
                        eff_tp = abs(tp_p - lp) / lp * 100.0
                    else:
                        tp_p = lp * (1.0 + tp_f) if is_l else lp * (1.0 - tp_f)
                        eff_tp = sp.fallback_tp_pct
                    b1_tr_f = (eff_tp * sp.be1_ratio) / 100.0
                    b1_o_f = (eff_tp * sp.be1_lock_ratio) / 100.0
                    b2_tr_f = (eff_tp * sp.be2_ratio) / 100.0
                    b2_o_f = (eff_tp * sp.be2_lock_ratio) / 100.0
                    p_new = {
                        "dir": pend["dir"], "ep": lp, "qty": notional / lp, "margin": margin,
                        "efee": notional * mk, "be1_on": False, "be2_on": False,
                        "sl_p": lp * (1 - sl_f) if is_l else lp * (1 + sl_f),
                        "tp_p": tp_p, "eff_tp_pct": eff_tp,
                        "be1_tr": lp * (1 + b1_tr_f) if is_l else lp * (1 - b1_tr_f),
                        "be1_p": lp * (1 + b1_o_f) if is_l else lp * (1 - b1_o_f),
                        "be2_tr": lp * (1 + b2_tr_f) if is_l else lp * (1 - b2_tr_f),
                        "be2_p": lp * (1 + b2_o_f) if is_l else lp * (1 - b2_o_f),
                        "eb": i, "mfe_pct": 0.0, "min_mdist": float(macro_dist[i]),
                        "passed_mprot": False, "e_atrp": float(atr_pct[i]),
                    }
                    pend = None
                    if (lo <= p_new["sl_p"]) if is_l else (hi >= p_new["sl_p"]):
                        _finish(p_new, p_new["sl_p"], "SL", False, i, ts)
                    else:
                        pos = p_new
                elif i >= pend["exp"]:
                    pend = None

        if pos is None and pend is None and (not halted) and i >= cd_until:
            if (sp.use_vol_cd and vol_cd[i]) or (sp.macro_shield and macro_w[i]) or np.isnan(atr[i]) or atr[i] <= 0.0:
                continue
            if h1_adx[i] > sp.h1_adx_max or atr_pct[i] < 0.10 or atr_pct[i] > 1.80:
                continue
            em = ema_mid[i]
            low_b = em - sp.k_band * atr[i]
            hi_b = em + sp.k_band * atr[i]
            if cl < em and rsi14[i] <= sp.rsi_lim and btc_r1h[i] > -sp.btc_1h_veto:
                if 0.0 < low_b < cl * 0.9992:
                    pend = {"dir": "LONG", "lp": float(low_b), "exp": i + sp.ttl_bars, "sig_em": float(em)}
            elif cl > em and rsi14[i] >= (100.0 - sp.rsi_lim) and btc_r1h[i] < sp.btc_1h_veto:
                if hi_b > cl * 1.0008:
                    pend = {"dir": "SHORT", "lp": float(hi_b), "exp": i + sp.ttl_bars, "sig_em": float(em)}

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
    l_net = sum(t["net"] for t in trades if t["dir"] == "LONG")
    s_net = sum(t["net"] for t in trades if t["dir"] == "SHORT")

    def _sl_forensics(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        sls = [t for t in sub if t["reason"] == "SL"]
        return {
            "total_sl": len(sls),
            "long_sl": sum(1 for t in sls if t["dir"] == "LONG"),
            "short_sl": sum(1 for t in sls if t["dir"] == "SHORT"),
            "macro_prot_exits": sum(1 for t in sub if t["reason"] == "MACRO_PROT"),
            "macro_within_60m": sum(1 for t in sls if t["min_macro_dist_hold"] <= 60.0),
            "macro_within_120m": sum(1 for t in sls if t["min_macro_dist_hold"] <= 120.0),
            "passed_mprot_then_sl": sum(1 for t in sls if t["passed_mprot_kept"]),
            "counter_btc_1h_08": sum(1 for t in sls if t["btc_1h_opp"]),
            "counter_btc_15m_05": sum(1 for t in sls if t["btc_15m_opp"]),
            "against_4h_trend": sum(1 for t in sls if t["against_4h_trend"]),
            "h1_adx_breakout_35": sum(1 for t in sls if t["h1_adx_breakout"]),
            "near_be_mfe_45pct_tp": sum(1 for t in sls if t["mfe_ratio_tp"] >= 0.45),
            "fast_sl_le_2bars": sum(1 for t in sls if t["bars_held"] <= 2),
        }

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
        "long_net_usd": round(l_net, 2),
        "short_net_usd": round(s_net, 2),
        "exit_reasons": dict(Counter(t["reason"] for t in trades)),
        "forensics": {
            "all_6m": _sl_forensics(trades),
            "dev_m1_m4": _sl_forensics(dev_tr),
            "blind_m5_m6": _sl_forensics(bl_tr),
        },
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
        MeanRevExamSpec(
            track="PRIMARY #1: WLFIUSDT (15M ZERO-LOOKAHEAD MEAN-REVERSION, LOCKED 1-120D, 18/18 PLATEAU)",
            symbol="WLFIUSDT",
            ema_len=20, k_band=3.2, h1_adx_max=40.0, rsi_lim=40.0, btc_1h_veto=1.0,
            tp_mode="entry_ema", fallback_tp_pct=1.40, sl_pct=2.50, ttl_bars=3,
            be1_ratio=0.65, be1_lock_ratio=0.22, be2_ratio=0.82, be2_lock_ratio=0.42,
            time_stop_bars=0, use_vol_cd=False,
        ),
        MeanRevExamSpec(
            track="BACKUP #2: RENDERUSDT (15M ZERO-LOOKAHEAD MEAN-REVERSION + TIME-STOP, LOCKED 1-120D)",
            symbol="RENDERUSDT",
            ema_len=16, k_band=3.0, h1_adx_max=40.0, rsi_lim=45.0, btc_1h_veto=1.2,
            tp_mode="entry_ema", fallback_tp_pct=1.40, sl_pct=2.10, ttl_bars=3,
            be1_ratio=0.65, be1_lock_ratio=0.22, be2_ratio=0.82, be2_lock_ratio=0.42,
            time_stop_bars=12, use_vol_cd=True,
        ),
    ]

    results = []
    print("\n" + "=" * 122)
    print("POLYGRAPH 6-MONTH EXAM: `coin2_serious_hunt_6m.py` (15x | SLIP 0% | MK 0.02% / TK 0.10% | US MACRO SHIELD)")
    print("=" * 122)
    for sp in specs:
        df_c = load_cached_15m(sp.symbol, cache_dir)
        df_full = prep_180d(df_c, df_btc, sp)
        r = sim_6m(df_full, sp, sim_start, sim_end, m_bounds, blind_split)
        results.append(r)
        mp = r["monthly_pct"]
        print(f"\n>>> {r['track']}")
        print(
            f"  M1:{mp['M1']:>+6.1f}% / M2:{mp['M2']:>+6.1f}% / M3:{mp['M3']:>+6.1f}% / M4:{mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5:{mp['M5']:>+6.1f}% / M6:{mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого: $1 000 -> ${r['end_usd']:,.2f} | Сделок: {r['trades']} ({r['trades_per_day']:.2f}/день) | "
            f"WR: {r['wr_pct']:.1f}% (Blind WR: {r['blind_wr_pct']:.1f}%) | PF: {r['pf']:.2f} | "
            f"Стопов: {r['sl_count']} (слепых {r['blind_sl']}) | Макс. DD: {r['max_dd_pct']:.1f}% (слепой {r['blind_max_dd_pct']:.1f}%)"
        )
        print(f"  L/S Net: LONG ${r['long_net_usd']:+,.2f} / SHORT ${r['short_net_usd']:+,.2f} | Выходы: {r['exit_reasons']}")
        print("  --- SL FORENSICS (M1-M4 vs BLIND M5-M6) ---")
        for k, d in r["forensics"].items():
            print(f"    {k:12s}: {d}")

    out_path = Path("data/coin2_serious_hunt_6m_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to {out_path}")


if __name__ == "__main__":
    main()
