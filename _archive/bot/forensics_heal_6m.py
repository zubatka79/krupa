"""POLYGRAPH 6-MONTH EXAM (`bot/forensics_heal_6m.py`):
Full 6-Month Forensics & Healed Rebuild of the `WR 86.23%` Branch (`fibo_trend_v2`, H1 ADX>=25).

Rules locked STRICTLY from Days 1-120 forensics (Days 121-180 / M5-M6 completely unseen during design):
  - Base: Exact `WR 86.23%` branch (`H1 ADX>=25`, `H4 ADX>=20`, Fib `0.618-0.786`, wave 36,
    Maker `0.065%` TTL 2 bars, `TP 1.50% / SL 3.50% / Smart BE +0.975% -> +0.15%`,
    15% margin x 10x isolated, Liquid Top-25, real Bybit fees + 0.04% slippage).
  - Locked Healed Champion (`V4`):
    1. Catalyst / News Wave Gate: SKIP dead calm (`|BTC 24h| < 1.8%` AND `BTC 4h range < 1.8%`),
       trade ONLY in BTC 24h wave direction without opposing BTC 4H trend.
    2. US Macro Pre-News Shield: block entries `-30m..+15m` around FOMC/CPI/NFP/PPI/Fed Chair;
       at `-30m`, protect open trades (`>+0.3%` -> BE, else close at market).
    3. Stronger 1H Trend Filter: `H1 ADX >= 30` (eliminated 32% of Base SLs in weak 25..30 ADX).
    4. Slippage-Resilient Payoff: `TP 1.75% / SL 3.50% / Smart BE (+1.14% -> +0.25%)` so BE
       stays genuinely profitable after 0.075% fees + 0.04% slippage.

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

from bot.blind_final_tuning import build_tuning_bundle
from bot.blind_walkforward import load_cached_15m
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import ema

US_MACRO_180D = [
    "2026-06-05T12:30:00Z", "2026-06-10T12:30:00Z", "2026-06-11T12:30:00Z", "2026-06-17T18:00:00Z",
    "2026-07-02T12:30:00Z", "2026-07-14T12:30:00Z", "2026-07-15T12:30:00Z", "2026-07-29T18:00:00Z",
    "2026-08-07T12:30:00Z", "2026-08-12T12:30:00Z", "2026-08-13T12:30:00Z", "2026-08-21T14:00:00Z",
    "2026-09-04T12:30:00Z", "2026-09-11T12:30:00Z", "2026-09-14T12:30:00Z", "2026-09-16T18:00:00Z",
    "2026-10-02T12:30:00Z", "2026-10-14T12:30:00Z", "2026-10-15T12:30:00Z", "2026-10-28T18:00:00Z",
    "2026-11-06T13:30:00Z", "2026-11-10T13:30:00Z", "2026-11-12T13:30:00Z", "2026-11-16T17:00:00Z",
    "2026-11-18T19:00:00Z",
]


@dataclass
class HealSpec:
    name: str
    h1_adx_min: float = 25.0
    catalyst_filter: bool = False
    catalyst_thresh: float = 1.8
    pre_news_protect: bool = False
    tp_pct: float = 1.50
    sl_pct: float = 3.50
    smart_be: bool = True
    be_trigger_ratio: float = 0.65
    be_offset_pct: float = 0.15
    margin_pct: float = 15.0
    max_open: int = 5
    max_same_dir: int = 3
    max_daily_stops: int = 4


def build_btc_context_180d(df_btc: pd.DataFrame, timeline: np.ndarray) -> Dict[str, np.ndarray]:
    close_ts = timeline + 15 * 60_000
    n = len(timeline)
    macro_win = np.zeros(n, dtype=bool)
    macro_wick = np.zeros(n, dtype=bool)
    protect_30m = np.zeros(n, dtype=bool)
    for iso in US_MACRO_180D:
        ev = int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)
        macro_win |= (close_ts >= ev - 120 * 60_000) & (close_ts <= ev + 180 * 60_000)
        macro_wick |= (close_ts >= ev - 30 * 60_000) & (close_ts <= ev + 15 * 60_000)
        protect_30m |= (close_ts >= ev - 35 * 60_000) & (close_ts <= ev - 20 * 60_000)

    btc = df_btc[["timestamp", "high", "low", "close"]].copy().sort_values("timestamp")
    btc["close_ts"] = btc["timestamp"] + INTERVAL_TO_MS["15"]
    btc["r1h"] = btc["close"].pct_change(4) * 100.0
    btc["r24h"] = btc["close"].pct_change(96) * 100.0
    btc["vol4h"] = (btc["high"].rolling(16, min_periods=16).max() - btc["low"].rolling(16, min_periods=16).min()) / btc["close"] * 100.0

    btc_4h = BybitClient.resample_klines(df_btc, "240")
    b4_e20, b4_e50 = ema(btc_4h["close"], 20), ema(btc_4h["close"], 50)
    b4_df = pd.DataFrame({
        "close_ts": (btc_4h["timestamp"] + INTERVAL_TO_MS["240"]).astype("int64"),
        "b4_bull": (btc_4h["close"] > b4_e50) & (b4_e20 > b4_e50),
        "b4_bear": (btc_4h["close"] < b4_e50) & (b4_e20 < b4_e50),
    }).sort_values("close_ts")
    btc = pd.merge_asof(btc, b4_df, on="close_ts", direction="backward")
    al = pd.DataFrame({"timestamp": timeline}).merge(
        btc[["timestamp", "r1h", "r24h", "vol4h", "b4_bull", "b4_bear"]], on="timestamp", how="left"
    )
    return {
        "macro_win": macro_win,
        "macro_wick": macro_wick,
        "protect_30m": protect_30m,
        "r1h": al["r1h"].fillna(0.0).to_numpy(dtype=np.float64),
        "r24h": al["r24h"].fillna(0.0).to_numpy(dtype=np.float64),
        "vol4h": al["vol4h"].fillna(0.0).to_numpy(dtype=np.float64),
        "b4_bull": al["b4_bull"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "b4_bear": al["b4_bear"].astype("boolean").fillna(False).to_numpy(dtype=bool),
    }


def simulate_6m_spec(
    cfg: HealSpec,
    syms: List[str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    ctx: Dict[str, np.ndarray],
    timeline: np.ndarray,
    month_bounds: List[Tuple[str, int, int]],
    blind_split_ms: int,
) -> Dict[str, Any]:
    mk, tk, slip = 0.0002, 0.00055, 0.0004
    tp_f, sl_f, pb_f = cfg.tp_pct / 100.0, cfg.sl_pct / 100.0, 0.00065
    be_tr_f, be_off_f = tp_f * cfg.be_trigger_ratio, cfg.be_offset_pct / 100.0

    r24h, vol4h, b4_bull, b4_bear = ctx["r24h"], ctx["vol4h"], ctx["b4_bull"], ctx["b4_bear"]
    macro_wick, protect_30m, macro_win, r1h = ctx["macro_wick"], ctx["protect_30m"], ctx["macro_win"], ctx["r1h"]

    equity = 1000.0
    peak_eq = 1000.0
    max_dd = 0.0
    open_pos, pending, cd_until, trades = [], [], {}, []
    cur_day, day_sl, halted = -1, 0, False
    m_start, m_end = {}, {}

    def _close(p, trig_p, reason, is_mk, bar_i, ts):
        nonlocal equity, peak_eq, max_dd, day_sl, halted
        is_l = p["dir"] == "LONG"
        ex_p = trig_p if is_mk else (trig_p * (1 - slip) if is_l else trig_p * (1 + slip))
        gross = (ex_p - p["ep"]) * p["qty"] if is_l else (p["ep"] - ex_p) * p["qty"]
        net = max(-p["margin"], gross - p["efee"] - abs(p["qty"] * ex_p) * (mk if is_mk else tk))
        equity = max(1.0, equity + net)
        if reason == "SL":
            day_sl += 1
            if day_sl >= cfg.max_daily_stops: halted = True
        if equity > peak_eq: peak_eq = equity
        dd = (peak_eq - equity) / peak_eq * 100.0
        if dd > max_dd: max_dd = dd
        trades.append({
            "reason": reason, "net": net, "is_blind": ts >= blind_split_ms,
            "entry_macro": bool(macro_win[p["entry_i"]]) or bool(macro_win[bar_i]),
            "entry_calm": bool((abs(r24h[p["entry_i"]]) < 1.8) and (vol4h[p["entry_i"]] < 1.8)),
            "entry_counter_btc": bool((is_l and (r24h[p["entry_i"]] < 0 or b4_bear[p["entry_i"]])) or ((not is_l) and (r24h[p["entry_i"]] > 0 or b4_bull[p["entry_i"]]))),
            "exit_btc_shock": bool(abs(r1h[bar_i]) >= 1.5),
            "weak_adx": bool(p["h1_adx"] < 30.0),
        })
        cd_until[p["sym"]] = bar_i + 3

    for i in range(len(timeline)):
        ts = int(timeline[i])
        for mn, ms, me in month_bounds:
            if mn not in m_start and ts >= ms: m_start[mn] = equity
            if ts <= me: m_end[mn] = equity
        dk = ts // 86_400_000
        if dk != cur_day: cur_day, day_sl, halted = dk, 0, False

        if open_pos:
            still = []
            for p in open_pos:
                sd = bundle[p["sym"]]
                hi, lo, cl = sd["high"][i], sd["low"][i], sd["close"][i]
                if not np.isfinite(lo): still.append(p); continue
                is_l = p["dir"] == "LONG"
                if cfg.smart_be and p["be_on"]:
                    if (lo <= p["be_p"]) if is_l else (hi >= p["be_p"]): _close(p, p["be_p"], "BE", False, i, ts); continue
                    if (hi >= p["tp_p"]) if is_l else (lo <= p["tp_p"]): _close(p, p["tp_p"], "TP", True, i, ts); continue
                else:
                    if (lo <= p["sl_p"]) if is_l else (hi >= p["sl_p"]): _close(p, p["sl_p"], "SL", False, i, ts); continue
                    if (hi >= p["tp_p"]) if is_l else (lo <= p["tp_p"]): _close(p, p["tp_p"], "TP", True, i, ts); continue
                    if cfg.smart_be and ((hi >= p["be_tr"]) if is_l else (lo <= p["be_tr"])): p["be_on"] = True
                if cfg.pre_news_protect and protect_30m[i]:
                    upnl = ((cl - p["ep"]) / p["ep"] * 100.0) if is_l else ((p["ep"] - cl) / p["ep"] * 100.0)
                    if upnl > 0.30: p["be_on"] = True; still.append(p)
                    else: _close(p, float(cl), "PRE_NEWS", False, i, ts)
                    continue
                still.append(p)
            open_pos = still

        has_cat = (not cfg.catalyst_filter) or ((abs(r24h[i]) >= cfg.catalyst_thresh) or (vol4h[i] >= cfg.catalyst_thresh))
        can_L = (not (cfg.pre_news_protect and macro_wick[i])) and has_cat and ((not cfg.catalyst_filter) or ((r24h[i] > 0) and (not b4_bear[i])))
        can_S = (not (cfg.pre_news_protect and macro_wick[i])) and has_cat and ((not cfg.catalyst_filter) or ((r24h[i] < 0) and (not b4_bull[i])))

        if pending:
            npend = []
            for o in pending:
                if i > o["exp"] or halted: continue
                if (o["dir"] == "LONG" and not can_L) or (o["dir"] == "SHORT" and not can_S): continue
                if any(p["sym"] == o["sym"] for p in open_pos) or i < cd_until.get(o["sym"], -1): continue
                sd = bundle[o["sym"]]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    if i < o["exp"]: npend.append(o); continue
                is_l = o["dir"] == "LONG"
                if not ((lo < o["lp"]) if is_l else (hi > o["lp"])):
                    if i < o["exp"]: npend.append(o); continue
                if len(open_pos) >= cfg.max_open or sum(1 for p in open_pos if p["dir"] == o["dir"]) >= cfg.max_same_dir: continue
                margin = equity * (cfg.margin_pct / 100.0)
                if sum(p["margin"] for p in open_pos) + margin > equity * 0.95: continue
                notional = margin * 10.0
                ep = o["lp"]
                pos = {
                    "sym": o["sym"], "dir": o["dir"], "ep": ep, "qty": notional / ep, "margin": margin,
                    "efee": notional * mk, "be_on": False, "entry_i": i, "h1_adx": float(sd["h1_adx"][i]),
                    "sl_p": ep * (1 - sl_f) if is_l else ep * (1 + sl_f),
                    "tp_p": ep * (1 + tp_f) if is_l else ep * (1 - tp_f),
                    "be_tr": ep * (1 + be_tr_f) if is_l else ep * (1 - be_tr_f),
                    "be_p": ep * (1 + be_off_f) if is_l else ep * (1 - be_off_f),
                }
                if (lo <= pos["sl_p"]) if is_l else (hi >= pos["sl_p"]): _close(pos, pos["sl_p"], "SL", False, i, ts)
                else: open_pos.append(pos)
            pending = npend

        if halted or len(open_pos) >= cfg.max_open or (not can_L and not can_S): continue
        busy = {p["sym"] for p in open_pos} | {o["sym"] for o in pending}
        cands = []
        for sym in syms:
            if sym in busy or i < cd_until.get(sym, -1): continue
            sd = bundle[sym]
            if sd["h1_adx"][i] < cfg.h1_adx_min or sd["h4_adx"][i] < 20.0 or sd["atr_pct"][i] > 1.50: continue
            if can_L and sd["raw_L"][i]: cands.append((float(sd["q"][i]), sym, "LONG", float(sd["close"][i])))
            elif can_S and sd["raw_S"][i]: cands.append((float(sd["q"][i]), sym, "SHORT", float(sd["close"][i])))
        cands.sort(key=lambda x: x[0], reverse=True)
        for _, sym, d, cl in cands:
            if len(open_pos) + len(pending) >= cfg.max_open + 3: break
            if sum(1 for p in open_pos if p["dir"] == d) >= cfg.max_same_dir: continue
            lp = cl * (1 - pb_f) if d == "LONG" else cl * (1 + pb_f)
            pending.append({"sym": sym, "dir": d, "lp": lp, "exp": i + 2})

    if open_pos:
        last_ts = int(timeline[-1])
        for p in open_pos:
            cl = bundle[p["sym"]]["close"][-1]
            _close(p, float(cl) if np.isfinite(cl) else p["ep"], "EOD", False, len(timeline) - 1, last_ts)
        for mn, _, me in month_bounds:
            if last_ts <= me: m_end[mn] = equity

    mp = {mn: round((m_end.get(mn, 1000.0) - m_start.get(mn, 1000.0)) / max(m_start.get(mn, 1000.0), 1.0) * 100.0, 2) for mn, _, _ in month_bounds}
    n = len(trades)
    sl_tr = [t for t in trades if t["reason"] == "SL"]
    sl_n = len(sl_tr)
    wr = round(sum(1 for t in trades if t["net"] > 0) / max(n, 1) * 100.0, 2)
    bl_tr = [t for t in trades if t["is_blind"]]
    bl_wr = round(sum(1 for t in bl_tr if t["net"] > 0) / max(len(bl_tr), 1) * 100.0, 2)

    forensics = {
        "total_sl": sl_n,
        "sl_in_calm": sum(t["entry_calm"] for t in sl_tr),
        "sl_counter_btc": sum(t["entry_counter_btc"] for t in sl_tr),
        "sl_in_macro": sum(t["entry_macro"] for t in sl_tr),
        "sl_on_btc_shock": sum(t["exit_btc_shock"] for t in sl_tr),
        "sl_weak_adx_25_30": sum(t["weak_adx"] for t in sl_tr),
    }

    return {
        "name": cfg.name,
        "spec": asdict(cfg),
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n,
        "wr_pct": wr,
        "blind_wr_pct": bl_wr,
        "sl_count": sl_n,
        "tp_count": sum(1 for t in trades if t["reason"] == "TP"),
        "be_count": sum(1 for t in trades if t["reason"] == "BE"),
        "pre_news_count": sum(1 for t in trades if t["reason"] == "PRE_NEWS"),
        "max_dd_pct": round(max_dd, 2),
        "sl_forensics": forensics,
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    top25 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"][:25]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    day_ms = 86_400_000
    sim_end = int(df_btc["timestamp"].iloc[-1])
    sim_start = sim_end - 180 * day_ms
    blind_split = sim_start + 120 * day_ms

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start) & (df_btc["timestamp"] <= sim_end)]
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

    ctx = build_btc_context_180d(df_btc, tl)
    bundle = {sym: build_tuning_bundle(load_cached_15m(sym, cache_dir), df_btc, tl_df) for sym in top25}

    specs = [
        HealSpec("BASE (WR 86.23% ветка: H1 ADX>=25, TP1.50/SL3.50/BE+0.15)"),
        HealSpec(
            "LOCKED CHAMPION (V4: Catalyst 1.8% + PreNews30m + H1 ADX>=30 + TP1.75/SL3.50/BE+0.25)",
            h1_adx_min=30.0, catalyst_filter=True, pre_news_protect=True, tp_pct=1.75, be_offset_pct=0.25,
        ),
        HealSpec(
            "ALT-1 (V1: Catalyst 1.8% + PreNews30m + H1 ADX>=25, TP1.50/SL3.50/BE+0.15)",
            h1_adx_min=25.0, catalyst_filter=True, pre_news_protect=True,
        ),
        HealSpec(
            "ALT-2 (V5: Catalyst 1.8% + PreNews30m + H1 ADX>=25, TP1.75/SL2.80/NoBE)",
            h1_adx_min=25.0, catalyst_filter=True, pre_news_protect=True, tp_pct=1.75, sl_pct=2.80, smart_be=False,
        ),
    ]

    results = []
    print("\n" + "=" * 114)
    print("POLYGRAPH 6-MONTH EXAM: `forensics_heal_6m.py` (15% x 10x ISOLATED | BYBIT FEES + 0.04% SLIPPAGE)")
    print("=" * 114)
    for sp in specs:
        r = simulate_6m_spec(sp, top25, bundle, ctx, tl, m_bounds, blind_split)
        results.append(r)
        mp = r["monthly_pct"]
        f = r["sl_forensics"]
        print(f"\n>>> {r['name']}")
        print(
            f"  M1:{mp['M1']:>+6.1f}% / M2:{mp['M2']:>+6.1f}% / M3:{mp['M3']:>+6.1f}% / M4:{mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5:{mp['M5']:>+6.1f}% / M6:{mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого: $1 000 -> ${r['end_usd']:,.2f} | Сделок: {r['trades']} | WR: {r['wr_pct']:.1f}% (Blind WR: {r['blind_wr_pct']:.1f}%) | "
            f"Стопов: {r['sl_count']} | Макс. DD: {r['max_dd_pct']:.1f}%"
        )
        if sp.catalyst_filter is False:
            sl_n = max(f["total_sl"], 1)
            print(
                f"  6m SL Forensics ({f['total_sl']} SLs): Calm={f['sl_in_calm']} ({f['sl_in_calm']/sl_n*100:.1f}%) | "
                f"CounterBTC={f['sl_counter_btc']} ({f['sl_counter_btc']/sl_n*100:.1f}%) | "
                f"WeakADX25-30={f['sl_weak_adx_25_30']} ({f['sl_weak_adx_25_30']/sl_n*100:.1f}%) | "
                f"BTCShock1h={f['sl_on_btc_shock']} ({f['sl_on_btc_shock']/sl_n*100:.1f}%) | "
                f"USMacro={f['sl_in_macro']} ({f['sl_in_macro']/sl_n*100:.1f}%)"
            )

    Path("data/forensics_heal_6m_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/forensics_heal_6m_results.json")


if __name__ == "__main__":
    main()
