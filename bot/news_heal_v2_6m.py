"""POLYGRAPH 6-MONTH EXAM (`bot/news_heal_v2_6m.py`): News Healing of `fibo_trend_v2` (Variant B).

Compares 3 configurations across all 180 days (M1-M4 Dev || M5-M6 STRICTLY BLIND):
  1. Base Variant B (no news layer): Liquid Top-25, H1 ADX>=30, H4 ADX>=20, Fib 0.618-0.786,
     Maker 0.065%, TP 1.50% / SL 3.50% / Smart BE (+0.975% -> +0.15%), 15% margin x 10x isolated,
     max 5 open (3 same dir), 4 SL/day, real Bybit fees + 0.04% slippage on market/SL/BE exits.
  2. Shield-Pause (safe): Wide USD High macro window (-2h .. +3h), pre-news protection at -30m
     (close if PnL <= +0.3%, move SL to BE +0.15% if PnL > +0.3%), 6h pause on BTC shock
     (>2.0% in 1h or >1.2% in 15m) OR after 2 SLs in a UTC day.
  3. News Short / Reversal: Rule derived strictly on Days 1-120 (BTC 15m dump <= -1.2% or
     bearish macro drop <= -0.6%): forbids LONGs for 6h and opens shock SHORTs on weak alts
     below VWAP & 1H EMA50 with identical 10x/TP/SL/Smart BE/fees/slippage.

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
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
from bot.indicators import compute_daily_vwap, ema

# Official 2026 US High-Impact Macro Releases (FOMC, CPI, NFP, PPI) across May 23 - Nov 19, 2026
US_HIGH_MACRO_2026_UTC = [
    ("2026-06-05T12:30:00Z", "US NFP"),
    ("2026-06-10T12:30:00Z", "US CPI"),
    ("2026-06-11T12:30:00Z", "US PPI"),
    ("2026-06-17T18:00:00Z", "FOMC"),
    ("2026-07-02T12:30:00Z", "US NFP"),
    ("2026-07-14T12:30:00Z", "US CPI"),
    ("2026-07-15T12:30:00Z", "US PPI"),
    ("2026-07-29T18:00:00Z", "FOMC"),
    ("2026-08-07T12:30:00Z", "US NFP"),
    ("2026-08-12T12:30:00Z", "US CPI"),
    ("2026-08-13T12:30:00Z", "US PPI"),
    ("2026-09-04T12:30:00Z", "US NFP"),
    ("2026-09-11T12:30:00Z", "US CPI"),
    ("2026-09-14T12:30:00Z", "US PPI"),
    ("2026-09-16T18:00:00Z", "FOMC"),
    ("2026-10-02T12:30:00Z", "US NFP"),
    ("2026-10-14T12:30:00Z", "US CPI"),
    ("2026-10-15T12:30:00Z", "US PPI"),
    ("2026-10-28T18:00:00Z", "FOMC"),
    ("2026-11-06T13:30:00Z", "US NFP"),
    ("2026-11-10T13:30:00Z", "US CPI"),
    ("2026-11-12T13:30:00Z", "US PPI"),
]


@dataclass
class _Pending:
    symbol: str
    direction: str
    limit_price: float
    expires_idx: int


@dataclass
class _OpenPos:
    symbol: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    be_trigger_price: float
    be_stop_price: float
    qty: float
    margin: float
    entry_fee: float
    be_active: bool = False


def build_macro_and_btc_masks(df_btc_15m: pd.DataFrame, timeline: np.ndarray) -> Dict[str, np.ndarray]:
    close_ts = timeline + 15 * 60_000
    n = len(timeline)

    macro_block = np.zeros(n, dtype=bool)     # -2h before .. +3h after USD High
    pre_news_30m = np.zeros(n, dtype=bool)    # exact -30m bar before USD High (protects open trades)
    macro_release = np.zeros(n, dtype=bool)   # 0..+1h release window for news-short check

    for iso_ts, _ in US_HIGH_MACRO_2026_UTC:
        ev_ms = int(datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).timestamp() * 1000)
        macro_block |= (close_ts >= ev_ms - 120 * 60_000) & (close_ts <= ev_ms + 180 * 60_000)
        pre_news_30m |= (close_ts >= ev_ms - 35 * 60_000) & (close_ts <= ev_ms - 20 * 60_000)
        macro_release |= (close_ts >= ev_ms) & (close_ts <= ev_ms + 60 * 60_000)

    btc = df_btc_15m[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc["r15"] = (btc["close"] - btc["open"]) / btc["open"].replace(0.0, np.nan) * 100.0
    btc["r1h"] = btc["close"].pct_change(4) * 100.0
    al = pd.DataFrame({"timestamp": timeline}).merge(btc[["timestamp", "r15", "r1h"]], on="timestamp", how="left")
    r15 = al["r15"].fillna(0.0).to_numpy(dtype=np.float64)
    r1h = al["r1h"].fillna(0.0).to_numpy(dtype=np.float64)

    btc_shock_any = (np.abs(r1h) > 2.0) | (np.abs(r15) > 1.2)
    btc_bear_shock = (r15 <= -1.2) | (macro_release & (r15 <= -0.6))

    return {
        "macro_block": macro_block,
        "pre_news_30m": pre_news_30m,
        "btc_shock_any": btc_shock_any,
        "btc_bear_shock": btc_bear_shock,
    }


def simulate_6m_variant(
    mode: str,  # "base", "shield_pause", "news_short"
    symbols: List[str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    weak_short_ok: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    timeline: np.ndarray,
    month_bounds: List[Tuple[str, int, int]],
    initial_deposit: float = 1000.0,
) -> Dict[str, Any]:
    maker_rate = 0.0002
    taker_rate = 0.00055
    slip = 0.0004  # 0.04% realistic slippage on ALL market entries, SL, BE, and pre-news market closes!

    tp_frac = 0.015
    sl_frac = 0.035
    pb_frac = 0.00065
    be_trig_frac = tp_frac * 0.65
    be_off_frac = 0.0015
    margin_pct = 15.0
    leverage = 10.0
    max_open = 5
    max_same_dir = 3
    max_daily_stops = 4

    use_shield = mode == "shield_pause"
    use_news_short = mode == "news_short"

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_pct = 0.0

    open_pos: List[_OpenPos] = []
    pending: List[_Pending] = []
    cd_until: Dict[str, int] = {}
    trades: List[Tuple[str, float, int]] = []

    macro_block = masks["macro_block"]
    pre_news_30m = masks["pre_news_30m"]
    btc_shock_any = masks["btc_shock_any"]
    btc_bear_shock = masks["btc_bear_shock"]

    pause_all_until = -1
    pause_long_until = -1

    cur_day = -1
    day_sl_count = 0
    daily_halted = False

    m_start_eq: Dict[str, float] = {}
    m_end_eq: Dict[str, float] = {}

    def _close(pos: _OpenPos, trig_p: float, reason: str, is_maker_exit: bool, bar_i: int, ts: int) -> None:
        nonlocal equity, peak_eq, max_dd_pct, day_sl_count, daily_halted, pause_all_until
        is_l = pos.direction == "LONG"
        exec_p = trig_p if is_maker_exit else (trig_p * (1.0 - slip) if is_l else trig_p * (1.0 + slip))
        gross = (exec_p - pos.entry_price) * pos.qty if is_l else (pos.entry_price - exec_p) * pos.qty
        ex_fee = abs(pos.qty * exec_p) * (maker_rate if is_maker_exit else taker_rate)
        net = max(-pos.margin, gross - (pos.entry_fee + ex_fee))
        equity = max(1.0, equity + net)

        if reason == "SL":
            day_sl_count += 1
            if day_sl_count >= max_daily_stops:
                daily_halted = True
            if use_shield and day_sl_count >= 2:
                # 6-hour (24 bars) pause after 2 stop-losses in one day!
                pause_all_until = max(pause_all_until, bar_i + 24)

        if equity > peak_eq:
            peak_eq = equity
        dd = ((peak_eq - equity) / peak_eq) * 100.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        trades.append((reason, net, ts))
        cd_until[pos.symbol] = bar_i + 3

    def _open_market_short(sym: str, cl_p: float, bar_i: int, ts: int) -> None:
        margin_req = equity * (margin_pct / 100.0)
        if sum(p.margin for p in open_pos) + margin_req > equity * 0.95:
            return
        ep = cl_p * (1.0 - slip)  # Taker short + 0.04% slippage
        notional = margin_req * leverage
        pos = _OpenPos(
            symbol=sym,
            direction="SHORT",
            entry_price=ep,
            sl_price=ep * (1.0 + sl_frac),
            tp_price=ep * (1.0 - tp_frac),
            be_trigger_price=ep * (1.0 - be_trig_frac),
            be_stop_price=ep * (1.0 - be_off_frac),
            qty=notional / ep,
            margin=margin_req,
            entry_fee=notional * taker_rate,
        )
        open_pos.append(pos)

    for i in range(len(timeline)):
        ts = int(timeline[i])
        for mn, ms, me in month_bounds:
            if mn not in m_start_eq and ts >= ms:
                m_start_eq[mn] = equity
            if ts <= me:
                m_end_eq[mn] = equity

        dk = ts // 86_400_000
        if dk != cur_day:
            cur_day = dk
            day_sl_count = 0
            daily_halted = False

        # Trigger 6h pause on BTC shock (Variant 2) or 6h long-ban on bearish shock (Variant 3)
        if use_shield and btc_shock_any[i]:
            pause_all_until = max(pause_all_until, i + 24)
        if use_news_short and btc_bear_shock[i]:
            pause_long_until = max(pause_long_until, i + 24)

        # 1. Manage existing open positions
        if open_pos:
            still: List[_OpenPos] = []
            for pos in open_pos:
                sd = bundle[pos.symbol]
                hi, lo, cl = sd["high"][i], sd["low"][i], sd["close"][i]
                if not np.isfinite(lo):
                    still.append(pos); continue
                is_l = pos.direction == "LONG"

                # Standard SL / BE / TP check first (conservative)
                if pos.be_active:
                    if (lo <= pos.be_stop_price) if is_l else (hi >= pos.be_stop_price):
                        _close(pos, pos.be_stop_price, "BE", False, i, ts)
                        continue
                    if (hi >= pos.tp_price) if is_l else (lo <= pos.tp_price):
                        _close(pos, pos.tp_price, "TP", True, i, ts)
                        continue
                else:
                    if (lo <= pos.sl_price) if is_l else (hi >= pos.sl_price):
                        _close(pos, pos.sl_price, "SL", False, i, ts)
                        continue
                    if (hi >= pos.tp_price) if is_l else (lo <= pos.tp_price):
                        _close(pos, pos.tp_price, "TP", True, i, ts)
                        continue
                    if (hi >= pos.be_trigger_price) if is_l else (lo <= pos.be_trigger_price):
                        pos.be_active = True

                # Pre-news protection 30m before USD High release (Variants 2 & 3)
                if (use_shield or use_news_short) and pre_news_30m[i]:
                    upnl_pct = ((cl - pos.entry_price) / pos.entry_price * 100.0) if is_l else ((pos.entry_price - cl) / pos.entry_price * 100.0)
                    if upnl_pct > 0.30:
                        pos.be_active = True  # lock SL at BE (+0.15%)
                        still.append(pos)
                    else:
                        _close(pos, float(cl), "PRE_NEWS", False, i, ts)
                    continue

                still.append(pos)
            open_pos = still

        # Pause flags for NEW entries
        new_entries_blocked = daily_halted
        if use_shield and (macro_block[i] or i < pause_all_until):
            new_entries_blocked = True

        # 2. Process pending Maker entries
        if pending:
            npend: List[_Pending] = []
            for ord_p in pending:
                if i > ord_p.expires_idx or new_entries_blocked:
                    continue
                if use_news_short and ord_p.direction == "LONG" and (i < pause_long_until or macro_block[i]):
                    continue
                if any(p.symbol == ord_p.symbol for p in open_pos) or i < cd_until.get(ord_p.symbol, -1):
                    continue
                sd = bundle[ord_p.symbol]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    if i < ord_p.expires_idx: npend.append(ord_p)
                    continue
                is_l = ord_p.direction == "LONG"
                if not ((lo < ord_p.limit_price) if is_l else (hi > ord_p.limit_price)):
                    if i < ord_p.expires_idx: npend.append(ord_p)
                    continue
                if len(open_pos) >= max_open or sum(1 for p in open_pos if p.direction == ord_p.direction) >= max_same_dir:
                    continue
                margin_req = equity * (margin_pct / 100.0)
                if sum(p.margin for p in open_pos) + margin_req > equity * 0.95:
                    continue

                notional = margin_req * leverage
                ep = ord_p.limit_price
                pos = _OpenPos(
                    symbol=ord_p.symbol,
                    direction=ord_p.direction,
                    entry_price=ep,
                    sl_price=ep * (1.0 - sl_frac) if is_l else ep * (1.0 + sl_frac),
                    tp_price=ep * (1.0 + tp_frac) if is_l else ep * (1.0 - tp_frac),
                    be_trigger_price=ep * (1.0 + be_trig_frac) if is_l else ep * (1.0 - be_trig_frac),
                    be_stop_price=ep * (1.0 + be_off_frac) if is_l else ep * (1.0 - be_off_frac),
                    qty=notional / ep,
                    margin=margin_req,
                    entry_fee=notional * maker_rate,
                )
                if (lo <= pos.sl_price) if is_l else (hi >= pos.sl_price):
                    _close(pos, pos.sl_price, "SL", False, i, ts)
                else:
                    open_pos.append(pos)
            pending = npend

        if new_entries_blocked or len(open_pos) >= max_open:
            continue

        # 3. Variant 3 Special: Bearish News / BTC Shock SHORT entries (strictly derived on Days 1-120!)
        if use_news_short and btc_bear_shock[i]:
            busy = {p.symbol for p in open_pos} | {o.symbol for o in pending}
            short_cands: List[Tuple[float, str, float]] = []
            for sym in symbols:
                if sym in busy or i < cd_until.get(sym, -1):
                    continue
                if weak_short_ok[sym][i]:
                    sd = bundle[sym]
                    short_cands.append((float(sd["h1_adx"][i]), sym, float(sd["close"][i])))
            short_cands.sort(key=lambda x: x[0], reverse=True)
            for _, sym, cl_p in short_cands:
                if len(open_pos) >= max_open or sum(1 for p in open_pos if p.direction == "SHORT") >= max_same_dir:
                    break
                _open_market_short(sym, cl_p, i, ts)

        if len(open_pos) >= max_open:
            continue

        # 4. Standard `fibo_trend_v2` Variant B scan (H1 ADX >= 30, H4 ADX >= 20, ATR <= 1.5%)
        busy = {p.symbol for p in open_pos} | {o.symbol for o in pending}
        cands: List[Tuple[float, str, str, float]] = []
        allow_long_now = not (use_news_short and (i < pause_long_until or macro_block[i]))

        for sym in symbols:
            if sym in busy or i < cd_until.get(sym, -1):
                continue
            sd = bundle[sym]
            if sd["h1_adx"][i] < 30.0 or sd["h4_adx"][i] < 20.0 or sd["atr_pct"][i] > 1.50:
                continue
            if allow_long_now and sd["raw_L"][i]:
                cands.append((float(sd["q"][i]), sym, "LONG", float(sd["close"][i])))
            elif sd["raw_S"][i]:
                cands.append((float(sd["q"][i]), sym, "SHORT", float(sd["close"][i])))

        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)
        for _, sym, direction, cl_p in cands:
            if len(open_pos) + len(pending) >= max_open + 3:
                break
            if sum(1 for p in open_pos if p.direction == direction) >= max_same_dir:
                continue
            lp = cl_p * (1.0 - pb_frac) if direction == "LONG" else cl_p * (1.0 + pb_frac)
            pending.append(_Pending(symbol=sym, direction=direction, limit_price=lp, expires_idx=i + 2))

    if open_pos and len(timeline) > 0:
        last_ts = int(timeline[-1])
        for pos in open_pos:
            cl = bundle[pos.symbol]["close"][-1]
            _close(pos, float(cl) if np.isfinite(cl) else pos.entry_price, "EOD", False, len(timeline) - 1, last_ts)
        for mn, _, me in month_bounds:
            if last_ts <= me:
                m_end_eq[mn] = equity

    mp = {
        mn: round(((m_end_eq.get(mn, 1000.0) - m_start_eq.get(mn, 1000.0)) / max(m_start_eq.get(mn, 1000.0), 1.0)) * 100.0, 2)
        for mn, _, _ in month_bounds
    }
    n_tr = len(trades)
    wins = sum(1 for _, net, _ in trades if net > 0)
    sl_cnt = sum(1 for r, _, _ in trades if r == "SL")
    pre_cnt = sum(1 for r, _, _ in trades if r == "PRE_NEWS")
    return {
        "mode": mode,
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n_tr,
        "wr_pct": round((wins / n_tr) * 100.0, 2) if n_tr else 0.0,
        "sl_count": sl_cnt,
        "pre_news_closes": pre_cnt,
        "max_dd_pct": round(max_dd_pct, 2),
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    top25 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"][:25]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    day_ms = 86_400_000
    sim_end = int(df_btc["timestamp"].iloc[-1])
    sim_start = sim_end - 180 * day_ms

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start) & (df_btc["timestamp"] <= sim_end)]
    timeline = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    month_bounds = [
        ("M1", sim_start + 0 * day_ms, sim_start + 30 * day_ms - 1),
        ("M2", sim_start + 30 * day_ms, sim_start + 60 * day_ms - 1),
        ("M3", sim_start + 60 * day_ms, sim_start + 90 * day_ms - 1),
        ("M4", sim_start + 90 * day_ms, sim_start + 120 * day_ms - 1),
        ("M5", sim_start + 120 * day_ms, sim_start + 150 * day_ms - 1),
        ("M6", sim_start + 150 * day_ms, sim_end),
    ]

    masks = build_macro_and_btc_masks(df_btc, timeline)
    bundle: Dict[str, Dict[str, np.ndarray]] = {}
    weak_short_ok: Dict[str, np.ndarray] = {}
    syms: List[str] = []

    h1_ms = INTERVAL_TO_MS["60"]
    for sym in top25:
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 500:
            continue
        bundle[sym] = build_tuning_bundle(df_s, df_btc, timeline_df)

        # Weak altcoin condition for Variant 3 news-short: below daily VWAP & below 1H EMA50 & red bar
        df_w = df_s.copy().sort_values("timestamp").reset_index(drop=True)
        df_w["close_ts"] = (df_w["timestamp"] + INTERVAL_TO_MS["15"]).astype("int64")
        df_w["vwap"] = compute_daily_vwap(df_w)
        h1 = BybitClient.resample_klines(df_w, "60")
        h1_df = pd.DataFrame({"close_ts": (h1["timestamp"] + h1_ms).astype("int64"), "h1_e50": ema(h1["close"], 50).values}).sort_values("close_ts")
        df_w = pd.merge_asof(df_w, h1_df, on="close_ts", direction="backward")
        df_w["weak_s"] = (df_w["close"] < df_w["vwap"]) & (df_w["close"] < df_w["h1_e50"]) & (df_w["close"] < df_w["open"])
        al_w = timeline_df.merge(df_w[["timestamp", "weak_s"]], on="timestamp", how="left")
        weak_short_ok[sym] = al_w["weak_s"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        syms.append(sym)

    out = []
    print("\n" + "=" * 112)
    print("POLYGRAPH 6-MONTH EXAM: NEWS HEALING ON `fibo_trend_v2` VARIANT B (15% x 10x | FEES + 0.04% SLIPPAGE)")
    print("=" * 112)
    for mode, label in (
        ("base", "1. База Вариант B (без новостного фона)"),
        ("shield_pause", "2. Щит-пауза (-2ч/+3ч новости, -30м защита, 6ч пауза на BTC-шок / 2 SL)"),
        ("news_short", "3. Новостной шорт / разворот (запрет лонгов + шорт слабых альтов на проливе BTC)"),
    ):
        r = simulate_6m_variant(mode, syms, bundle, weak_short_ok, masks, timeline, month_bounds)
        r["label"] = label
        out.append(r)
        mp = r["monthly_pct"]
        print(f"\n>>> {label}")
        print(
            f"  M1: {mp['M1']:>+6.1f}% / M2: {mp['M2']:>+6.1f}% / M3: {mp['M3']:>+6.1f}% / M4: {mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5: {mp['M5']:>+6.1f}% / M6: {mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого за 6 мес.: $1 000 -> ${r['end_usd']:,.2f} | Сделок: {r['trades']} | WR: {r['wr_pct']:.1f}% | "
            f"Стопов: {r['sl_count']} (PreNews: {r['pre_news_closes']}) | Макс. DD: {r['max_dd_pct']:.1f}%"
        )

    Path("data/news_heal_v2_6m_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/news_heal_v2_6m_results.json")


if __name__ == "__main__":
    main()
