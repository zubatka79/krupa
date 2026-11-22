"""POLYGRAPH 6-MONTH EXAM (`bot/news_decision_6m.py`): Situational News & Catalyst Decision Engine.

All decision rules derived STRICTLY from Days 1-120 forensics (Months 5-6 completely unseen):
  1. Red US Macro Release (-30m..+15m around FOMC/CPI/NFP/PPI/Fed Chair):
     PAUSE new entries & PROTECT open trades (>+0.3% -> move SL to BE +0.15%, else close at market).
  2. Dead Calm / No-News Chop (|BTC 24h| < 1.8% AND BTC 4h range < 1.8%):
     SKIP / PAUSE all new entries (no market catalyst; Days 1-120 showed 30% SL rate in calm chop).
  3. Bullish Catalyst / News Wave (BTC 24h >= +1.8%, OR active BTC 4h range >= 1.8% with BTC 24h > 0 & BTC 4H not bearish):
     ALLOW ONLY LONG (forbid counter-wave shorts).
  4. Bearish Catalyst / News Wave (BTC 24h <= -1.8%, OR active BTC 4h range >= 1.8% with BTC 24h < 0 & BTC 4H not bullish):
     ALLOW ONLY SHORT (forbid knife-catching longs).

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
from bot.indicators import ema

# Official 2026 US High-Impact Releases (FOMC, CPI, NFP, PPI, Fed Chair) across May 23 - Nov 19, 2026
US_MACRO_2026_UTC = [
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
    ("2026-08-21T14:00:00Z", "Fed Chair Jackson Hole"),
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
    ("2026-11-16T17:00:00Z", "Fed Chair Speaks"),
    ("2026-11-18T19:00:00Z", "FOMC Minutes"),
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


def build_news_decision_signals(df_btc_15m: pd.DataFrame, timeline: np.ndarray) -> Dict[str, np.ndarray]:
    """Build strictly causal situational decision flags (ALLOW_LONG, ALLOW_SHORT, PROTECT_30M)."""
    close_ts = timeline + 15 * 60_000
    n = len(timeline)

    macro_wick_block = np.zeros(n, dtype=bool)
    protect_30m = np.zeros(n, dtype=bool)
    for iso_ts, _ in US_MACRO_2026_UTC:
        ev = int(datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).timestamp() * 1000)
        # Block entries only during the immediate release coin-flip (-30m .. +15m)
        macro_wick_block |= (close_ts >= ev - 30 * 60_000) & (close_ts <= ev + 15 * 60_000)
        protect_30m |= (close_ts >= ev - 35 * 60_000) & (close_ts <= ev - 20 * 60_000)

    btc = df_btc_15m[["timestamp", "high", "low", "close"]].copy().sort_values("timestamp")
    btc["close_ts"] = btc["timestamp"] + INTERVAL_TO_MS["15"]
    btc["r24h"] = btc["close"].pct_change(96) * 100.0
    btc["vol4h"] = (btc["high"].rolling(16, min_periods=16).max() - btc["low"].rolling(16, min_periods=16).min()) / btc["close"] * 100.0

    btc_4h = BybitClient.resample_klines(df_btc_15m, "240")
    b4_e20, b4_e50 = ema(btc_4h["close"], 20), ema(btc_4h["close"], 50)
    b4_df = pd.DataFrame({
        "close_ts": (btc_4h["timestamp"] + INTERVAL_TO_MS["240"]).astype("int64"),
        "b4_strict_bull": (btc_4h["close"] > b4_e50) & (b4_e20 > b4_e50),
        "b4_strict_bear": (btc_4h["close"] < b4_e50) & (b4_e20 < b4_e50),
    }).sort_values("close_ts")
    btc = pd.merge_asof(btc, b4_df, on="close_ts", direction="backward")

    al = pd.DataFrame({"timestamp": timeline}).merge(
        btc[["timestamp", "r24h", "vol4h", "b4_strict_bull", "b4_strict_bear"]],
        on="timestamp",
        how="left",
    )
    r24h = al["r24h"].fillna(0.0).to_numpy(dtype=np.float64)
    vol4h = al["vol4h"].fillna(0.0).to_numpy(dtype=np.float64)
    b4_bull = al["b4_strict_bull"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    b4_bear = al["b4_strict_bear"].astype("boolean").fillna(False).to_numpy(dtype=bool)

    # Situational Agent Decision Rules (strictly from Days 1-120 forensics):
    # - Active catalyst wave exists if |BTC 24h| >= 1.8% OR BTC 4h range >= 1.8%
    # - Direction MUST align with BTC 24h wave and NOT fight BTC 4H trend!
    has_catalyst = (np.abs(r24h) >= 1.8) | (vol4h >= 1.8)
    allow_long = (~macro_wick_block) & has_catalyst & (r24h > 0.0) & (~b4_bear)
    allow_short = (~macro_wick_block) & has_catalyst & (r24h < 0.0) & (~b4_bull)

    return {
        "allow_long": allow_long,
        "allow_short": allow_short,
        "protect_30m": protect_30m,
    }


def simulate_6m(
    use_agent_news: bool,
    symbols: List[str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    decision: Dict[str, np.ndarray],
    timeline: np.ndarray,
    month_bounds: List[Tuple[str, int, int]],
    initial_deposit: float = 1000.0,
) -> Dict[str, Any]:
    maker_rate = 0.0002
    taker_rate = 0.00055
    slip = 0.0004  # 0.04% slippage on ALL SL, BE, and pre-news market exits

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

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_pct = 0.0

    open_pos: List[_OpenPos] = []
    pending: List[_Pending] = []
    cd_until: Dict[str, int] = {}
    trades: List[Tuple[str, float]] = []

    allow_long = decision["allow_long"]
    allow_short = decision["allow_short"]
    protect_30m = decision["protect_30m"]

    cur_day = -1
    day_sl_count = 0
    daily_halted = False
    m_start_eq: Dict[str, float] = {}
    m_end_eq: Dict[str, float] = {}

    def _close(pos: _OpenPos, trig_p: float, reason: str, is_mk_exit: bool, bar_i: int) -> None:
        nonlocal equity, peak_eq, max_dd_pct, day_sl_count, daily_halted
        is_l = pos.direction == "LONG"
        exec_p = trig_p if is_mk_exit else (trig_p * (1.0 - slip) if is_l else trig_p * (1.0 + slip))
        gross = (exec_p - pos.entry_price) * pos.qty if is_l else (pos.entry_price - exec_p) * pos.qty
        ex_fee = abs(pos.qty * exec_p) * (maker_rate if is_mk_exit else taker_rate)
        net = max(-pos.margin, gross - (pos.entry_fee + ex_fee))
        equity = max(1.0, equity + net)
        if reason == "SL":
            day_sl_count += 1
            if day_sl_count >= max_daily_stops:
                daily_halted = True
        if equity > peak_eq:
            peak_eq = equity
        dd = ((peak_eq - equity) / peak_eq) * 100.0
        if dd > max_dd_pct:
            max_dd_pct = dd
        trades.append((reason, net))
        cd_until[pos.symbol] = bar_i + 3

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

        # 1. Manage open positions
        if open_pos:
            still: List[_OpenPos] = []
            for pos in open_pos:
                sd = bundle[pos.symbol]
                hi, lo, cl = sd["high"][i], sd["low"][i], sd["close"][i]
                if not np.isfinite(lo):
                    still.append(pos); continue
                is_l = pos.direction == "LONG"

                if pos.be_active:
                    if (lo <= pos.be_stop_price) if is_l else (hi >= pos.be_stop_price):
                        _close(pos, pos.be_stop_price, "BE", False, i); continue
                    if (hi >= pos.tp_price) if is_l else (lo <= pos.tp_price):
                        _close(pos, pos.tp_price, "TP", True, i); continue
                else:
                    if (lo <= pos.sl_price) if is_l else (hi >= pos.sl_price):
                        _close(pos, pos.sl_price, "SL", False, i); continue
                    if (hi >= pos.tp_price) if is_l else (lo <= pos.tp_price):
                        _close(pos, pos.tp_price, "TP", True, i); continue
                    if (hi >= pos.be_trigger_price) if is_l else (lo <= pos.be_trigger_price):
                        pos.be_active = True

                # Protect open positions 30m before red US macro release
                if use_agent_news and protect_30m[i]:
                    upnl_pct = ((cl - pos.entry_price) / pos.entry_price * 100.0) if is_l else ((pos.entry_price - cl) / pos.entry_price * 100.0)
                    if upnl_pct > 0.30:
                        pos.be_active = True
                        still.append(pos)
                    else:
                        _close(pos, float(cl), "PRE_NEWS", False, i)
                    continue
                still.append(pos)
            open_pos = still

        # 2. Pending Maker limit fills
        if pending:
            npend: List[_Pending] = []
            for o in pending:
                if i > o.expires_idx or daily_halted:
                    continue
                if use_agent_news and ((o.direction == "LONG" and not allow_long[i]) or (o.direction == "SHORT" and not allow_short[i])):
                    continue
                if any(p.symbol == o.symbol for p in open_pos) or i < cd_until.get(o.symbol, -1):
                    continue
                sd = bundle[o.symbol]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    if i < o.expires_idx: npend.append(o)
                    continue
                is_l = o.direction == "LONG"
                if not ((lo < o.limit_price) if is_l else (hi > o.limit_price)):
                    if i < o.expires_idx: npend.append(o)
                    continue
                if len(open_pos) >= max_open or sum(1 for p in open_pos if p.direction == o.direction) >= max_same_dir:
                    continue
                margin_req = equity * (margin_pct / 100.0)
                if sum(p.margin for p in open_pos) + margin_req > equity * 0.95:
                    continue
                notional = margin_req * leverage
                ep = o.limit_price
                pos = _OpenPos(
                    symbol=o.symbol,
                    direction=o.direction,
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
                    _close(pos, pos.sl_price, "SL", False, i)
                else:
                    open_pos.append(pos)
            pending = npend

        if daily_halted or len(open_pos) >= max_open:
            continue

        # 3. Scan closed 15M candle
        can_L = (not use_agent_news) or bool(allow_long[i])
        can_S = (not use_agent_news) or bool(allow_short[i])
        if not can_L and not can_S:
            continue

        busy = {p.symbol for p in open_pos} | {o.symbol for o in pending}
        cands: List[Tuple[float, str, str, float]] = []
        for sym in symbols:
            if sym in busy or i < cd_until.get(sym, -1):
                continue
            sd = bundle[sym]
            if sd["h1_adx"][i] < 30.0 or sd["h4_adx"][i] < 20.0 or sd["atr_pct"][i] > 1.50:
                continue
            if can_L and sd["raw_L"][i]:
                cands.append((float(sd["q"][i]), sym, "LONG", float(sd["close"][i])))
            elif can_S and sd["raw_S"][i]:
                cands.append((float(sd["q"][i]), sym, "SHORT", float(sd["close"][i])))

        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)
        for _, sym, d, cl_p in cands:
            if len(open_pos) + len(pending) >= max_open + 3:
                break
            if sum(1 for p in open_pos if p.direction == d) >= max_same_dir:
                continue
            lp = cl_p * (1.0 - pb_frac) if d == "LONG" else cl_p * (1.0 + pb_frac)
            pending.append(_Pending(symbol=sym, direction=d, limit_price=lp, expires_idx=i + 2))

    if open_pos and len(timeline) > 0:
        for pos in open_pos:
            cl = bundle[pos.symbol]["close"][-1]
            _close(pos, float(cl) if np.isfinite(cl) else pos.entry_price, "EOD", False, len(timeline) - 1)
        for mn, _, me in month_bounds:
            if int(timeline[-1]) <= me:
                m_end_eq[mn] = equity

    mp = {
        mn: round(((m_end_eq.get(mn, 1000.0) - m_start_eq.get(mn, 1000.0)) / max(m_start_eq.get(mn, 1000.0), 1.0)) * 100.0, 2)
        for mn, _, _ in month_bounds
    }
    n_tr = len(trades)
    wins = sum(1 for _, net in trades if net > 0)
    return {
        "name": "Решение агента с новостным/катализаторным фильтром" if use_agent_news else "База без новостного фона",
        "monthly_pct": mp,
        "end_usd": round(equity, 2),
        "trades": n_tr,
        "wr_pct": round((wins / n_tr) * 100.0, 2) if n_tr else 0.0,
        "sl_count": sum(1 for r, _ in trades if r == "SL"),
        "tp_count": sum(1 for r, _ in trades if r == "TP"),
        "be_count": sum(1 for r, _ in trades if r == "BE"),
        "pre_news_count": sum(1 for r, _ in trades if r == "PRE_NEWS"),
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

    decision = build_news_decision_signals(df_btc, timeline)
    bundle = {}
    syms = []
    for sym in top25:
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 500:
            continue
        bundle[sym] = build_tuning_bundle(df_s, df_btc, timeline_df)
        syms.append(sym)

    res_base = simulate_6m(False, syms, bundle, decision, timeline, month_bounds)
    res_agent = simulate_6m(True, syms, bundle, decision, timeline, month_bounds)

    print("\n" + "=" * 108)
    print("POLYGRAPH 6-MONTH EXAM: SITUATIONAL NEWS & CATALYST DECISION ENGINE (`news_decision_6m.py`)")
    print("=" * 108)
    for r in (res_base, res_agent):
        mp = r["monthly_pct"]
        print(f"\n>>> {r['name']}")
        print(
            f"  M1: {mp['M1']:>+6.1f}% / M2: {mp['M2']:>+6.1f}% / M3: {mp['M3']:>+6.1f}% / M4: {mp['M4']:>+6.1f}% "
            f"|| СЛЕПЫЕ: M5: {mp['M5']:>+6.1f}% / M6: {mp['M6']:>+6.1f}%"
        )
        print(
            f"  Итого: $1 000 -> ${r['end_usd']:,.2f} | Сделок: {r['trades']} | WR: {r['wr_pct']:.1f}% | "
            f"Стопов: {r['sl_count']} (TP={r['tp_count']}, BE={r['be_count']}, PreNews={r['pre_news_count']}) | Макс. DD: {r['max_dd_pct']:.1f}%"
        )

    Path("data/news_decision_6m_results.json").write_text(
        json.dumps({"base": res_base, "agent_decision": res_agent}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/news_decision_6m_results.json")


if __name__ == "__main__":
    main()
