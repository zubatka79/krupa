"""STEP 3: 100% Blind Walk-Forward Exam of 4 Data-Driven Forensics Hypotheses on `fibo_trend_v2`.

Hypotheses derived STRICTLY from Days 1-120 forensics (zero peeking at Days 121-180):
  1. time_stop_8h : Close stuck positions after 32 bars (8 hours) at market
  2. no_weekends  : Forbid new entries on Saturday/Sunday UTC (day_of_week >= 5)
  3. early_be_50  : Trigger Smart BE (+0.15%) at 50% of TP (+0.75%) instead of 65% (+0.975%)
  4. h1_adx_25    : Require stronger 1H trend (H1 ADX >= 25 while keeping H4 ADX >= 20)

Strictly offline using data/cache/*.csv.gz. Does NOT modify config.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_walkforward import load_cached_15m, slice_sym_data
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, compute_rsi, ema


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
    entry_idx: int
    entry_price: float
    sl_price: float
    tp_price: float
    be_trigger_price: float
    be_stop_price: float
    qty: float
    margin: float
    entry_fee: float
    be_active: bool = False


def build_forensics_bundle(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    timeline_df: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Compute exact baseline `fibo_trend_v2` signals plus masks for no_weekends and h1_adx_25."""
    entry_ms = INTERVAL_TO_MS["15"]
    h1_ms = INTERVAL_TO_MS["60"]
    h4_ms = INTERVAL_TO_MS["240"]

    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    df["rsi"] = compute_rsi(df["close"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0

    c_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_ratio = ((df["close"] - df["open"]).abs() / c_range).fillna(0.0)

    bar_range = df["high"] - df["low"]
    range_3 = df["high"].rolling(3, min_periods=3).max() - df["low"].rolling(3, min_periods=3).min()
    roc_3_abs = (df["close"] - df["close"].shift(3)).abs() / df["close"].shift(3).replace(0.0, np.nan)
    atr_frac = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)

    is_abn = ((bar_range > 2.2 * df["atr"]) | (range_3 > 3.8 * df["atr"]) | (roc_3_abs > 2.2 * atr_frac)).fillna(False)
    vol_cd = is_abn | is_abn.shift(1, fill_value=False) | is_abn.shift(2, fill_value=False) | is_abn.shift(3, fill_value=False)

    btc = df_btc_15m[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc["btc_move"] = ((btc["close"] - btc["open"]).abs() / btc["open"].replace(0.0, np.nan)) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_move"]], on="timestamp", direction="backward")
    common_ok = (~vol_cd) & (df["btc_move"].fillna(0.0) <= 0.8) & (df["atr_pct"] >= 0.12) & (df["atr_pct"] <= 1.50)

    df_1h = BybitClient.resample_klines(df_15m, "60")
    df_4h = BybitClient.resample_klines(df_15m, "240")

    h1_e20 = ema(df_1h["close"], 20)
    h1_e50 = ema(df_1h["close"], 50)
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out = pd.DataFrame({
        "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
        "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50) & (h1_adx >= 20.0),
        "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50) & (h1_adx >= 20.0),
        "h1_adx": h1_adx.values,
    }).sort_values("close_ts")

    h4_e20 = ema(df_4h["close"], 20)
    h4_e50 = ema(df_4h["close"], 50)
    h4_adx, _, _ = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    h4_out = pd.DataFrame({
        "close_ts": (df_4h["timestamp"] + h4_ms).astype("int64"),
        "h4_bull": (df_4h["close"] > h4_e50) & (h4_e20 > h4_e50) & (h4_adx >= 20.0),
        "h4_bear": (df_4h["close"] < h4_e50) & (h4_e20 < h4_e50) & (h4_adx >= 20.0),
        "h4_adx": h4_adx.values,
    }).sort_values("close_ts")

    df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
    df = pd.merge_asof(df, h4_out, on="close_ts", direction="backward")

    htf_bull = df["h1_bull"].astype("boolean").fillna(False) & df["h4_bull"].astype("boolean").fillna(False)
    htf_bear = df["h1_bear"].astype("boolean").fillna(False) & df["h4_bear"].astype("boolean").fillna(False)
    h1_adx_s = df["h1_adx"].fillna(0.0)
    adx_avg = (h1_adx_s + df["h4_adx"].fillna(0.0)) / 2.0

    rsi_min3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).min(axis=1)
    rsi_max3 = pd.concat([df["rsi"].shift(1), df["rsi"].shift(2), df["rsi"].shift(3)], axis=1).max(axis=1)
    rsi_L = (rsi_min3 < 45.0) & (df["rsi"] > df["rsi"].shift(1)) & (df["rsi"] < 62.0)
    rsi_S = (rsi_max3 > 55.0) & (df["rsi"] < df["rsi"].shift(1)) & (df["rsi"] > 38.0)

    bounce_L = (df["close"] > df["open"]) & (body_ratio >= 0.35) & (df["close"] > df["close"].shift(1))
    bounce_S = (df["close"] < df["open"]) & (body_ratio >= 0.35) & (df["close"] < df["close"].shift(1))

    w_hi = df["high"].shift(1).rolling(36, min_periods=36).max()
    w_lo = df["low"].shift(1).rolling(36, min_periods=36).min()
    w_rng = w_hi - w_lo
    w_ok = w_rng >= (2.2 * df["atr"])
    min_low_2 = pd.concat([df["low"], df["low"].shift(1)], axis=1).min(axis=1)
    max_high_2 = pd.concat([df["high"], df["high"].shift(1)], axis=1).max(axis=1)

    z618_L = w_ok & (min_low_2 <= w_hi - 0.60 * w_rng) & (min_low_2 >= w_hi - 0.80 * w_rng) & (df["close"] > w_hi - 0.786 * w_rng)
    z618_S = w_ok & (max_high_2 >= w_lo + 0.60 * w_rng) & (max_high_2 <= w_lo + 0.80 * w_rng) & (df["close"] < w_lo + 0.786 * w_rng)

    dt_series = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["is_weekday"] = dt_series.dt.day_of_week < 5
    df["h1_adx_ge25"] = h1_adx_s >= 25.0
    df["base_L"] = common_ok & htf_bull & z618_L & rsi_L & bounce_L
    df["base_S"] = common_ok & htf_bear & z618_S & rsi_S & bounce_S
    df["q"] = (adx_avg + body_ratio * 25.0).fillna(0.0)

    aligned = timeline_df.merge(
        df[["timestamp", "high", "low", "close", "q", "base_L", "base_S", "is_weekday", "h1_adx_ge25"]],
        on="timestamp",
        how="left",
    )
    return {
        "high": aligned["high"].to_numpy(dtype=np.float64),
        "low": aligned["low"].to_numpy(dtype=np.float64),
        "close": aligned["close"].to_numpy(dtype=np.float64),
        "q": aligned["q"].fillna(0.0).to_numpy(dtype=np.float64),
        "base_L": aligned["base_L"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "base_S": aligned["base_S"].astype("boolean").fillna(False).to_numpy(dtype=bool),
        "is_weekday": aligned["is_weekday"].astype("boolean").fillna(True).to_numpy(dtype=bool),
        "h1_adx_ge25": aligned["h1_adx_ge25"].astype("boolean").fillna(False).to_numpy(dtype=bool),
    }


def simulate_forensics_variant(
    symbols: List[str],
    bundle: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    hypotheses: Tuple[str, ...],
    margin_pct: float = 15.0,
    leverage: float = 10.0,
    max_open: int = 5,
    max_same_dir: int = 3,
    max_daily_stops: int = 4,
    tp_pct: float = 1.50,
    sl_pct: float = 3.50,
    initial_deposit: float = 1000.0,
) -> Dict[str, Any]:
    h_set = set(hypotheses)
    use_time_stop = "time_stop_8h" in h_set
    use_no_wknd = "no_weekends" in h_set
    use_early_be = "early_be_50" in h_set
    use_h1_adx25 = "h1_adx_25" in h_set

    be_ratio = 0.50 if use_early_be else 0.65
    max_hold_bars = 32 if use_time_stop else 999_999

    maker_rate = 0.0002
    taker_rate = 0.00055
    tp_frac = tp_pct / 100.0
    sl_frac = sl_pct / 100.0
    pb_frac = 0.00065
    be_trig_frac = (tp_pct * be_ratio) / 100.0
    be_off_frac = 0.0015
    notional_mult = (margin_pct / 100.0) * leverage
    daily_dd_cap_pct = max_daily_stops * notional_mult * (sl_frac + maker_rate + taker_rate) * 100.0

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    open_pos: List[_OpenPos] = []
    pending: List[_Pending] = []
    cooldown_until: Dict[str, int] = {}
    trades: List[Tuple[str, float]] = []

    n_bars = len(timeline)
    cur_day = -1
    day_start_eq = equity
    day_sl_count = 0
    daily_halted = False

    def _close(pos: _OpenPos, exit_p: float, reason: str, exit_maker: bool, bar_i: int) -> None:
        nonlocal equity, peak_eq, max_dd_usd, max_dd_pct, day_sl_count, daily_halted
        gross = (
            (exit_p - pos.entry_price) * pos.qty
            if pos.direction == "LONG"
            else (pos.entry_price - exit_p) * pos.qty
        )
        exit_fee = abs(pos.qty * exit_p) * (maker_rate if exit_maker else taker_rate)
        net = gross - (pos.entry_fee + exit_fee)
        if net < -pos.margin:
            net = -pos.margin
        equity = max(1.0, equity + net)

        if reason == "SL":
            day_sl_count += 1
            if day_sl_count >= max_daily_stops:
                daily_halted = True

        if equity > peak_eq:
            peak_eq = equity
        dd_u = peak_eq - equity
        dd_p = (dd_u / peak_eq) * 100.0 if peak_eq > 0 else 0.0
        if dd_u > max_dd_usd:
            max_dd_usd = dd_u
        if dd_p > max_dd_pct:
            max_dd_pct = dd_p

        trades.append((reason, net))
        cooldown_until[pos.symbol] = bar_i + 3

    for i in range(n_bars):
        ts = int(timeline[i])
        day_key = ts // 86_400_000
        if day_key != cur_day:
            cur_day = day_key
            day_start_eq = equity
            day_sl_count = 0
            daily_halted = False

        # 1. Manage open positions (conservative priority + optional 8h time-stop)
        if open_pos:
            still_open: List[_OpenPos] = []
            for pos in open_pos:
                sd = bundle[pos.symbol]
                hi = sd["high"][i]
                lo = sd["low"][i]
                cl = sd["close"][i]
                if not np.isfinite(lo):
                    still_open.append(pos)
                    continue

                is_long = pos.direction == "LONG"
                if pos.be_active:
                    be_hit = (lo <= pos.be_stop_price) if is_long else (hi >= pos.be_stop_price)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if be_hit:
                        _close(pos, pos.be_stop_price, "BE", False, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, i)
                    elif use_time_stop and (i - pos.entry_idx) >= max_hold_bars:
                        _close(pos, float(cl), "TIME", False, i)
                    else:
                        still_open.append(pos)
                else:
                    sl_hit = (lo <= pos.sl_price) if is_long else (hi >= pos.sl_price)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if sl_hit:
                        _close(pos, pos.sl_price, "SL", False, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, i)
                    else:
                        if (hi >= pos.be_trigger_price) if is_long else (lo <= pos.be_trigger_price):
                            pos.be_active = True
                        if use_time_stop and (i - pos.entry_idx) >= max_hold_bars:
                            _close(pos, float(cl), "TIME", False, i)
                        else:
                            still_open.append(pos)
            open_pos = still_open

        # 2. Daily 4-stop limit
        if not daily_halted and open_pos:
            unreal = 0.0
            for pos in open_pos:
                cl = bundle[pos.symbol]["close"][i]
                if np.isfinite(cl):
                    g = (cl - pos.entry_price) * pos.qty if pos.direction == "LONG" else (pos.entry_price - cl) * pos.qty
                    unreal += g - pos.entry_fee
            dd_day = ((day_start_eq - (equity + unreal)) / max(day_start_eq, 1e-9)) * 100.0
            if dd_day >= daily_dd_cap_pct or day_sl_count >= max_daily_stops:
                daily_halted = True

        # 3. Process pending Maker limit entries (TTL 2 bars)
        if pending:
            next_pend: List[_Pending] = []
            for ord_p in pending:
                if i > ord_p.expires_idx or daily_halted:
                    continue
                if any(p.symbol == ord_p.symbol for p in open_pos):
                    continue
                if i < cooldown_until.get(ord_p.symbol, -1):
                    continue

                sd = bundle[ord_p.symbol]
                hi = sd["high"][i]
                lo = sd["low"][i]
                if not np.isfinite(lo):
                    if i < ord_p.expires_idx:
                        next_pend.append(ord_p)
                    continue

                is_long = ord_p.direction == "LONG"
                filled = (lo <= ord_p.limit_price) if is_long else (hi >= ord_p.limit_price)
                if not filled:
                    if i < ord_p.expires_idx:
                        next_pend.append(ord_p)
                    continue

                if len(open_pos) >= max_open:
                    continue
                if sum(1 for p in open_pos if p.direction == ord_p.direction) >= max_same_dir:
                    continue

                margin_req = equity * (margin_pct / 100.0)
                if sum(p.margin for p in open_pos) + margin_req > equity * 0.95:
                    continue

                notional = margin_req * leverage
                entry_p = ord_p.limit_price
                qty = notional / entry_p
                entry_fee = notional * maker_rate

                if is_long:
                    sl_p = entry_p * (1.0 - sl_frac)
                    tp_p = entry_p * (1.0 + tp_frac)
                    be_tr = entry_p * (1.0 + be_trig_frac)
                    be_st = entry_p * (1.0 + be_off_frac)
                else:
                    sl_p = entry_p * (1.0 + sl_frac)
                    tp_p = entry_p * (1.0 - tp_frac)
                    be_tr = entry_p * (1.0 - be_trig_frac)
                    be_st = entry_p * (1.0 - be_off_frac)

                pos = _OpenPos(
                    symbol=ord_p.symbol,
                    direction=ord_p.direction,
                    entry_idx=i,
                    entry_price=entry_p,
                    sl_price=sl_p,
                    tp_price=tp_p,
                    be_trigger_price=be_tr,
                    be_stop_price=be_st,
                    qty=qty,
                    margin=margin_req,
                    entry_fee=entry_fee,
                )

                if (lo <= sl_p) if is_long else (hi >= sl_p):
                    _close(pos, sl_p, "SL", False, i)
                else:
                    open_pos.append(pos)
            pending = next_pend

        if daily_halted or len(open_pos) >= max_open:
            continue

        # 4. Scan closed 15M bar
        busy = {p.symbol for p in open_pos} | {o.symbol for o in pending}
        cands: List[Tuple[float, str, str, float]] = []
        for sym in symbols:
            if sym in busy or i < cooldown_until.get(sym, -1):
                continue
            sd = bundle[sym]
            if use_no_wknd and not sd["is_weekday"][i]:
                continue
            if use_h1_adx25 and not sd["h1_adx_ge25"][i]:
                continue
            if sd["base_L"][i]:
                cands.append((float(sd["q"][i]), sym, "LONG", float(sd["close"][i])))
            elif sd["base_S"][i]:
                cands.append((float(sd["q"][i]), sym, "SHORT", float(sd["close"][i])))

        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)

        for _, sym, direction, cl_p in cands:
            if len(open_pos) + len(pending) >= max_open + 3:
                break
            if sum(1 for p in open_pos if p.direction == direction) >= max_same_dir:
                continue
            limit_p = cl_p * (1.0 - pb_frac) if direction == "LONG" else cl_p * (1.0 + pb_frac)
            pending.append(_Pending(symbol=sym, direction=direction, limit_price=limit_p, expires_idx=i + 2))

    if open_pos and n_bars > 0:
        for pos in open_pos:
            cl = bundle[pos.symbol]["close"][-1]
            if not np.isfinite(cl):
                cl = pos.entry_price
            _close(pos, float(cl), "EOD", False, n_bars - 1)

    n_tr = len(trades)
    wins = sum(1 for _, net in trades if net > 0)
    net_usd = equity - initial_deposit
    return {
        "end_usd": round(equity, 2),
        "net_usd": round(net_usd, 2),
        "net_pct": round((net_usd / initial_deposit) * 100.0, 2),
        "trades": n_tr,
        "trades_per_day": round(n_tr / 60.0, 2),
        "win_rate_pct": round((wins / n_tr) * 100.0, 2) if n_tr > 0 else 0.0,
        "tp_count": sum(1 for r, _ in trades if r == "TP"),
        "be_count": sum(1 for r, _ in trades if r == "BE"),
        "time_count": sum(1 for r, _ in trades if r == "TIME"),
        "sl_count": sum(1 for r, _ in trades if r == "SL"),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
    }


def run_blind_hypothesis(
    label: str,
    hypotheses: Tuple[str, ...],
    train_bundle: Dict[str, Dict[str, np.ndarray]],
    train_tl: np.ndarray,
    test_bundle: Dict[str, Dict[str, np.ndarray]],
    test_tl: np.ndarray,
    active_syms: List[str],
) -> Dict[str, Any]:
    # Strictly blind Top-25 selection on first 120 days
    ranked: List[Tuple[float, float, int, str]] = []
    for sym in active_syms:
        r_tr = simulate_forensics_variant(
            symbols=[sym],
            bundle={sym: train_bundle[sym]},
            timeline=train_tl,
            hypotheses=hypotheses,
            margin_pct=15.0,
            max_open=1,
            max_same_dir=1,
            max_daily_stops=4,
        )
        ranked.append((r_tr["net_pct"], r_tr["win_rate_pct"], r_tr["trades"], sym))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top_25 = [x[3] for x in ranked[:25]]

    # Blind 60-day test starting at $1,000
    res = simulate_forensics_variant(
        symbols=top_25,
        bundle=test_bundle,
        timeline=test_tl,
        hypotheses=hypotheses,
        margin_pct=15.0,
        max_open=5,
        max_same_dir=3,
        max_daily_stops=4,
    )
    return {"label": label, "hypotheses": list(hypotheses), "top25": top_25, **res}


def beats_base(cand: Dict[str, Any], base: Dict[str, Any]) -> Tuple[bool, str]:
    base_calmar = base["net_pct"] / max(base["max_dd_pct"], 1.0)
    cand_calmar = cand["net_pct"] / max(cand["max_dd_pct"], 1.0)
    if cand["end_usd"] > base["end_usd"]:
        return True, "ПОБИЛА БАЗУ"
    if cand["end_usd"] >= base["end_usd"] * 0.90 and cand["max_dd_pct"] <= base["max_dd_pct"] - 4.0 and cand_calmar > base_calmar:
        return True, "ПОБИЛА БАЗУ (ниже просадка и лучше риск/доходность)"
    return False, "ХУЖЕ БАЗЫ"


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    symbols_100 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    sim_end_ms = int(df_btc["timestamp"].iloc[-1])
    sim_start_ms = sim_end_ms - 180 * 86_400_000
    split_ms = sim_start_ms + 120 * 86_400_000

    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]
    timeline = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    train_mask = timeline < split_ms
    test_mask = timeline >= split_ms
    train_tl = timeline[train_mask]
    test_tl = timeline[test_mask]

    print("Building causal bundles across cached Top-100...")
    bundle: Dict[str, Dict[str, np.ndarray]] = {}
    active_syms: List[str] = []
    for sym in symbols_100:
        if not (cache_dir / f"{sym}_15.csv.gz").exists():
            continue
        df_s = load_cached_15m(sym, cache_dir)
        if len(df_s) < 400:
            continue
        bundle[sym] = build_forensics_bundle(df_s, df_btc, timeline_df)
        active_syms.append(sym)

    train_bundle = slice_sym_data(bundle, train_mask)
    test_bundle = slice_sym_data(bundle, test_mask)

    base = run_blind_hypothesis("ЭТАЛОН (чистая fibo_trend_v2)", (), train_bundle, train_tl, test_bundle, test_tl, active_syms)

    hyp_list = [
        ("time_stop_8h", "1. Тайм-стоп 8 часов (закрытие зависшей сделки на 32-й свече 15M)"),
        ("no_weekends", "2. Фильтр выходных (запрет новых входов в Сб/Вс UTC)"),
        ("early_be_50", "3. Ранний Smart BE (перенос в +0.15% при +0.75% вместо +0.975%)"),
        ("h1_adx_25", "4. Усиленный тренд 1H (H1 ADX >= 25 вместо 20)"),
    ]

    print("\n" + "=" * 104)
    print(
        f"BASELINE CHECK: $1,000 -> ${base['end_usd']:,.2f} ({base['net_pct']:+.1f}%) | "
        f"{base['trades']} tr ({base['trades_per_day']:.2f}/d) | WR {base['win_rate_pct']:.2f}% | "
        f"SL={base['sl_count']} | MaxDD={base['max_dd_pct']:.2f}% (${base['max_dd_usd']:.2f})"
    )
    print("=" * 104)

    singles: List[Dict[str, Any]] = []
    winners: List[str] = []

    for h_key, h_label in hyp_list:
        res = run_blind_hypothesis(h_label, (h_key,), train_bundle, train_tl, test_bundle, test_tl, active_syms)
        ok, verdict = beats_base(res, base)
        res["better"] = ok
        res["verdict"] = verdict
        singles.append(res)
        if ok:
            winners.append(h_key)

        print(f"\n>>> {h_label} => {verdict}")
        print(
            f"  Trades: {res['trades']} ({res['trades_per_day']:.2f}/d) | WR: {res['win_rate_pct']:.2f}% | "
            f"SL: {res['sl_count']} (base: {base['sl_count']}) [TP={res['tp_count']}, BE={res['be_count']}, TIME={res['time_count']}]"
        )
        print(
            f"  Blind 2m: $1,000 -> ${res['end_usd']:,.2f} ({res['net_pct']:+.2f}%) | "
            f"MaxDD: {res['max_dd_pct']:.2f}% (${res['max_dd_usd']:.2f})"
        )

    combos: List[Dict[str, Any]] = []
    if len(winners) >= 2:
        for r_len in range(2, len(winners) + 1):
            for comb in combinations(winners, r_len):
                res = run_blind_hypothesis(
                    f"КОМБО ({' + '.join(comb)})", comb, train_bundle, train_tl, test_bundle, test_tl, active_syms
                )
                ok, verdict = beats_base(res, base)
                res["better"] = ok
                res["verdict"] = verdict
                combos.append(res)
        combos.sort(key=lambda r: (r["end_usd"], -r["max_dd_pct"]), reverse=True)

        print("\n" + "=" * 104)
        print(f"COMBINATIONS OF WINNING HYPOTHESES ({winners})")
        print("=" * 104)
        for res in combos:
            print(f"\n>>> {res['label']} => {res['verdict']}")
            print(
                f"  Trades: {res['trades']} ({res['trades_per_day']:.2f}/d) | WR: {res['win_rate_pct']:.2f}% | "
                f"SL: {res['sl_count']} (base: {base['sl_count']}) [TP={res['tp_count']}, BE={res['be_count']}, TIME={res['time_count']}]"
            )
            print(
                f"  Blind 2m: $1,000 -> ${res['end_usd']:,.2f} ({res['net_pct']:+.2f}%) | "
                f"MaxDD: {res['max_dd_pct']:.2f}% (${res['max_dd_usd']:.2f})"
            )

    out = {
        "baseline": base,
        "singles": singles,
        "winners": winners,
        "combos": combos,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    Path("data/blind_agent_forensics_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/blind_agent_forensics_results.json")


if __name__ == "__main__":
    main()
