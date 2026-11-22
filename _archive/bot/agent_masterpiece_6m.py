"""POLYGRAPH 6-MONTH EXAM (`agent_masterpiece_6m.py`): 4 Fundamentally Different Architectures.

Strict Honesty Rules:
  - Universe: Uncurated Top-30 pure crypto Bybit USDT Perpetuals by volume (ZERO coin cherry-picking).
  - Start: $1,000 compounded, strictly 10x isolated margin, volatility-targeted risk sizing.
  - Real Bybit fees (Maker 0.02% / Taker 0.055%) + explicit 0.04% slippage on ALL market entries & SL/trailing exits.
  - Strictly closed candles, conservative intra-bar priority (SL/trail checked before TP).
  - Months 1-4 (d1-120) = Development || Months 5-6 (d121-180) = STRICTLY BLIND ONE-SHOT EXAM.
  - Disqualification: Max DD > 40% OR negative on Blind Months 5-6.
  - Does NOT touch `config.json` or `fibo_trend_v2`.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.blind_walkforward import load_cached_15m
from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import (
    compute_adx,
    compute_atr,
    compute_daily_vwap,
    compute_rsi,
    compute_rvol,
    ema,
)


def eff_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    return (
        (close - close.shift(period)).abs()
        / close.diff().abs().rolling(period, min_periods=period).sum().replace(0.0, np.nan)
    ).fillna(0.0)


def prepare_all_180d_features(
    symbols: List[str],
    cache_dir: Path,
    sim_start_ms: int,
    sim_end_ms: int,
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """Build strictly causal 15M + 1H + 4H features & signals for all 4 architectures."""
    entry_ms = INTERVAL_TO_MS["15"]
    h1_ms = INTERVAL_TO_MS["60"]
    h4_ms = INTERVAL_TO_MS["240"]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    btc_sim = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]
    timeline = np.sort(btc_sim["timestamp"].unique().astype(np.int64))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    btc_15 = df_btc[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc_15["btc_move"] = ((btc_15["close"] - btc_15["open"]).abs() / btc_15["open"].replace(0.0, np.nan)) * 100.0
    btc_15["btc_r1h"] = btc_15["close"].pct_change(4)

    btc_4h = BybitClient.resample_klines(df_btc, "240")
    b4_e50 = ema(btc_4h["close"], 50)
    b4_adx, _, _ = compute_adx(btc_4h["high"], btc_4h["low"], btc_4h["close"], 14)
    btc_4h_df = pd.DataFrame({
        "close_ts": (btc_4h["timestamp"] + h4_ms).astype("int64"),
        "btc_4h_bull": btc_4h["close"] > b4_e50,
        "btc_4h_bear": btc_4h["close"] < b4_e50,
        "btc_4h_adx": b4_adx.values,
    }).sort_values("close_ts")

    data: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in symbols:
        p = cache_dir / f"{sym}_15.csv.gz"
        if not p.exists():
            continue
        df = load_cached_15m(sym, cache_dir)
        if len(df) < 500:
            continue

        df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")
        df["rsi14"] = compute_rsi(df["close"], 14)
        df["atr14"] = compute_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = (df["atr14"] / df["close"].replace(0.0, np.nan)) * 100.0
        df["vwap"] = compute_daily_vwap(df)
        df["rvol"] = compute_rvol(df["volume"], 20)

        c_rng = (df["high"] - df["low"]).replace(0.0, np.nan)
        lo_wick = ((pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]) / c_rng).fillna(0.0)
        hi_wick = ((df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)) / c_rng).fillna(0.0)

        df = pd.merge_asof(df, btc_15[["timestamp", "btc_move", "btc_r1h"]], on="timestamp", direction="backward")
        df = pd.merge_asof(df, btc_4h_df, on="close_ts", direction="backward")

        # 1H strictly closed features
        df_1h = BybitClient.resample_klines(df, "60")
        h1_e20, h1_e50 = ema(df_1h["close"], 20), ema(df_1h["close"], 50)
        h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_atr = compute_atr(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_rvol = compute_rvol(df_1h["volume"], 20)
        h1_er = eff_ratio(df_1h["close"], 20)
        h1_hi36 = df_1h["high"].shift(1).rolling(36, min_periods=36).max()
        h1_lo36 = df_1h["low"].shift(1).rolling(36, min_periods=36).min()

        h1_df = pd.DataFrame({
            "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
            "h1_adx": h1_adx.values,
            "h1_er": h1_er.values,
            "h1_atr": h1_atr.values,
            "h1_atr_pct": ((h1_atr / df_1h["close"].replace(0.0, np.nan)) * 100.0).fillna(1.2).values,
            "h1_don_L": ((df_1h["close"] > h1_hi36) & (h1_e20 > h1_e50) & (h1_rvol >= 1.8) & (h1_adx >= 22.0)).values,
            "h1_don_S": ((df_1h["close"] < h1_lo36) & (h1_e20 < h1_e50) & (h1_rvol >= 1.8) & (h1_adx >= 22.0)).values,
            "h1_rvol": h1_rvol.values,
        }).sort_values("close_ts")

        df = pd.merge_asof(df, h1_df, on="close_ts", direction="backward")

        is_1h_close = (df["close_ts"] % h1_ms) == 0
        btc4_bull = df["btc_4h_bull"].astype("boolean").fillna(False)
        btc4_bear = df["btc_4h_bear"].astype("boolean").fillna(False)
        btc4_adx = df["btc_4h_adx"].fillna(0.0)
        h1_adx_s = df["h1_adx"].fillna(0.0)
        h1_er_s = df["h1_er"].fillna(0.0)
        h1_don_L = df["h1_don_L"].astype("boolean").fillna(False)
        h1_don_S = df["h1_don_S"].astype("boolean").fillna(False)
        vwap_dist = ((df["close"] - df["vwap"]) / df["vwap"].replace(0.0, np.nan) * 100.0).fillna(0.0)
        atr_ok = (df["atr_pct"] >= 0.15) & (df["atr_pct"] <= 1.80)

        # ------------------------------------------------------------------
        # 1. REGIME HYBRID: Trend (1H Donchian-36 Runner) + Chop (Extreme VWAP/RSI MR)
        # ------------------------------------------------------------------
        s1_tr_L = is_1h_close & atr_ok & h1_don_L & (h1_er_s >= 0.25) & btc4_bull
        s1_tr_S = is_1h_close & atr_ok & h1_don_S & (h1_er_s >= 0.25) & btc4_bear
        chop_ok = (h1_adx_s < 22.0) & (btc4_adx < 25.0) & (df["btc_move"].fillna(0.0) <= 0.8) & atr_ok
        s1_ch_L = chop_ok & (df["rsi14"] < 24.0) & (vwap_dist < -2.0) & (df["close"] > df["open"])
        s1_ch_S = chop_ok & (df["rsi14"] > 76.0) & (vwap_dist > 2.0) & (df["close"] < df["open"])

        df["s1_L"] = s1_tr_L | s1_ch_L
        df["s1_S"] = s1_tr_S | s1_ch_S
        df["s1_mode"] = np.where(s1_tr_L | s1_tr_S, 1, np.where(s1_ch_L | s1_ch_S, 2, 0))
        df["s1_Q"] = np.where(s1_tr_L | s1_tr_S, df["h1_rvol"].fillna(1.0) * 20.0, vwap_dist.abs() * 10.0)

        # ------------------------------------------------------------------
        # 2. SMART MONEY LIQUIDITY SWEEP + VWAP + CHANDELIER TRAILING
        # ------------------------------------------------------------------
        lo24 = df["low"].shift(1).rolling(24, min_periods=20).min()
        hi24 = df["high"].shift(1).rolling(24, min_periods=20).max()
        df["s2_L"] = (
            atr_ok & (df["btc_move"].fillna(0.0) <= 0.9)
            & (df["low"] < lo24 * 0.999) & (df["close"] > lo24)
            & (df["rvol"] >= 1.5) & (lo_wick >= 0.35) & (df["close"] >= df["open"])
            & (df["low"] <= df["vwap"] * 1.003) & btc4_bull
        )
        df["s2_S"] = (
            atr_ok & (df["btc_move"].fillna(0.0) <= 0.9)
            & (df["high"] > hi24 * 1.001) & (df["close"] < hi24)
            & (df["rvol"] >= 1.5) & (hi_wick >= 0.35) & (df["close"] <= df["open"])
            & (df["high"] >= df["vwap"] * 0.997) & btc4_bear
        )
        df["s2_Q"] = (df["rvol"] * 15.0 + pd.concat([lo_wick, hi_wick], axis=1).max(axis=1) * 30.0).fillna(0.0)

        # ------------------------------------------------------------------
        # 3. VOLATILITY BREAKOUT RUNNER: 1H Donchian-36 + ADX>=22 + Chandelier 3.0*ATR_1h
        # ------------------------------------------------------------------
        df["s3_L"] = is_1h_close & atr_ok & h1_don_L & btc4_bull
        df["s3_S"] = is_1h_close & atr_ok & h1_don_S & btc4_bear
        df["s3_Q"] = (df["h1_rvol"].fillna(1.0) * 15.0 + h1_adx_s).fillna(0.0)

        # ------------------------------------------------------------------
        # 4. QUANT JOKER: BTC Lead-Lag Beta Catch-Up + VWAP + BTC 4H
        # ------------------------------------------------------------------
        alt_r1h = df["close"].pct_change(4)
        lag = df["btc_r1h"] - alt_r1h
        df["s4_L"] = (
            atr_ok
            & (df["btc_r1h"] > 0.008) & (lag > 0.006)
            & (df["close"] > df["vwap"]) & (df["close"] > df["open"])
            & btc4_bull
        )
        df["s4_S"] = (
            atr_ok
            & (df["btc_r1h"] < -0.008) & (lag < -0.006)
            & (df["close"] < df["vwap"]) & (df["close"] < df["open"])
            & btc4_bear
        )
        df["s4_Q"] = lag.abs().fillna(0.0) * 1000.0

        cols = [
            "timestamp", "high", "low", "close", "h1_atr", "h1_atr_pct",
            "s1_L", "s1_S", "s1_mode", "s1_Q",
            "s2_L", "s2_S", "s2_Q",
            "s3_L", "s3_S", "s3_Q",
            "s4_L", "s4_S", "s4_Q",
        ]
        al = timeline_df.merge(df[cols], on="timestamp", how="left")
        data[sym] = {
            "high": al["high"].to_numpy(dtype=np.float64),
            "low": al["low"].to_numpy(dtype=np.float64),
            "close": al["close"].to_numpy(dtype=np.float64),
            "h1_atr": al["h1_atr"].ffill().fillna(0.0).to_numpy(dtype=np.float64),
            "h1_atr_pct": al["h1_atr_pct"].ffill().fillna(1.2).to_numpy(dtype=np.float64),
            "s1_L": al["s1_L"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s1_S": al["s1_S"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s1_mode": al["s1_mode"].fillna(0).to_numpy(dtype=np.int8),
            "s1_Q": al["s1_Q"].fillna(0.0).to_numpy(dtype=np.float64),
            "s2_L": al["s2_L"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s2_S": al["s2_S"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s2_Q": al["s2_Q"].fillna(0.0).to_numpy(dtype=np.float64),
            "s3_L": al["s3_L"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s3_S": al["s3_S"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s3_Q": al["s3_Q"].fillna(0.0).to_numpy(dtype=np.float64),
            "s4_L": al["s4_L"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s4_S": al["s4_S"].astype("boolean").fillna(False).to_numpy(dtype=bool),
            "s4_Q": al["s4_Q"].fillna(0.0).to_numpy(dtype=np.float64),
        }
    return timeline, data


@dataclass
class StrategySpec:
    sid: str
    name: str
    summary: str
    risk_pct: float
    max_open: int
    max_same_dir: int
    daily_loss_pct: float
    entry_maker_modes: Tuple[int, ...]  # modes using Maker entry; others use Taker + 0.04% slip


def simulate_6m_polygraph(
    spec: StrategySpec,
    symbols: List[str],
    data: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    month_bounds: List[Tuple[str, int, int]],
    blind_split_ms: int,
    initial_deposit: float = 1000.0,
) -> Dict[str, Any]:
    mk_fee = 0.0002
    tk_fee = 0.00055
    slip = 0.0004  # 0.04% realistic slippage on EVERY market entry & SL/trailing exit!
    pb_frac = 0.0005

    l_k, s_k, q_k = f"{spec.sid}_L", f"{spec.sid}_S", f"{spec.sid}_Q"

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_pct = 0.0

    # Separate tracking for Blind Months 5-6 (Days 121-180) starting from normalized $1,000
    blind_eq = float(initial_deposit)
    blind_peak = blind_eq
    blind_max_dd = 0.0

    open_pos: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    cd_until: Dict[str, int] = {}
    trades: List[Dict[str, Any]] = []

    cur_day, day_start_eq, halted = -1, equity, False
    m_start_eq: Dict[str, float] = {}
    m_end_eq: Dict[str, float] = {}

    def _open_pos(sym: str, direction: str, raw_ep: float, is_mk: bool, h1_atr: float, h1_atr_pct: float, mode: int, lo_b: float, hi_b: float, ts: int, bar_i: int) -> None:
        is_l = direction == "LONG"
        ep = raw_ep if is_mk else (raw_ep * (1.0 + slip) if is_l else raw_ep * (1.0 - slip))

        if spec.sid == "s1":
            if mode == 1:  # Trend runner
                sl_f, tp_f, trail_m = 0.035, 99.0, 3.0
            else:          # Chop MR
                sl_f, tp_f, trail_m = 0.025, 0.020, 0.0
        elif spec.sid == "s2":  # Liquidity Sweep + VWAP + ATR trail
            sl_f, tp_f, trail_m = 0.025, 99.0, 2.8
        elif spec.sid == "s3":  # 1H Donchian-36 ATR Runner
            sl_f, tp_f, trail_m = 0.035, 99.0, 3.0
        else:  # s4: BTC Lead-Lag Beta Catch-Up
            sl_f, tp_f, trail_m = 0.025, 0.030, 0.0

        sl_p = ep * (1.0 - sl_f) if is_l else ep * (1.0 + sl_f)
        tp_p = ep * (1.0 + tp_f) if is_l else ep * (1.0 - tp_f)
        ef_r = mk_fee if is_mk else tk_fee

        # Volatility-targeted sizing strictly at 10x isolated margin
        eff_loss = sl_f + ef_r + tk_fee + slip
        notional_t = (equity * (spec.risk_pct / 100.0)) / max(eff_loss, 1e-6)
        margin = float(np.clip(notional_t / 10.0, equity * 0.05, equity * 0.16))
        if sum(x["margin"] for x in open_pos) + margin > equity * 0.85:
            return
        notional = margin * 10.0
        pos = {
            "sym": sym, "dir": direction, "ep": ep, "sl_p": sl_p, "tp_p": tp_p,
            "has_tp": tp_f < 50.0, "trail_m": trail_m, "peak_p": ep,
            "qty": notional / ep, "margin": margin, "efee": notional * ef_r,
            "eq_at_entry": equity, "is_blind": ts >= blind_split_ms,
        }
        if is_mk and ((lo_b <= sl_p) if is_l else (hi_b >= sl_p)):
            _close_pos(pos, sl_p, "SL", False, ts, bar_i)
        else:
            open_pos.append(pos)

    def _close_pos(p: Dict[str, Any], trig_p: float, reason: str, mk_exit: bool, ts: int, bar_i: int) -> None:
        nonlocal equity, peak_eq, max_dd_pct, blind_eq, blind_peak, blind_max_dd
        is_l = p["dir"] == "LONG"
        exec_p = trig_p if mk_exit else (trig_p * (1.0 - slip) if is_l else trig_p * (1.0 + slip))
        gross = (exec_p - p["ep"]) * p["qty"] if is_l else (p["ep"] - exec_p) * p["qty"]
        ex_fee = abs(p["qty"] * exec_p) * (mk_fee if mk_exit else tk_fee)
        net = max(-p["margin"], gross - p["efee"] - ex_fee)
        ret_frac = net / max(p["eq_at_entry"], 1.0)

        equity = max(1.0, equity + net)
        if equity > peak_eq:
            peak_eq = equity
        dd = ((peak_eq - equity) / peak_eq) * 100.0
        if dd > max_dd_pct:
            max_dd_pct = dd

        if p["is_blind"]:
            blind_eq = max(1.0, blind_eq * (1.0 + ret_frac))
            if blind_eq > blind_peak:
                blind_peak = blind_eq
            bdd = ((blind_peak - blind_eq) / blind_peak) * 100.0
            if bdd > blind_max_dd:
                blind_max_dd = bdd

        trades.append({"ts": ts, "is_blind": p["is_blind"], "reason": reason, "net": net, "ret_pct": ret_frac * 100.0})
        cd_until[p["sym"]] = bar_i + 4

    for i in range(len(timeline)):
        ts = int(timeline[i])
        for mn, ms, me in month_bounds:
            if mn not in m_start_eq and ts >= ms:
                m_start_eq[mn] = equity
            if ts <= me:
                m_end_eq[mn] = equity

        dk = ts // 86_400_000
        if dk != cur_day:
            cur_day, day_start_eq, halted = dk, equity, False

        if open_pos:
            still = []
            for p in open_pos:
                sd = data[p["sym"]]
                hi, lo = sd["high"][i], sd["low"][i]
                h1_a = sd["h1_atr"][i]
                if not np.isfinite(lo):
                    still.append(p); continue
                is_l = p["dir"] == "LONG"
                if (lo <= p["sl_p"]) if is_l else (hi >= p["sl_p"]):
                    _close_pos(p, p["sl_p"], "SL_OR_TRAIL", False, ts, i)
                elif p["has_tp"] and ((hi >= p["tp_p"]) if is_l else (lo <= p["tp_p"])):
                    _close_pos(p, p["tp_p"], "TP", True, ts, i)
                else:
                    if p["trail_m"] > 0 and h1_a > 0:
                        if is_l:
                            if hi > p["peak_p"]: p["peak_p"] = hi
                            p["sl_p"] = max(p["sl_p"], p["peak_p"] - p["trail_m"] * h1_a)
                        else:
                            if lo < p["peak_p"]: p["peak_p"] = lo
                            p["sl_p"] = min(p["sl_p"], p["peak_p"] + p["trail_m"] * h1_a)
                    still.append(p)
            open_pos = still

        if not halted and open_pos:
            unreal = sum(
                ((data[p["sym"]]["close"][i] - p["ep"]) * p["qty"] if p["dir"] == "LONG" else (p["ep"] - data[p["sym"]]["close"][i]) * p["qty"]) - p["efee"]
                for p in open_pos if np.isfinite(data[p["sym"]]["close"][i])
            )
            if ((day_start_eq - (equity + unreal)) / max(day_start_eq, 1.0)) * 100.0 >= spec.daily_loss_pct:
                halted = True

        if pending:
            npend = []
            for o in pending:
                if i > o["exp"] or halted or any(x["sym"] == o["sym"] for x in open_pos) or i < cd_until.get(o["sym"], -1):
                    continue
                sd = data[o["sym"]]
                hi, lo = sd["high"][i], sd["low"][i]
                if not np.isfinite(lo):
                    if i < o["exp"]: npend.append(o)
                    continue
                is_l = o["dir"] == "LONG"
                if not ((lo < o["lp"]) if is_l else (hi > o["lp"])):
                    if i < o["exp"]: npend.append(o)
                    continue
                if len(open_pos) >= spec.max_open or sum(1 for x in open_pos if x["dir"] == o["dir"]) >= spec.max_same_dir:
                    continue
                _open_pos(o["sym"], o["dir"], o["lp"], True, o["h1a"], o["h1ap"], o["mode"], lo, hi, ts, i)
            pending = npend

        if halted or len(open_pos) >= spec.max_open:
            continue

        busy = {x["sym"] for x in open_pos} | {x["sym"] for x in pending}
        cands = []
        for sym in symbols:
            if sym in busy or i < cd_until.get(sym, -1):
                continue
            sd = data[sym]
            if sd[l_k][i]:
                m = int(sd["s1_mode"][i]) if spec.sid == "s1" else 0
                cands.append((float(sd[q_k][i]), sym, "LONG", float(sd["close"][i]), float(sd["h1_atr"][i]), float(sd["h1_atr_pct"][i]), m))
            elif sd[s_k][i]:
                m = int(sd["s1_mode"][i]) if spec.sid == "s1" else 0
                cands.append((float(sd[q_k][i]), sym, "SHORT", float(sd["close"][i]), float(sd["h1_atr"][i]), float(sd["h1_atr_pct"][i]), m))
        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)
        for _, sym, d, cl, h1a, h1ap, mode in cands:
            if len(open_pos) >= spec.max_open:
                break
            if sum(1 for x in open_pos if x["dir"] == d) >= spec.max_same_dir:
                continue
            if mode in spec.entry_maker_modes:
                if len(open_pos) + len(pending) >= spec.max_open + 3:
                    break
                lp = cl * (1.0 - pb_frac) if d == "LONG" else cl * (1.0 + pb_frac)
                pending.append({"sym": sym, "dir": d, "lp": lp, "exp": i + 2, "h1a": h1a, "h1ap": h1ap, "mode": mode})
            else:
                _open_pos(sym, d, cl, False, h1a, h1ap, mode, cl, cl, ts, i)

    if open_pos and len(timeline) > 0:
        last_ts = int(timeline[-1])
        for p in open_pos:
            cl = data[p["sym"]]["close"][-1]
            _close_pos(p, float(cl) if np.isfinite(cl) else p["ep"], "EOD", False, last_ts, len(timeline) - 1)
        for mn, _, me in month_bounds:
            if last_ts <= me:
                m_end_eq[mn] = equity

    mp = {
        mn: round(((m_end_eq.get(mn, 1000.0) - m_start_eq.get(mn, 1000.0)) / max(m_start_eq.get(mn, 1000.0), 1.0)) * 100.0, 2)
        for mn, _, _ in month_bounds
    }

    blind_trades = [t for t in trades if t["is_blind"]]
    n_tot = len(trades)
    n_bl = len(blind_trades)
    bl_wr = round((sum(1 for t in blind_trades if t["net"] > 0) / n_bl) * 100.0, 2) if n_bl else 0.0
    bl_monthly_pace = round(((blind_eq / 1000.0) ** 0.5 - 1.0) * 100.0, 2)

    passed = bool((blind_eq > 1000.0) and (max_dd_pct <= 40.0))
    return {
        "sid": spec.sid,
        "name": spec.name,
        "summary": spec.summary,
        "passed": passed,
        "status": "ПРОШЛА СЛЕПОЙ ЭКЗАМЕН" if passed else "ПРОВАЛ",
        "monthly_pct": mp,
        "end_6m_usd": round(equity, 2),
        "blind_2m_usd": round(blind_eq, 2),
        "blind_monthly_pace_pct": bl_monthly_pace,
        "trades_6m": n_tot,
        "trades_per_day": round(n_tot / 180.0, 2),
        "blind_trades": n_bl,
        "blind_wr_pct": bl_wr,
        "max_dd_6m_pct": round(max_dd_pct, 2),
        "blind_max_dd_pct": round(blind_max_dd, 2),
    }


def main() -> None:
    t0 = time.time()
    cache_dir = Path("data/cache")
    symbols_100 = json.loads(Path("data/strategy_arena_results.json").read_text(encoding="utf-8"))["meta"]["symbols_100"]
    top30 = symbols_100[:30]

    df_btc = load_cached_15m("BTCUSDT", cache_dir)
    day_ms = 86_400_000
    sim_end = int(df_btc["timestamp"].iloc[-1])
    sim_start = sim_end - 180 * day_ms
    blind_split = sim_start + 120 * day_ms

    month_bounds = [
        ("M1", sim_start + 0 * day_ms, sim_start + 30 * day_ms - 1),
        ("M2", sim_start + 30 * day_ms, sim_start + 60 * day_ms - 1),
        ("M3", sim_start + 60 * day_ms, sim_start + 90 * day_ms - 1),
        ("M4", sim_start + 90 * day_ms, sim_start + 120 * day_ms - 1),
        ("M5", sim_start + 120 * day_ms, sim_start + 150 * day_ms - 1),
        ("M6", sim_start + 150 * day_ms, sim_end),
    ]

    print("Preparing 180-day causal features on uncurated Top-30 pure crypto...")
    tl, data = prepare_all_180d_features(top30, cache_dir, sim_start, sim_end)
    syms = list(data.keys())

    specs = [
        StrategySpec(
            sid="s1",
            name="1. Гибрид «Тренд + Боковик» (Regime Switching)",
            summary="В тренде (H1 ADX>=22, ER>=0.25) торгует 1H пробой Donchian-36 с трейлингом 3*ATR, а в боковике (ADX<22) переключается на возврат к VWAP (RSI <24/>76, Maker).",
            risk_pct=2.0, max_open=4, max_same_dir=2, daily_loss_pct=6.5, entry_maker_modes=(2,),
        ),
        StrategySpec(
            sid="s2",
            name="2. Smart Money: Liquidity Sweep + VWAP Trail",
            summary="Ловит вынос 24-свечного экстремума 15M на объёме RVOL>=1.5 с возвратом внутрь диапазона у VWAP в сторону BTC 4H (Maker-вход + трейлинг 2.8*ATR_1h).",
            risk_pct=2.0, max_open=4, max_same_dir=2, daily_loss_pct=6.5, entry_maker_modes=(0,),
        ),
        StrategySpec(
            sid="s3",
            name="3. Волатильностный пробой (1H Donchian-36 ATR Runner)",
            summary="Пробой 36-часового канала Donchian на 1H при ADX>=22, RVOL>=1.8 и тренде BTC 4H без потолка прибыли — полная ловля трендов Chandelier Trailing 3*ATR_1h (риск 2% / 10x).",
            risk_pct=2.0, max_open=5, max_same_dir=3, daily_loss_pct=7.0, entry_maker_modes=(),
        ),
        StrategySpec(
            sid="s4",
            name="4. Джокер: BTC Lead-Lag Beta Catch-Up",
            summary="Ловит запаздывание ликвидных альткоинов (>0.6% лаг у VWAP) за резким часовым импульсом BTC (>0.8% по тренду 4H), TP 3.0% Maker / SL 2.5% Taker (риск 2.5% / 10x).",
            risk_pct=2.5, max_open=4, max_same_dir=3, daily_loss_pct=7.5, entry_maker_modes=(),
        ),
    ]

    results = []
    print("\n" + "=" * 114)
    print("ONE-SHOT 6-MONTH POLYGRAPH EXAM | UNCURATED TOP-30 CRYPTO | 10x ISOLATED | BYBIT FEES + 0.04% SLIPPAGE")
    print("=" * 114)
    for sp in specs:
        r = simulate_6m_polygraph(sp, syms, data, tl, month_bounds, blind_split)
        results.append(r)
        mp = r["monthly_pct"]
        print(f"\n>>> {r['name']} => [{r['status']}]")
        print(f"    {r['summary']}")
        print(f"    M1: {mp['M1']:>+6.1f}% | M2: {mp['M2']:>+6.1f}% | M3: {mp['M3']:>+6.1f}% | M4: {mp['M4']:>+6.1f}% || BLIND M5: {mp['M5']:>+6.1f}% | BLIND M6: {mp['M6']:>+6.1f}%")
        print(f"    6m ($1k -> ${r['end_6m_usd']:,.2f}) | BLIND 2m ($1k -> ${r['blind_2m_usd']:,.2f}, avg {r['blind_monthly_pace_pct']:+.1f}%/mo)")
        print(f"    Trades: {r['trades_per_day']:.2f}/d (6m={r['trades_6m']}, blind={r['blind_trades']}) | Blind WR: {r['blind_wr_pct']:.1f}% | MaxDD 6m: {r['max_dd_6m_pct']:.1f}% (Blind DD: {r['blind_max_dd_pct']:.1f}%)")

    Path("data/agent_masterpiece_6m_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone in {time.time() - t0:.2f}s -> saved to data/agent_masterpiece_6m_results.json")


if __name__ == "__main__":
    main()
