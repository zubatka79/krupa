"""Market scanner and 0-100 scoring engine strictly on closed 15M Bybit candles."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from bot.bybit_client import BybitClient, INTERVAL_TO_MS, InstrumentSpec, round_tick
from bot.indicators import (
    check_volume_profile_clearance,
    compute_adx,
    compute_atr,
    compute_rsi,
    ema,
    enrich_entry_dataframe,
)
from bot.news_guard import NewsGuard, read_news_state
from bot.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    timestamp: int
    entry_price: float
    sl_pct: float
    score: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    atr_pct: float = 0.0
    rvol: float = 0.0
    body_ratio: float = 0.0
    spec: Optional[InstrumentSpec] = None
    tp_price: Optional[float] = None
    tp_pct: Optional[float] = None
    be1_trigger_pct: Optional[float] = None
    be1_offset_pct: Optional[float] = None
    be2_trigger_pct: Optional[float] = None
    be2_offset_pct: Optional[float] = None


# Strategies served by the generic scoring path (score_closed_bar) or fibo_trend_v2
KNOWN_MULTI_COIN_STRATEGIES = ("fibo_trend_v2", "generic_score")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _flag(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def load_blacklist(path: str | Path = "data/blacklist.json") -> Tuple[Set[str], Optional[str]]:
    """Load active (non-expired) blacklisted symbols and optional macro pause reason."""
    p = Path(path)
    if not p.exists():
        return set(), None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("root is not an object")
    except Exception as exc:
        # An unreadable file may hide a manual or macro pause, so treat it as a pause
        logger.error("Failed to parse blacklist %s: %s", p, exc)
        return set(), f"blacklist unreadable: {exc}"

    now = datetime.now(timezone.utc)
    active: Set[str] = set()

    entries = data.get("entries", [])
    if isinstance(entries, list) and entries:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("symbol", "")).upper().strip()
            if not sym:
                continue
            exp = entry.get("expires_at")
            if exp:
                try:
                    exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
                    if exp_dt.tzinfo is None:
                        exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                    if exp_dt <= now:
                        continue
                except (TypeError, ValueError):
                    pass
            active.add(sym)
    else:
        for sym in data.get("symbols", []):
            s = str(sym).upper().strip()
            if s:
                active.add(s)

    macro_pause = data.get("macro_pause_reason")
    return active, macro_pause


def compute_technical_sl_pct(
    direction: str,
    close: float,
    swing_low_10: float,
    swing_high_10: float,
    atr: float,
    atr_pct: float,
    sl_min_pct: float,
    sl_max_pct: float,
) -> Optional[float]:
    """
    Mandatory SL filter:
    Calculated technical SL must fit in [sl_min_pct, sl_max_pct] (1.2%-1.6%)
    and be >= 1.1 * ATR(14). Returns valid sl_pct or None if rejected.
    """
    if close <= 0 or not np.isfinite(atr_pct) or atr_pct <= 0:
        return None

    atr_floor_pct = 1.1 * atr_pct
    if atr_floor_pct > sl_max_pct:
        return None

    if direction == "LONG":
        tech_sl_price = swing_low_10 - 0.15 * atr
        tech_sl_pct = ((close - tech_sl_price) / close) * 100.0
    else:
        tech_sl_price = swing_high_10 + 0.15 * atr
        tech_sl_pct = ((tech_sl_price - close) / close) * 100.0

    # Reject if structural stop is wider than sl_max_pct or market is completely dead/flat
    if tech_sl_pct > sl_max_pct or tech_sl_pct < 0.45:
        return None

    sl_pct = max(sl_min_pct, tech_sl_pct, atr_floor_pct)
    if sl_pct < sl_min_pct or sl_pct > sl_max_pct:
        return None
    return round(float(sl_pct), 4)


def score_closed_bar(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    config: Dict[str, Any],
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    oi_drop_pct: Optional[float] = None,
    funding_rate: Optional[float] = None,
) -> Optional[Tuple[float, float, Dict[str, float]]]:
    """
    Score a single closed 15M candle `idx` in `direction` ("LONG" or "SHORT").
    Returns (total_score, sl_pct, breakdown) if all mandatory filters pass and
    total_score >= min_score_threshold, else None.
    """
    row = df.iloc[idx]

    # Mandatory filter 1: BTC shock pause
    btc_shock_limit = float(config.get("btc_shock_pause_pct", 0.8))
    if float(row["btc_bar_move_pct"]) > btc_shock_limit:
        return None

    close = float(row["close"])
    atr = float(row["atr"])
    atr_pct = float(row["atr_pct"])
    sl_min = float(config.get("sl_min_pct", 1.2))
    sl_max = float(config.get("sl_max_pct", 1.6))

    # Mandatory filter 2: Technical SL in [1.2%, 1.6%] and >= 1.1 * ATR(14)
    sl_pct = compute_technical_sl_pct(
        direction=direction,
        close=close,
        swing_low_10=float(row["swing_low_10"]),
        swing_high_10=float(row["swing_high_10"]),
        atr=atr,
        atr_pct=atr_pct,
        sl_min_pct=sl_min,
        sl_max_pct=sl_max,
    )
    if sl_pct is None:
        return None

    min_threshold = float(config.get("min_score_threshold", 70))
    is_long = direction == "LONG"

    # 1. Liquidity Sweep or Daily VWAP retest/hold (+25 pts)
    sweep_flag = bool(row["bull_sweep"] if is_long else row["bear_sweep"])
    vwap_retest = bool(row["bull_vwap_retest"] if is_long else row["bear_vwap_retest"])
    vwap_near = bool(row["bull_vwap_near"] if is_long else row["bear_vwap_near"])
    dir_candle = bool(row["is_bull_candle"] if is_long else row["is_bear_candle"])

    if sweep_flag or vwap_retest:
        pts_liq_vwap = 25.0
    elif vwap_near and dir_candle:
        pts_liq_vwap = 12.0
    else:
        pts_liq_vwap = 0.0

    # 2. RSI + MFI Divergence OR OI Flush / Funding Skew (+20 pts)
    # Graceful weight redistribution if OI is unavailable on long historical backtests
    div_both = bool(row["div_bull_both"] if is_long else row["div_bear_both"])
    div_single = bool(row["div_bull_single"] if is_long else row["div_bear_single"])
    osc_turn = bool(row["bull_osc_turn"] if is_long else row["bear_osc_turn"])

    oi_flush = oi_drop_pct is not None and oi_drop_pct >= 1.5
    fr_skew = False
    if funding_rate is not None:
        fr_skew = (funding_rate <= -0.0002) if is_long else (funding_rate >= 0.0003)

    if oi_drop_pct is not None:
        # Full live/short-history mode with OI available
        if div_both or oi_flush or fr_skew:
            pts_div_crowd = 20.0
        elif div_single or osc_turn:
            pts_div_crowd = 12.0
        else:
            pts_div_crowd = 0.0
    else:
        # Historical mode where 15m OI is unavailable: cleanly redistribute OI sub-weight
        # across divergence, funding skew (if available), and confirmed sweep+momentum alignment
        if div_both or fr_skew:
            pts_div_crowd = 20.0
        elif div_single:
            pts_div_crowd = 15.0
        elif osc_turn:
            pts_div_crowd = 12.0
        elif sweep_flag and dir_candle:
            # Redistributed proxy for stop-hunt / liquidity flush when historical OI isn't available
            pts_div_crowd = 10.0
        else:
            pts_div_crowd = 0.0

    # 3. Relative Strength vs BTC (+15 pts)
    rs_4 = float(row["rs_4"])
    rs_16 = float(row["rs_16"])
    if is_long:
        rs_both = rs_4 > 0.0 and rs_16 > 0.0
        rs_one = rs_4 > 0.0 or rs_16 > 0.0
    else:
        rs_both = rs_4 < 0.0 and rs_16 < 0.0
        rs_one = rs_4 < 0.0 or rs_16 < 0.0

    if rs_both:
        pts_rs = 15.0
    elif rs_one:
        pts_rs = 8.0
    else:
        pts_rs = 0.0

    # 4. Signal Candle Anatomy: RVOL in [1.5, 2.5] & body >= 55% in direction (+15 pts)
    rvol = float(row["rvol"])
    body_ratio = float(row["body_ratio"])
    if dir_candle and (1.5 <= rvol <= 2.5) and (body_ratio >= 0.55):
        pts_candle = 15.0
    elif dir_candle and (1.2 <= rvol <= 3.2) and (body_ratio >= 0.50):
        pts_candle = 10.0
    elif dir_candle and (rvol >= 1.0) and (body_ratio >= 0.45):
        pts_candle = 5.0
    else:
        pts_candle = 0.0

    # 6. 1H / 4H Context Alignment + ADX Trend Strength (+10 pts)
    h1_ok = bool(row["h1_bull"] if is_long else row["h1_bear"])
    h4_ok = bool(row["h4_bull"] if is_long else row["h4_bear"])
    h1_opp = bool(row["h1_bear"] if is_long else row["h1_bull"])
    h4_opp = bool(row["h4_bear"] if is_long else row["h4_bull"])
    htf_adx = max(float(row["h1_adx"]), float(row["h4_adx"]))
    adx_15m = float(row["adx_15m"])

    if h1_ok and h4_ok and htf_adx >= 20.0:
        pts_htf = 10.0
    elif (h1_ok or h4_ok) and htf_adx >= 18.0:
        pts_htf = 6.0
    elif not (h1_opp and h4_opp) and adx_15m >= 22.0:
        pts_htf = 3.0
    else:
        pts_htf = 0.0

    # Fast prune before horizontal volume profile computation (max VPVR contribution = +15)
    pre_score = pts_liq_vwap + pts_div_crowd + pts_rs + pts_candle + pts_htf
    if pre_score + 15.0 < min_threshold:
        return None

    # 5. Horizontal Volume Profile Clearance to TP1 / TP2 (+15 pts)
    tp1_frac = float(config.get("tp1_pct", 0.8)) / 100.0
    tp2_frac = float(config.get("tp2_pct", 1.6)) / 100.0
    if is_long:
        tp1_price = close * (1.0 + tp1_frac)
        tp2_price = close * (1.0 + tp2_frac)
    else:
        tp1_price = close * (1.0 - tp1_frac)
        tp2_price = close * (1.0 - tp2_frac)

    clear_tp1, clear_tp2 = check_volume_profile_clearance(
        highs=highs,
        lows=lows,
        volumes=volumes,
        idx=idx,
        direction=direction,
        entry_price=close,
        tp1_price=tp1_price,
        tp2_price=tp2_price,
    )
    if clear_tp2:
        pts_vpvr = 15.0
    elif clear_tp1:
        pts_vpvr = 8.0
    else:
        pts_vpvr = 0.0

    total_score = pre_score + pts_vpvr
    if total_score < min_threshold:
        return None

    breakdown = {
        "liq_vwap": pts_liq_vwap,
        "div_crowd": pts_div_crowd,
        "rel_strength_btc": pts_rs,
        "candle_anatomy": pts_candle,
        "volume_profile": pts_vpvr,
        "htf_adx": pts_htf,
    }
    return round(total_score, 2), sl_pct, breakdown


class MarketScanner:
    """Live scanner over whitelist or top-100 Bybit USDT perpetuals."""

    def __init__(
        self,
        client: BybitClient,
        risk_manager: RiskManager,
        config: Dict[str, Any],
        blacklist_path: str | Path = "data/blacklist.json",
        news_state_path: str | Path = "data/news_state.json",
        universe_cache_path: str | Path = "data/universe_cache.json",
        news_guard: Optional[NewsGuard] = None,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.config = config
        self.blacklist_path = Path(blacklist_path)
        self.news_state_path = Path(news_state_path)
        self.universe_cache_path = Path(universe_cache_path)
        self.news_guard = news_guard
        # Filtered setups of the last scan_once call, for the signal journal
        self.journal_events: List[Dict[str, Any]] = []
        self._journal_bar_ts = 0

    def update_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.risk_manager.update_config(config)

    def _note(self, outcome: str, reason: str, symbol: str = "*", direction: str = "", details: Optional[Dict[str, Any]] = None) -> None:
        self.journal_events.append(
            {
                "bar_ts_ms": self._journal_bar_ts,
                "symbol": symbol,
                "direction": direction,
                "outcome": outcome,
                "reason": reason,
                "details": details or {},
            }
        )

    def _blocked(self, reason: str, spec: InstrumentSpec, direction: str, msg: str, *args: Any) -> None:
        logger.info(msg, *args)
        self._note("FILTERED", reason, spec.symbol, direction, {"message": msg % args})

    def get_universe(self, top_n: Optional[int] = None, crypto_only: bool = True) -> List[InstrumentSpec]:
        """
        Use strict whitelist if configured; otherwise maintain a 30-day cached
        Top-N pure crypto liquid universe (`data/universe_cache.json`) that
        auto-refreshes every `universe_refresh_days` without stopping trading.
        """
        enforce_crypto = bool(self.config.get("crypto_only", crypto_only))
        whitelist = [str(s).upper().strip() for s in self.config.get("whitelist", []) if str(s).strip()]
        if whitelist:
            specs_map = self.client.get_instrument_specs_for_symbols(whitelist)
            specs = [specs_map[s] for s in whitelist if s in specs_map]
            return [s for s in specs if (not enforce_crypto or s.is_pure_crypto)]

        target_n = int(top_n if top_n is not None else self.config.get("top_n_liquid", 25))
        refresh_days = int(self.config.get("universe_refresh_days", 30))
        now = datetime.now(timezone.utc)

        cached_syms: Optional[List[str]] = None
        if self.universe_cache_path.exists():
            try:
                u_data = json.loads(self.universe_cache_path.read_text(encoding="utf-8"))
                upd = datetime.fromisoformat(str(u_data.get("updated_at", "")).replace("Z", "+00:00"))
                syms = u_data.get("symbols", [])
                if syms and len(syms) == target_n and (now - upd) < timedelta(days=refresh_days):
                    cached_syms = [str(s).upper() for s in syms]
            except Exception as exc:
                logger.warning("Universe cache unreadable (%s); refreshing live Top-%d.", exc, target_n)

        if cached_syms:
            specs_map = self.client.get_instrument_specs_for_symbols(cached_syms)
            specs = [specs_map[s] for s in cached_syms if s in specs_map and (not enforce_crypto or specs_map[s].is_pure_crypto)]
            if len(specs) >= max(10, target_n - 3):
                return specs

        fresh_raw = self.client.get_top_usdt_perpetuals(
            top_n=target_n + 5,
            max_min_order_usdt=25.0,
            crypto_only=enforce_crypto,
        )
        fresh = [s for s in fresh_raw if s.symbol != "BTCUSDT"][:target_n]
        if fresh:
            self.universe_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "refresh_days": refresh_days,
                "crypto_only": enforce_crypto,
                "symbols": [s.symbol for s in fresh],
            }
            tmp = self.universe_cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self.universe_cache_path)
            logger.info("Refreshed liquid pure-crypto Top-%d universe (next auto-refresh in %dd).", len(fresh), refresh_days)
        return fresh

    def allowed_direction_list(self) -> List[str]:
        mode = str(self.config.get("allowed_directions", "long_short")).lower()
        if mode == "paused":
            return []
        if mode == "only_long":
            return ["LONG"]
        if mode == "only_short":
            return ["SHORT"]
        if mode in ("long_short", "both"):
            return ["LONG", "SHORT"]
        logger.error("Unknown allowed_directions=%r, scanning paused until config is fixed.", mode)
        return []

    def _eval_fibo_trend_v2_latest(
        self,
        spec: InstrumentSpec,
        df_15m: pd.DataFrame,
        df_btc: pd.DataFrame,
        directions: List[str],
    ) -> Optional[TradeSignal]:
        """
        Evaluate exact production `fibo_trend_v2` (Variant B) strictly on the latest closed 15M bar:
          - Wave 36, deep Fib 0.618-0.786
          - 4H & 1H EMA20/EMA50 trend alignment + H1 ADX >= 30 & H4 ADX >= 20
          - Cooled RSI (<45 LONG / >55 SHORT) turning in direction + bounce candle (body >= 35%)
          - Anti-pump/dump ATR/ROC filter + 3-bar vol cooldown + BTC shock <= 0.8% + ATR% <= 1.5%
          - Maker limit entry at 0.065% micro-pullback
        """
        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        trend_tf = str(self.config.get("trend_tf", "240"))
        entry_ms = INTERVAL_TO_MS[entry_tf]
        h1_ms = INTERVAL_TO_MS[context_tf]
        h4_ms = INTERVAL_TO_MS[trend_tf]

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

        max_atr_pct = float(self.config.get("max_atr_pct", 1.5))
        if bool(vol_cd.iloc[-1]):
            return None
        cur_atr_pct = float(df["atr_pct"].iloc[-1])
        if not np.isfinite(cur_atr_pct) or cur_atr_pct < 0.12 or cur_atr_pct > max_atr_pct:
            return None

        # Strictly closed 1H and 4H trends + ADX
        df_1h = BybitClient.resample_klines(df, context_tf)
        df_4h = BybitClient.resample_klines(df, trend_tf)
        if len(df_1h) < 55 or len(df_4h) < 55:
            return None

        h1_e20, h1_e50 = ema(df_1h["close"], 20), ema(df_1h["close"], 50)
        h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_out = pd.DataFrame({
            "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
            "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
            "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
            "h1_adx": h1_adx.values,
        }).sort_values("close_ts")

        h4_e20, h4_e50 = ema(df_4h["close"], 20), ema(df_4h["close"], 50)
        h4_adx, _, _ = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], 14)
        h4_out = pd.DataFrame({
            "close_ts": (df_4h["timestamp"] + h4_ms).astype("int64"),
            "h4_bull": (df_4h["close"] > h4_e50) & (h4_e20 > h4_e50),
            "h4_bear": (df_4h["close"] < h4_e50) & (h4_e20 < h4_e50),
            "h4_adx": h4_adx.values,
        }).sort_values("close_ts")

        df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
        df = pd.merge_asof(df, h4_out, on="close_ts", direction="backward")

        h1_adx_min = float(self.config.get("h1_adx_min", 30.0))
        h4_adx_min = float(self.config.get("h4_adx_min", 20.0))
        cur_h1_adx = _finite(df["h1_adx"].iloc[-1])
        cur_h4_adx = _finite(df["h4_adx"].iloc[-1])
        if cur_h1_adx < h1_adx_min or cur_h4_adx < h4_adx_min:
            return None

        wave_n = int(self.config.get("fib_wave_period", 36))
        if len(df) < wave_n + 5:
            return None

        w_hi = float(df["high"].iloc[-(wave_n + 1) : -1].max())
        w_lo = float(df["low"].iloc[-(wave_n + 1) : -1].min())
        w_rng = w_hi - w_lo
        cur_atr = float(df["atr"].iloc[-1])
        if w_rng < 2.2 * cur_atr:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        min_low_2 = min(float(last["low"]), float(prev["low"]))
        max_high_2 = max(float(last["high"]), float(prev["high"]))
        close_p = float(last["close"])
        open_p = float(last["open"])
        cur_body = float(body_ratio.iloc[-1])
        min_body = float(self.config.get("min_body_ratio", 0.35))

        rsi_now = float(last["rsi"])
        rsi_prev = float(prev["rsi"])
        rsi_min3 = float(df["rsi"].iloc[-4:-1].min())
        rsi_max3 = float(df["rsi"].iloc[-4:-1].max())
        rsi_long_max = float(self.config.get("rsi_long_max", 45.0))
        rsi_short_min = float(self.config.get("rsi_short_min", 55.0))

        h1_bull = bool(last["h1_bull"])
        h4_bull = bool(last["h4_bull"])
        h1_bear = bool(last["h1_bear"])
        h4_bear = bool(last["h4_bear"])

        signal_dir: Optional[str] = None
        if (
            "LONG" in directions
            and h1_bull and h4_bull
            and (min_low_2 <= w_hi - 0.60 * w_rng)
            and (min_low_2 >= w_hi - 0.80 * w_rng)
            and (close_p > w_hi - 0.786 * w_rng)
            and (rsi_min3 < rsi_long_max) and (rsi_now > rsi_prev) and (rsi_now < 62.0)
            and (close_p > open_p) and (cur_body >= min_body) and (close_p > float(prev["close"]))
        ):
            signal_dir = "LONG"
        elif (
            "SHORT" in directions
            and h1_bear and h4_bear
            and (max_high_2 >= w_lo + 0.60 * w_rng)
            and (max_high_2 <= w_lo + 0.80 * w_rng)
            and (close_p < w_lo + 0.786 * w_rng)
            and (rsi_max3 > rsi_short_min) and (rsi_now < rsi_prev) and (rsi_now > 38.0)
            and (close_p < open_p) and (cur_body >= min_body) and (close_p < float(prev["close"]))
        ):
            signal_dir = "SHORT"

        if signal_dir is None:
            return None

        pb = float(self.config.get("maker_pullback_pct", 0.065)) / 100.0
        entry_p = close_p * (1.0 - pb) if signal_dir == "LONG" else close_p * (1.0 + pb)
        if spec.tick_size > 0:
            entry_p = round_tick(entry_p, spec.tick_size)

        quality = round(((cur_h1_adx + cur_h4_adx) / 2.0) + cur_body * 25.0, 2)
        sl_pct = float(self.config.get("sl_pct", 3.5))
        return TradeSignal(
            symbol=spec.symbol,
            direction=signal_dir,
            timestamp=int(last["timestamp"]),
            entry_price=entry_p,
            sl_pct=sl_pct,
            score=quality,
            breakdown={
                "strategy": 1.0,
                "h1_adx": round(cur_h1_adx, 2),
                "h4_adx": round(cur_h4_adx, 2),
                "rsi": round(rsi_now, 2),
                "body_ratio": round(cur_body, 3),
                "close_price": close_p,
                "maker_limit_price": entry_p,
            },
            atr_pct=cur_atr_pct,
            rvol=0.0,
            body_ratio=cur_body,
            spec=spec,
        )

    def get_latest_1h_trend(self, symbol: str) -> str:
        """Return strictly closed 1H EMA20/EMA50 trend state: 'BULL', 'BEAR', or 'NEUTRAL'."""
        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        ef = int(self.config.get("h1_ema_fast", 20))
        es = int(self.config.get("h1_ema_slow", 50))
        df_15m = self.client.get_klines(symbol, interval=entry_tf, days=6, use_cache=True, closed_only=True)
        if df_15m.empty or len(df_15m) < 220:
            return "NEUTRAL"
        df_1h = BybitClient.resample_klines(df_15m, context_tf)
        # Drop any incomplete trailing 1H bucket (keep strictly closed 1H candles)
        entry_ms = INTERVAL_TO_MS[entry_tf]
        h1_ms = INTERVAL_TO_MS[context_tf]
        latest_15m_close_ts = int(df_15m["timestamp"].iloc[-1]) + entry_ms
        df_1h = df_1h[df_1h["timestamp"] + h1_ms <= latest_15m_close_ts].reset_index(drop=True)
        if len(df_1h) < es + 5:
            return "NEUTRAL"
        h1_e20 = ema(df_1h["close"], ef)
        h1_e50 = ema(df_1h["close"], es)
        c_last = float(df_1h["close"].iloc[-1])
        e20_last = float(h1_e20.iloc[-1])
        e50_last = float(h1_e50.iloc[-1])
        if c_last > e50_last and e20_last > e50_last:
            return "BULL"
        if c_last < e50_last and e20_last < e50_last:
            return "BEAR"
        return "NEUTRAL"

    def _eval_ada_fibo50_15x_latest(
        self,
        spec: InstrumentSpec,
        df_15m: pd.DataFrame,
        df_btc: pd.DataFrame,
        directions: List[str],
    ) -> Optional[TradeSignal]:
        """
        Evaluate exact locked `ada_fibo50_15x` (Variant B) strictly on the latest closed 15M candle:
          - Wave 20 bars, height >= 2.2 x ATR(14), impulse direction confirmed
          - Spike cooldown: bar_range > 3.2 x ATR on current or previous bar -> skip
          - ATR% in [0.10%, 1.80%]
          - Strictly closed 1H EMA20/EMA50 trend + 1H ADX(14) >= 18.0
          - BTC 1H Veto (>= 1.5%): block LONG if BTC 1H <= -1.5%, block SHORT if BTC 1H >= +1.5%
          - Resting PostOnly Maker limit order at Fib 0.50 retracement (distance >= 0.08% from close)
        """
        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        entry_ms = INTERVAL_TO_MS[entry_tf]
        h1_ms = INTERVAL_TO_MS[context_tf]

        wave_n = int(self.config.get("fib_wave_period", 20))
        min_wave_atr = float(self.config.get("fib_min_wave_atr", 2.2))
        fib_entry = float(self.config.get("fib_entry_level", 0.50))
        min_dist_frac = float(self.config.get("fib_min_dist_pct", 0.08)) / 100.0
        abn_mult = float(self.config.get("abn_spike_atr_mult", 3.2))
        min_atr_pct = float(self.config.get("min_atr_pct", 0.10))
        max_atr_pct = float(self.config.get("max_atr_pct", 1.80))
        ef = int(self.config.get("h1_ema_fast", 20))
        es = int(self.config.get("h1_ema_slow", 50))
        h1_adx_min = float(self.config.get("h1_adx_min", 18.0))
        btc_1h_veto = float(self.config.get("btc_1h_veto_pct", 1.5))
        max_dist_ema200_pct = float(self.config.get("max_dist_ema200_pct", 5.0))

        df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
        if len(df) < max(wave_n + 20, 220):
            return None
        df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
        df["e200"] = ema(df["close"], 200)

        bar_r = df["high"] - df["low"]
        is_abn = (bar_r > abn_mult * df["atr"]).fillna(False)
        vol_cd = is_abn | is_abn.shift(1, fill_value=False)

        if bool(vol_cd.iloc[-1]):
            return None
        cur_atr = float(df["atr"].iloc[-1])
        cur_atr_pct = float(df["atr_pct"].iloc[-1])
        if not np.isfinite(cur_atr) or cur_atr <= 0 or not np.isfinite(cur_atr_pct):
            return None
        if cur_atr_pct < min_atr_pct or cur_atr_pct > max_atr_pct:
            return None

        # Strictly closed 1H EMA20/EMA50 + 1H ADX(14)
        df_1h = BybitClient.resample_klines(df, context_tf)
        if len(df_1h) < es + 10:
            return None
        h1_e20 = ema(df_1h["close"], ef)
        h1_e50 = ema(df_1h["close"], es)
        h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_out = pd.DataFrame({
            "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
            "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
            "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
            "h1_adx": h1_adx.fillna(0.0).values,
        }).sort_values("close_ts")

        df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
        cur_h1_adx = _finite(df["h1_adx"].iloc[-1])
        if cur_h1_adx < h1_adx_min:
            return None

        h1_bull = _flag(df["h1_bull"].iloc[-1])
        h1_bear = _flag(df["h1_bear"].iloc[-1])
        if not h1_bull and not h1_bear:
            return None

        # BTC 1H return (4 closed 15M candles) for BTC 1H Veto (>= 1.5%)
        btc_r1h = 0.0
        if len(df_btc) >= 5:
            b_prev = float(df_btc["close"].iloc[-5])
            b_now = float(df_btc["close"].iloc[-1])
            if b_prev > 0:
                btc_r1h = (b_now - b_prev) / b_prev * 100.0

        # Impulse wave over strictly previous `wave_n` bars (iloc[-(wave_n+1):-1])
        seg_h = df["high"].iloc[-(wave_n + 1) : -1].to_numpy(dtype=np.float64)
        seg_l = df["low"].iloc[-(wave_n + 1) : -1].to_numpy(dtype=np.float64)
        ih = int(np.argmax(seg_h))
        il = int(np.argmin(seg_l))
        whi = float(seg_h[ih])
        wlo = float(seg_l[il])
        rng = whi - wlo
        if not np.isfinite(rng) or rng < min_wave_atr * cur_atr:
            return None

        close_p = float(df["close"].iloc[-1])
        cur_e200 = float(df["e200"].iloc[-1])
        dist_ema200_pct = ((close_p / cur_e200 - 1.0) * 100.0) if (np.isfinite(cur_e200) and cur_e200 > 0) else 0.0
        signal_dir: Optional[str] = None
        limit_p: float = 0.0

        if "LONG" in directions and h1_bull and ih > il:
            if max_dist_ema200_pct > 0 and dist_ema200_pct >= max_dist_ema200_pct:
                self._blocked("trend_exhaustion", spec, "LONG",
                    "Trend exhaustion filter blocked %s LONG (dist_ema200=%+.2f%% >= +%.1f%%).",
                    spec.symbol, dist_ema200_pct, max_dist_ema200_pct,
                )
                return None
            if btc_1h_veto > 0 and btc_r1h <= -btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "LONG","BTC 1H Veto blocked %s LONG (BTC 1H=%.2f%% <= -%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = whi - fib_entry * rng
            if cand_lp < close_p * (1.0 - min_dist_frac):
                signal_dir = "LONG"
                limit_p = cand_lp
        elif "SHORT" in directions and h1_bear and il > ih:
            if max_dist_ema200_pct > 0 and (-dist_ema200_pct) >= max_dist_ema200_pct:
                self._blocked("trend_exhaustion", spec, "SHORT",
                    "Trend exhaustion filter blocked %s SHORT (dist_ema200=%+.2f%% <= -%.1f%%).",
                    spec.symbol, dist_ema200_pct, max_dist_ema200_pct,
                )
                return None
            if btc_1h_veto > 0 and btc_r1h >= btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "SHORT","BTC 1H Veto blocked %s SHORT (BTC 1H=%+.2f%% >= +%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = wlo + fib_entry * rng
            if cand_lp > close_p * (1.0 + min_dist_frac):
                signal_dir = "SHORT"
                limit_p = cand_lp

        if signal_dir is None:
            return None

        if spec.tick_size > 0:
            limit_p = round_tick(limit_p, spec.tick_size)

        sl_pct = float(self.config.get("sl_pct", 2.5))
        return TradeSignal(
            symbol=spec.symbol,
            direction=signal_dir,
            timestamp=int(df["timestamp"].iloc[-1]),
            entry_price=limit_p,
            sl_pct=sl_pct,
            score=round(cur_h1_adx + (rng / cur_atr) * 5.0, 2),
            breakdown={
                "strategy": "ada_fibo50_15x",
                "h1_adx": round(cur_h1_adx, 2),
                "wave_atr_ratio": round(rng / cur_atr, 2),
                "wave_high": whi,
                "wave_low": wlo,
                "fib_level": fib_entry,
                "ema200_15m": round(cur_e200, 6),
                "dist_ema200_pct": round(dist_ema200_pct, 3),
                "btc_r1h_pct": round(btc_r1h, 2),
                "close_price": close_p,
                "maker_limit_price": limit_p,
            },
            atr_pct=cur_atr_pct,
            rvol=0.0,
            body_ratio=0.0,
            spec=spec,
        )

    def _eval_wlfi_meanrev_15x_v2a_latest(
        self,
        spec: InstrumentSpec,
        df_15m: pd.DataFrame,
        df_btc: pd.DataFrame,
        directions: List[str],
    ) -> Optional[TradeSignal]:
        """Evaluate locked WLFIUSDT-V2A 15M Zero-Lookahead Keltner/ATR Mean-Reversion on latest closed 15M bar."""
        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        entry_ms = INTERVAL_TO_MS[entry_tf]
        h1_ms = INTERVAL_TO_MS[context_tf]

        ema_len = int(self.config.get("mr_ema_period", 20))
        k_band = float(self.config.get("mr_k_band", 3.2))
        min_atr_pct = float(self.config.get("min_atr_pct", 0.10))
        max_atr_pct = float(self.config.get("max_atr_pct", 1.40))
        h1_adx_max = float(self.config.get("h1_adx_max", 40.0))
        rsi_period = int(self.config.get("mr_rsi_period", 14))
        rsi_long_max = float(self.config.get("mr_rsi_long_max", 40.0))
        rsi_short_min = float(self.config.get("mr_rsi_short_min", 60.0))
        btc_1h_veto = float(self.config.get("btc_1h_veto_pct", 1.0))
        min_dist_frac = float(self.config.get("fib_min_dist_pct", 0.08)) / 100.0
        tp_min_pct = float(self.config.get("tp_min_pct", 0.60))
        fallback_tp_pct = float(self.config.get("tp_pct", 1.40))
        sl_pct = float(self.config.get("sl_pct", 2.50))
        be1_ratio = float(self.config.get("smart_be1_tp_ratio", 0.65))
        be1_lock = float(self.config.get("smart_be1_lock_ratio", 0.22))
        be2_ratio = float(self.config.get("smart_be2_tp_ratio", 0.82))
        be2_lock = float(self.config.get("smart_be2_lock_ratio", 0.42))

        if len(df_15m) < 120:
            return None

        df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
        df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
        df["ema_mid"] = ema(df["close"], ema_len)
        df["rsi"] = compute_rsi(df["close"], rsi_period)

        cur_atr = float(df["atr"].iloc[-1])
        cur_atr_pct = float(df["atr_pct"].iloc[-1])
        cur_ema = float(df["ema_mid"].iloc[-1])
        cur_rsi = float(df["rsi"].iloc[-1])
        close_p = float(df["close"].iloc[-1])

        if not np.isfinite(cur_atr) or cur_atr <= 0 or not np.isfinite(cur_atr_pct) or not np.isfinite(cur_ema):
            return None
        if cur_atr_pct < min_atr_pct or cur_atr_pct > max_atr_pct:
            return None

        df_1h = BybitClient.resample_klines(df, context_tf)
        if len(df_1h) < 35:
            return None
        h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_out = pd.DataFrame({
            "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
            "h1_adx": h1_adx.fillna(0.0).values,
        }).sort_values("close_ts")
        df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
        cur_h1_adx = _finite(df["h1_adx"].iloc[-1])
        if cur_h1_adx > h1_adx_max:
            return None

        btc_r1h = 0.0
        if len(df_btc) >= 5:
            b_prev = float(df_btc["close"].iloc[-5])
            b_now = float(df_btc["close"].iloc[-1])
            if b_prev > 0:
                btc_r1h = (b_now - b_prev) / b_prev * 100.0

        signal_dir: Optional[str] = None
        limit_p: float = 0.0

        if "LONG" in directions and close_p < cur_ema and cur_rsi <= rsi_long_max:
            if btc_1h_veto > 0 and btc_r1h <= -btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "LONG","BTC 1H Veto blocked %s LONG (BTC 1H=%.2f%% <= -%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = cur_ema - k_band * cur_atr
            if 0.0 < cand_lp < close_p * (1.0 - min_dist_frac):
                signal_dir = "LONG"
                limit_p = cand_lp
        elif "SHORT" in directions and close_p > cur_ema and cur_rsi >= rsi_short_min:
            if btc_1h_veto > 0 and btc_r1h >= btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "SHORT","BTC 1H Veto blocked %s SHORT (BTC 1H=%+.2f%% >= +%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = cur_ema + k_band * cur_atr
            if cand_lp > close_p * (1.0 + min_dist_frac):
                signal_dir = "SHORT"
                limit_p = cand_lp

        if signal_dir is None:
            return None

        if spec.tick_size > 0:
            limit_p = round_tick(limit_p, spec.tick_size)

        is_l = signal_dir == "LONG"
        tp_min_f = tp_min_pct / 100.0
        if (cur_ema >= limit_p * (1.0 + tp_min_f)) if is_l else (cur_ema <= limit_p * (1.0 - tp_min_f)):
            tp_p = round_tick(cur_ema, spec.tick_size) if spec.tick_size > 0 else cur_ema
            eff_tp_pct = abs(tp_p - limit_p) / limit_p * 100.0
        else:
            eff_tp_pct = fallback_tp_pct
            raw_tp = limit_p * (1.0 + eff_tp_pct / 100.0) if is_l else limit_p * (1.0 - eff_tp_pct / 100.0)
            tp_p = round_tick(raw_tp, spec.tick_size) if spec.tick_size > 0 else raw_tp

        return TradeSignal(
            symbol=spec.symbol,
            direction=signal_dir,
            timestamp=int(df["timestamp"].iloc[-1]),
            entry_price=limit_p,
            sl_pct=sl_pct,
            score=round(abs(50.0 - cur_rsi) * 2.0 + (40.0 - cur_h1_adx), 2),
            breakdown={
                "strategy": "wlfi_meanrev_15x_v2a",
                "ema20": round(cur_ema, 6),
                "k_band": k_band,
                "atr": round(cur_atr, 6),
                "atr_pct": round(cur_atr_pct, 4),
                "rsi14": round(cur_rsi, 2),
                "h1_adx": round(cur_h1_adx, 2),
                "btc_r1h_pct": round(btc_r1h, 2),
                "close_price": close_p,
                "maker_limit_price": limit_p,
                "entry_ema_tp_price": tp_p,
                "eff_tp_pct": round(eff_tp_pct, 4),
            },
            atr_pct=cur_atr_pct,
            rvol=0.0,
            body_ratio=0.0,
            spec=spec,
            tp_price=tp_p,
            tp_pct=eff_tp_pct,
            be1_trigger_pct=eff_tp_pct * be1_ratio,
            be1_offset_pct=eff_tp_pct * be1_lock,
            be2_trigger_pct=eff_tp_pct * be2_ratio,
            be2_offset_pct=eff_tp_pct * be2_lock,
        )

    def _eval_ltc_trend_bb_10x_latest(
        self,
        spec: InstrumentSpec,
        df_15m: pd.DataFrame,
        df_btc: pd.DataFrame,
        directions: List[str],
    ) -> Optional[TradeSignal]:
        """Evaluate System 3 (LTCUSDT 15M Trend + Bollinger Band Standard-Deviation Dip) on latest closed 15M bar."""
        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        entry_ms = INTERVAL_TO_MS[entry_tf]
        h1_ms = INTERVAL_TO_MS[context_tf]

        bb_len = int(self.config.get("bb_period", 20))
        bb_k_sd = float(self.config.get("bb_k_sd", 1.4))
        abn_mult = float(self.config.get("abn_spike_atr_mult", 3.2))
        min_atr_pct = float(self.config.get("min_atr_pct", 0.10))
        max_atr_pct = float(self.config.get("max_atr_pct", 1.90))
        ef = int(self.config.get("h1_ema_fast", 20))
        es = int(self.config.get("h1_ema_slow", 50))
        h1_adx_min = float(self.config.get("h1_adx_min", 14.0))
        btc_1h_veto = float(self.config.get("btc_1h_veto_pct", 0.7))
        min_dist_frac = float(self.config.get("fib_min_dist_pct", 0.08)) / 100.0
        tp_pct = float(self.config.get("tp_pct", 1.35))
        sl_pct = float(self.config.get("sl_pct", 2.50))
        be1_tr = float(self.config.get("smart_be1_trigger_pct", round(tp_pct * 0.65, 4)))
        be1_off = float(self.config.get("smart_be1_offset_pct", round(tp_pct * 0.22, 4)))
        be2_tr = float(self.config.get("smart_be2_trigger_pct", round(tp_pct * 0.82, 4)))
        be2_off = float(self.config.get("smart_be2_offset_pct", round(tp_pct * 0.42, 4)))

        if len(df_15m) < 120:
            return None

        df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
        df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
        df["ema_mid"] = ema(df["close"], bb_len)
        df["sd"] = df["close"].rolling(bb_len, min_periods=bb_len).std().fillna(0.0)

        bar_r = df["high"] - df["low"]
        is_abn = (bar_r > abn_mult * df["atr"]).fillna(False)
        vol_cd = is_abn | is_abn.shift(1, fill_value=False)
        if bool(vol_cd.iloc[-1]):
            return None

        cur_atr = float(df["atr"].iloc[-1])
        cur_atr_pct = float(df["atr_pct"].iloc[-1])
        cur_ema = float(df["ema_mid"].iloc[-1])
        cur_sd = float(df["sd"].iloc[-1])
        close_p = float(df["close"].iloc[-1])

        if not np.isfinite(cur_atr) or cur_atr <= 0 or not np.isfinite(cur_atr_pct) or cur_sd <= 0:
            return None
        if cur_atr_pct < min_atr_pct or cur_atr_pct > max_atr_pct:
            return None

        df_1h = BybitClient.resample_klines(df, context_tf)
        if len(df_1h) < es + 10:
            return None
        h1_e20 = ema(df_1h["close"], ef)
        h1_e50 = ema(df_1h["close"], es)
        h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_out = pd.DataFrame({
            "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
            "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
            "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
            "h1_adx": h1_adx.fillna(0.0).values,
        }).sort_values("close_ts")

        df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
        cur_h1_adx = _finite(df["h1_adx"].iloc[-1])
        if cur_h1_adx < h1_adx_min:
            return None

        h1_bull = _flag(df["h1_bull"].iloc[-1])
        h1_bear = _flag(df["h1_bear"].iloc[-1])
        if not h1_bull and not h1_bear:
            return None

        btc_r1h = 0.0
        if len(df_btc) >= 5:
            b_prev = float(df_btc["close"].iloc[-5])
            b_now = float(df_btc["close"].iloc[-1])
            if b_prev > 0:
                btc_r1h = (b_now - b_prev) / b_prev * 100.0

        signal_dir: Optional[str] = None
        limit_p: float = 0.0

        if "LONG" in directions and h1_bull and close_p < cur_ema:
            if btc_1h_veto > 0 and btc_r1h <= -btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "LONG","BTC 1H Veto blocked %s LONG (BTC 1H=%.2f%% <= -%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = cur_ema - bb_k_sd * cur_sd
            if 0.0 < cand_lp < close_p * (1.0 - min_dist_frac):
                signal_dir = "LONG"
                limit_p = cand_lp
        elif "SHORT" in directions and h1_bear and close_p > cur_ema:
            if btc_1h_veto > 0 and btc_r1h >= btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "SHORT","BTC 1H Veto blocked %s SHORT (BTC 1H=%+.2f%% >= +%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = cur_ema + bb_k_sd * cur_sd
            if cand_lp > close_p * (1.0 + min_dist_frac):
                signal_dir = "SHORT"
                limit_p = cand_lp

        if signal_dir is None:
            return None

        if spec.tick_size > 0:
            limit_p = round_tick(limit_p, spec.tick_size)

        return TradeSignal(
            symbol=spec.symbol,
            direction=signal_dir,
            timestamp=int(df["timestamp"].iloc[-1]),
            entry_price=limit_p,
            sl_pct=sl_pct,
            score=round(cur_h1_adx + (abs(close_p - cur_ema) / max(cur_sd, 1e-9)) * 10.0, 2),
            breakdown={
                "strategy": "ltc_trend_bb_10x",
                "ema20": round(cur_ema, 4),
                "sd20": round(cur_sd, 4),
                "bb_k_sd": bb_k_sd,
                "h1_adx": round(cur_h1_adx, 2),
                "btc_r1h_pct": round(btc_r1h, 2),
                "close_price": close_p,
                "maker_limit_price": limit_p,
            },
            atr_pct=cur_atr_pct,
            rvol=0.0,
            body_ratio=0.0,
            spec=spec,
            tp_pct=tp_pct,
            be1_trigger_pct=be1_tr,
            be1_offset_pct=be1_off,
            be2_trigger_pct=be2_tr,
            be2_offset_pct=be2_off,
        )

    def _eval_mnt_breakout_don_10x_latest(
        self,
        spec: InstrumentSpec,
        df_15m: pd.DataFrame,
        df_btc: pd.DataFrame,
        directions: List[str],
    ) -> Optional[TradeSignal]:
        """Evaluate System 4 (MNTUSDT 15M Donchian-10 Volatility Breakout + 0.30*ATR Micro-Pullback) on latest closed 15M bar."""
        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        entry_ms = INTERVAL_TO_MS[entry_tf]
        h1_ms = INTERVAL_TO_MS[context_tf]

        don_len = int(self.config.get("don_period", 10))
        pb_atr_k = float(self.config.get("don_pullback_atr", 0.30))
        abn_mult = float(self.config.get("abn_spike_atr_mult", 3.2))
        min_atr_pct = float(self.config.get("min_atr_pct", 0.10))
        max_atr_pct = float(self.config.get("max_atr_pct", 1.90))
        ef = int(self.config.get("h1_ema_fast", 20))
        es = int(self.config.get("h1_ema_slow", 50))
        h1_adx_min = float(self.config.get("h1_adx_min", 18.0))
        h1_slope_min = float(self.config.get("h1_ema50_slope_min_pct", 0.0))
        btc_1h_veto = float(self.config.get("btc_1h_veto_pct", 0.5))
        min_dist_frac = float(self.config.get("fib_min_dist_pct", 0.08)) / 100.0
        tp_pct = float(self.config.get("tp_pct", 1.40))
        sl_pct = float(self.config.get("sl_pct", 2.60))
        be1_tr = float(self.config.get("smart_be1_trigger_pct", round(tp_pct * 0.50, 4)))
        be1_off = float(self.config.get("smart_be1_offset_pct", round(tp_pct * 0.15, 4)))
        be2_tr = float(self.config.get("smart_be2_trigger_pct", round(tp_pct * 0.75, 4)))
        be2_off = float(self.config.get("smart_be2_offset_pct", round(tp_pct * 0.35, 4)))

        if len(df_15m) < 120:
            return None

        df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
        df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
        df["don_hi"] = df["high"].shift(1).rolling(don_len, min_periods=don_len).max()
        df["don_lo"] = df["low"].shift(1).rolling(don_len, min_periods=don_len).min()

        bar_r = df["high"] - df["low"]
        is_abn = (bar_r > abn_mult * df["atr"]).fillna(False)
        vol_cd = is_abn | is_abn.shift(1, fill_value=False)
        if bool(vol_cd.iloc[-1]):
            return None

        cur_atr = float(df["atr"].iloc[-1])
        cur_atr_pct = float(df["atr_pct"].iloc[-1])
        cur_don_hi = float(df["don_hi"].iloc[-1])
        cur_don_lo = float(df["don_lo"].iloc[-1])
        close_p = float(df["close"].iloc[-1])

        if not np.isfinite(cur_atr) or cur_atr <= 0 or not np.isfinite(cur_atr_pct) or not np.isfinite(cur_don_hi):
            return None
        if cur_atr_pct < min_atr_pct or cur_atr_pct > max_atr_pct:
            return None

        df_1h = BybitClient.resample_klines(df, context_tf)
        if len(df_1h) < es + 10:
            return None
        h1_e20 = ema(df_1h["close"], ef)
        h1_e50 = ema(df_1h["close"], es)
        h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_out = pd.DataFrame({
            "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
            "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
            "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
            "h1_adx": h1_adx.fillna(0.0).values,
            "h1_slope": (h1_e50.pct_change(3) * 100.0).fillna(0.0).values,
        }).sort_values("close_ts")

        df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
        cur_h1_adx = _finite(df["h1_adx"].iloc[-1])
        cur_h1_slope = _finite(df["h1_slope"].iloc[-1])
        if cur_h1_adx < h1_adx_min:
            return None

        h1_bull = _flag(df["h1_bull"].iloc[-1])
        h1_bear = _flag(df["h1_bear"].iloc[-1])
        if not h1_bull and not h1_bear:
            return None

        btc_r1h = 0.0
        if len(df_btc) >= 5:
            b_prev = float(df_btc["close"].iloc[-5])
            b_now = float(df_btc["close"].iloc[-1])
            if b_prev > 0:
                btc_r1h = (b_now - b_prev) / b_prev * 100.0

        signal_dir: Optional[str] = None
        limit_p: float = 0.0

        if "LONG" in directions and h1_bull and close_p > cur_don_hi:
            if cur_h1_slope <= h1_slope_min:
                self._blocked("h1_slope", spec, "LONG",
                    "1H EMA50 slope filter blocked %s LONG (slope=%+.4f%% <= %+.4f%%).",
                    spec.symbol, cur_h1_slope, h1_slope_min,
                )
                return None
            if btc_1h_veto > 0 and btc_r1h <= -btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "LONG","BTC 1H Veto blocked %s LONG (BTC 1H=%.2f%% <= -%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = close_p - pb_atr_k * cur_atr
            if 0.0 < cand_lp < close_p * (1.0 - min_dist_frac):
                signal_dir = "LONG"
                limit_p = cand_lp
        elif "SHORT" in directions and h1_bear and close_p < cur_don_lo:
            if cur_h1_slope >= -h1_slope_min:
                self._blocked("h1_slope", spec, "SHORT",
                    "1H EMA50 slope filter blocked %s SHORT (slope=%+.4f%% >= %+.4f%%).",
                    spec.symbol, cur_h1_slope, -h1_slope_min,
                )
                return None
            if btc_1h_veto > 0 and btc_r1h >= btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "SHORT","BTC 1H Veto blocked %s SHORT (BTC 1H=%+.2f%% >= +%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = close_p + pb_atr_k * cur_atr
            if cand_lp > close_p * (1.0 + min_dist_frac):
                signal_dir = "SHORT"
                limit_p = cand_lp

        if signal_dir is None:
            return None

        if spec.tick_size > 0:
            limit_p = round_tick(limit_p, spec.tick_size)

        return TradeSignal(
            symbol=spec.symbol,
            direction=signal_dir,
            timestamp=int(df["timestamp"].iloc[-1]),
            entry_price=limit_p,
            sl_pct=sl_pct,
            score=round(cur_h1_adx + 25.0, 2),
            breakdown={
                "strategy": "mnt_breakout_don_10x",
                "don_hi": round(cur_don_hi, 5),
                "don_lo": round(cur_don_lo, 5),
                "pb_atr_k": pb_atr_k,
                "h1_adx": round(cur_h1_adx, 2),
                "h1_ema50_slope_pct": round(cur_h1_slope, 4),
                "btc_r1h_pct": round(btc_r1h, 2),
                "close_price": close_p,
                "maker_limit_price": limit_p,
            },
            atr_pct=cur_atr_pct,
            rvol=0.0,
            body_ratio=0.0,
            spec=spec,
            tp_pct=tp_pct,
            be1_trigger_pct=be1_tr,
            be1_offset_pct=be1_off,
            be2_trigger_pct=be2_tr,
            be2_offset_pct=be2_off,
        )

    def _eval_crv_stoch_10x_latest(
        self,
        spec: InstrumentSpec,
        df_15m: pd.DataFrame,
        df_btc: pd.DataFrame,
        directions: List[str],
    ) -> Optional[TradeSignal]:
        """Evaluate System 5 (CRVUSDT 1H Trend + 15M Stochastic(14) Pullback to EMA20 - 0.70*ATR) on latest closed 15M bar."""
        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        entry_ms = INTERVAL_TO_MS[entry_tf]
        h1_ms = INTERVAL_TO_MS[context_tf]

        ema_period = int(self.config.get("crv_ema_period", 20))
        stoch_period = int(self.config.get("crv_stoch_period", 14))
        stoch_long_max = float(self.config.get("crv_stoch_long_max", 20.0))
        stoch_short_min = float(self.config.get("crv_stoch_short_min", 80.0))
        pb_atr_k = float(self.config.get("crv_pullback_atr", 0.70))
        abn_mult = float(self.config.get("abn_spike_atr_mult", 3.2))
        min_atr_pct = float(self.config.get("min_atr_pct", 0.10))
        max_atr_pct = float(self.config.get("max_atr_pct", 1.80))
        ef = int(self.config.get("h1_ema_fast", 20))
        es = int(self.config.get("h1_ema_slow", 50))
        h1_adx_min = float(self.config.get("h1_adx_min", 14.0))
        max_dist_ema200_pct = float(self.config.get("max_dist_ema200_pct", 5.0))
        btc_1h_veto = float(self.config.get("btc_1h_veto_pct", 1.0))
        min_dist_frac = float(self.config.get("fib_min_dist_pct", 0.08)) / 100.0
        tp_pct = float(self.config.get("tp_pct", 1.50))
        sl_pct = float(self.config.get("sl_pct", 2.50))
        be1_tr = float(self.config.get("smart_be1_trigger_pct", round(tp_pct * 0.55, 4)))
        be1_off = float(self.config.get("smart_be1_offset_pct", round(tp_pct * 0.18, 4)))
        be2_tr = float(self.config.get("smart_be2_trigger_pct", round(tp_pct * 0.78, 4)))
        be2_off = float(self.config.get("smart_be2_offset_pct", round(tp_pct * 0.38, 4)))

        if len(df_15m) < 220:
            return None

        df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
        df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
        df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
        df["e20"] = ema(df["close"], ema_period)
        df["e200"] = ema(df["close"], 200)
        lo_st = df["low"].rolling(stoch_period, min_periods=stoch_period).min()
        hi_st = df["high"].rolling(stoch_period, min_periods=stoch_period).max()
        df["stoch_k"] = ((df["close"] - lo_st) / (hi_st - lo_st).replace(0.0, np.nan) * 100.0).fillna(50.0)

        bar_r = df["high"] - df["low"]
        is_abn = (bar_r > abn_mult * df["atr"]).fillna(False)
        vol_cd = is_abn | is_abn.shift(1, fill_value=False)
        if bool(vol_cd.iloc[-1]):
            return None

        cur_atr = float(df["atr"].iloc[-1])
        cur_atr_pct = float(df["atr_pct"].iloc[-1])
        cur_e20 = float(df["e20"].iloc[-1])
        cur_e200 = float(df["e200"].iloc[-1])
        cur_stoch_k = float(df["stoch_k"].iloc[-1])
        close_p = float(df["close"].iloc[-1])

        if not np.isfinite(cur_atr) or cur_atr <= 0 or not np.isfinite(cur_atr_pct) or not np.isfinite(cur_e20):
            return None
        if cur_atr_pct < min_atr_pct or cur_atr_pct > max_atr_pct:
            return None

        df_1h = BybitClient.resample_klines(df, context_tf)
        if len(df_1h) < es + 10:
            return None
        h1_e20 = ema(df_1h["close"], ef)
        h1_e50 = ema(df_1h["close"], es)
        h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
        h1_out = pd.DataFrame({
            "close_ts": (df_1h["timestamp"] + h1_ms).astype("int64"),
            "h1_bull": (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50),
            "h1_bear": (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50),
            "h1_adx": h1_adx.fillna(0.0).values,
        }).sort_values("close_ts")

        df = pd.merge_asof(df, h1_out, on="close_ts", direction="backward")
        cur_h1_adx = _finite(df["h1_adx"].iloc[-1])
        if cur_h1_adx < h1_adx_min:
            return None

        h1_bull = _flag(df["h1_bull"].iloc[-1])
        h1_bear = _flag(df["h1_bear"].iloc[-1])
        if not h1_bull and not h1_bear:
            return None

        btc_r1h = 0.0
        if len(df_btc) >= 5:
            b_prev = float(df_btc["close"].iloc[-5])
            b_now = float(df_btc["close"].iloc[-1])
            if b_prev > 0:
                btc_r1h = (b_now - b_prev) / b_prev * 100.0

        dist_ema200_pct = ((close_p / cur_e200 - 1.0) * 100.0) if (np.isfinite(cur_e200) and cur_e200 > 0) else 0.0
        signal_dir: Optional[str] = None
        limit_p: float = 0.0

        if "LONG" in directions and h1_bull and cur_stoch_k <= stoch_long_max:
            if max_dist_ema200_pct > 0 and dist_ema200_pct >= max_dist_ema200_pct:
                self._blocked("trend_exhaustion", spec, "LONG",
                    "Trend exhaustion filter blocked %s LONG (dist_ema200=%+.2f%% >= +%.1f%%).",
                    spec.symbol, dist_ema200_pct, max_dist_ema200_pct,
                )
                return None
            if btc_1h_veto > 0 and btc_r1h <= -btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "LONG","BTC 1H Veto blocked %s LONG (BTC 1H=%.2f%% <= -%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = cur_e20 - pb_atr_k * cur_atr
            if 0.0 < cand_lp < close_p * (1.0 - min_dist_frac):
                signal_dir = "LONG"
                limit_p = cand_lp
        elif "SHORT" in directions and h1_bear and cur_stoch_k >= stoch_short_min:
            if max_dist_ema200_pct > 0 and (-dist_ema200_pct) >= max_dist_ema200_pct:
                self._blocked("trend_exhaustion", spec, "SHORT",
                    "Trend exhaustion filter blocked %s SHORT (dist_ema200=%+.2f%% <= -%.1f%%).",
                    spec.symbol, dist_ema200_pct, max_dist_ema200_pct,
                )
                return None
            if btc_1h_veto > 0 and btc_r1h >= btc_1h_veto:
                self._blocked("btc_1h_veto", spec, "SHORT","BTC 1H Veto blocked %s SHORT (BTC 1H=%+.2f%% >= +%.1f%%).", spec.symbol, btc_r1h, btc_1h_veto)
                return None
            cand_lp = cur_e20 + pb_atr_k * cur_atr
            if cand_lp > close_p * (1.0 + min_dist_frac):
                signal_dir = "SHORT"
                limit_p = cand_lp

        if signal_dir is None:
            return None

        if spec.tick_size > 0:
            limit_p = round_tick(limit_p, spec.tick_size)

        return TradeSignal(
            symbol=spec.symbol,
            direction=signal_dir,
            timestamp=int(df["timestamp"].iloc[-1]),
            entry_price=limit_p,
            sl_pct=sl_pct,
            score=round(cur_h1_adx + 25.0, 2),
            breakdown={
                "strategy": "crv_stoch_10x",
                "stoch_k14": round(cur_stoch_k, 2),
                "ema20": round(cur_e20, 5),
                "pb_atr_k": pb_atr_k,
                "h1_adx": round(cur_h1_adx, 2),
                "dist_ema200_pct": round(dist_ema200_pct, 2),
                "btc_r1h_pct": round(btc_r1h, 2),
                "close_price": close_p,
                "maker_limit_price": limit_p,
            },
            atr_pct=cur_atr_pct,
            rvol=0.0,
            body_ratio=0.0,
            spec=spec,
            tp_pct=tp_pct,
            be1_trigger_pct=be1_tr,
            be1_offset_pct=be1_off,
            be2_trigger_pct=be2_tr,
            be2_offset_pct=be2_off,
        )

    def scan_once(self, equity: float, top_n: Optional[int] = None) -> List[TradeSignal]:
        """Scan eligible symbols on the latest closed 15M bar and return signals sorted by quality/score desc."""
        self.last_scan_incomplete = False
        self.journal_events = []
        entry_ms_cfg = INTERVAL_TO_MS.get(str(self.config.get("entry_tf", "15")), 15 * 60_000)
        self._journal_bar_ts = (int(time.time() * 1000) // entry_ms_cfg - 1) * entry_ms_cfg
        directions = self.allowed_direction_list()
        if not directions:
            logger.info("Scanner skipped: allowed_directions is 'paused'.")
            self._note("SCAN_BLOCKED", "directions_paused")
            return []

        # HARD US Macro Shield check (blocks new entries during -30m..+15m UTC OR fail-safe stale calendar)
        if self.news_guard is not None:
            news_state = self.news_guard.evaluate_macro_shield(allow_network_refresh=True)
        else:
            news_state = read_news_state(self.news_state_path)
        if not news_state.get("entry_allowed", news_state.get("trading_allowed", False)):
            logger.warning(
                "Scanner HARD-BLOCKED by US Macro Shield | fail_safe=%s | pause_until=%s | Reason: %s",
                news_state.get("fail_safe_active"),
                news_state.get("pause_until_utc"),
                news_state.get("reason"),
            )
            reason = "macro_fail_safe" if news_state.get("fail_safe_active") else "macro_shield"
            self._note("SCAN_BLOCKED", reason, details={"reason": str(news_state.get("reason", ""))})
            return []

        blacklist, macro_pause = load_blacklist(self.blacklist_path)
        if macro_pause:
            logger.info("Scanner paused due to blacklist macro flag: %s", macro_pause)
            self._note("SCAN_BLOCKED", "blacklist_pause", details={"reason": str(macro_pause)})
            return []

        entry_tf = str(self.config.get("entry_tf", "15"))
        context_tf = str(self.config.get("context_tf", "60"))
        trend_tf = str(self.config.get("trend_tf", "240"))
        strategy_name = str(self.config.get("strategy", "")).strip().lower()
        single_coin_strats = (
            "ada_fibo50_15x",
            "wlfi_meanrev_15x_v2a",
            "ltc_trend_bb_10x",
            "mnt_breakout_don_10x",
            "crv_stoch_10x",
        )
        if strategy_name not in single_coin_strats + KNOWN_MULTI_COIN_STRATEGIES:
            logger.error("Unknown strategy %r in config, scanning skipped.", strategy_name)
            return []

        entry_ms = INTERVAL_TO_MS[entry_tf]
        expected_last_ts = (int(time.time() * 1000) // entry_ms - 1) * entry_ms

        # Fetch BTC reference data first
        df_btc = self.client.get_klines("BTCUSDT", interval=entry_tf, days=4, use_cache=True, closed_only=True)
        if df_btc.empty or len(df_btc) < 30:
            logger.warning("Insufficient BTCUSDT candles for scan.")
            self.last_scan_incomplete = True
            return []
        if int(df_btc["timestamp"].iloc[-1]) != expected_last_ts:
            logger.warning(
                "BTCUSDT last closed bar %s is not the expected %s, skipping scan.",
                int(df_btc["timestamp"].iloc[-1]), expected_last_ts,
            )
            self.last_scan_incomplete = True
            return []
        if strategy_name not in single_coin_strats:
            last_btc = df_btc.iloc[-1]
            btc_move = abs(float(last_btc["close"]) - float(last_btc["open"])) / float(last_btc["open"]) * 100.0
            btc_shock = float(self.config.get("btc_shock_pause_pct", 0.8))
            if btc_move > btc_shock:
                logger.warning("BTC 15M shock filter triggered (%.2f%% > %.2f%%). Skipping scan.", btc_move, btc_shock)
                self._note("SCAN_BLOCKED", "btc_shock", details={"btc_move_pct": round(btc_move, 3)})
                return []

        universe = self.get_universe(top_n=top_n)
        signals: List[TradeSignal] = []

        for spec in universe:
            if spec.symbol == "BTCUSDT":
                continue
            if spec.symbol in blacklist:
                self._note("FILTERED", "blacklisted", spec.symbol)
                continue

            try:
                days_needed = 7 if strategy_name in single_coin_strats else 16
                df_15m = self.client.get_klines(spec.symbol, interval=entry_tf, days=days_needed, use_cache=True, closed_only=True)
                if len(df_15m) < 120:
                    continue
                if int(df_15m["timestamp"].iloc[-1]) != expected_last_ts:
                    logger.warning(
                        "%s last closed bar %s is not the expected %s, skipping.",
                        spec.symbol, int(df_15m["timestamp"].iloc[-1]), expected_last_ts,
                    )
                    self.last_scan_incomplete = True
                    continue

                if strategy_name == "ada_fibo50_15x":
                    sig = self._eval_ada_fibo50_15x_latest(spec, df_15m, df_btc, directions)
                    if sig is not None:
                        signals.append(sig)
                    continue

                if strategy_name == "wlfi_meanrev_15x_v2a":
                    sig = self._eval_wlfi_meanrev_15x_v2a_latest(spec, df_15m, df_btc, directions)
                    if sig is not None:
                        signals.append(sig)
                    continue

                if strategy_name == "ltc_trend_bb_10x":
                    sig = self._eval_ltc_trend_bb_10x_latest(spec, df_15m, df_btc, directions)
                    if sig is not None:
                        signals.append(sig)
                    continue

                if strategy_name == "mnt_breakout_don_10x":
                    sig = self._eval_mnt_breakout_don_10x_latest(spec, df_15m, df_btc, directions)
                    if sig is not None:
                        signals.append(sig)
                    continue

                if strategy_name == "crv_stoch_10x":
                    sig = self._eval_crv_stoch_10x_latest(spec, df_15m, df_btc, directions)
                    if sig is not None:
                        signals.append(sig)
                    continue

                if strategy_name == "fibo_trend_v2":
                    sig = self._eval_fibo_trend_v2_latest(spec, df_15m, df_btc, directions)
                    if sig is not None:
                        signals.append(sig)
                    continue

                df_1h = BybitClient.resample_klines(df_15m, context_tf)
                df_4h = BybitClient.resample_klines(df_15m, trend_tf)
                if len(df_1h) < 55 or len(df_4h) < 25:
                    continue

                enriched = enrich_entry_dataframe(
                    df_15m=df_15m,
                    df_btc_15m=df_btc,
                    df_1h=df_1h,
                    df_4h=df_4h,
                    entry_interval_ms=INTERVAL_TO_MS[entry_tf],
                    context_interval_ms=INTERVAL_TO_MS[context_tf],
                    trend_interval_ms=INTERVAL_TO_MS[trend_tf],
                )

                oi_drop_pct: Optional[float] = None
                oi_df = self.client.get_open_interest(spec.symbol, interval_time="15min", limit=6)
                if oi_df is not None and len(oi_df) >= 3:
                    prev_oi = float(oi_df["openInterest"].iloc[-3])
                    curr_oi = float(oi_df["openInterest"].iloc[-1])
                    if prev_oi > 0:
                        oi_drop_pct = ((prev_oi - curr_oi) / prev_oi) * 100.0

                highs = enriched["high"].to_numpy(dtype=np.float64)
                lows = enriched["low"].to_numpy(dtype=np.float64)
                volumes = enriched["volume"].to_numpy(dtype=np.float64)
                idx = len(enriched) - 1
                last_row = enriched.iloc[idx]

                best_for_symbol: Optional[TradeSignal] = None
                for direction in directions:
                    scored = score_closed_bar(
                        df=enriched,
                        idx=idx,
                        direction=direction,
                        config=self.config,
                        highs=highs,
                        lows=lows,
                        volumes=volumes,
                        oi_drop_pct=oi_drop_pct,
                        funding_rate=spec.funding_rate,
                    )
                    if scored is None:
                        continue
                    total_score, sl_pct, breakdown = scored

                    plan, _ = self.risk_manager.build_position_plan(
                        symbol=spec.symbol,
                        direction=direction,
                        entry_price=float(last_row["close"]),
                        sl_pct=sl_pct,
                        equity=equity,
                        spec=spec,
                    )
                    if plan is None:
                        continue

                    cand = TradeSignal(
                        symbol=spec.symbol,
                        direction=direction,
                        timestamp=int(last_row["timestamp"]),
                        entry_price=float(last_row["close"]),
                        sl_pct=sl_pct,
                        score=total_score,
                        breakdown=breakdown,
                        atr_pct=float(last_row["atr_pct"]),
                        rvol=float(last_row["rvol"]),
                        body_ratio=float(last_row["body_ratio"]),
                        spec=spec,
                    )
                    if best_for_symbol is None or cand.score > best_for_symbol.score:
                        best_for_symbol = cand

                if best_for_symbol is not None:
                    signals.append(best_for_symbol)

            except Exception as exc:
                logger.warning("Scan error on %s: %s", spec.symbol, exc)

        signals.sort(key=lambda s: s.score, reverse=True)
        return signals
