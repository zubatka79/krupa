"""
Strictly Causal Market Regime & Microstructure Helpers for L2_news:
  1) BTC 4H V1 Trend Switch + Strictly Closed Previous-UTC-Day 1D EMA20 & 1D ATR(14) VolStorm (>4%).
     ZERO look-ahead: 1D indicators NEVER include the currently forming UTC day (day_ts + DAY_MS <= now_ms).
  2) Live Retail Long/Short Account Ratio Z-Score via Bybit V5 `/v5/market/account-ratio?period=1h`
     (72h rolling z-score on strictly closed 1h intervals; blocks LONG if z > +1.0, SHORT if z < -1.0).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from ai_trader_v2.vika_engine import DAY_MS, H1_MS, H4_MS, atr_arr, ema_arr
from bot.bybit_client import BybitClient

logger = logging.getLogger("vika.l2_market_data")

_ls_thread_local = threading.local()
_ls_cache_lock = threading.Lock()
# Cache: symbol -> (fetched_mono_ts, last_closed_hour_ts, z_score, buy_ratio)
_ls_cache: Dict[str, Tuple[float, int, float, float]] = {}


def _get_ls_session() -> requests.Session:
    sess = getattr(_ls_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        sess.headers.update({"User-Agent": "bybit-l2-news-engine/1.0"})
        _ls_thread_local.session = sess
    return sess


@dataclass(frozen=True)
class BtcL2Regime:
    bar_ts_4h: int
    close_4h: float
    ema20_4h: float
    ema50_4h: float
    v1_long: bool
    v1_short: bool
    day_ts_1d: int
    close_1d: float
    ema20_1d: float
    d1_dir: int            # +1 (bullish) or -1 (bearish)
    atr1d_pct: float
    in_storm: bool         # True when strictly closed 1D ATR(14)% > storm_atr_pct (4.0%)

    @property
    def allow_long(self) -> bool:
        return self.v1_long and (self.d1_dir == 1)

    @property
    def allow_short(self) -> bool:
        return self.v1_short and (self.d1_dir == -1)

    @property
    def dist_ema50_4h_pct(self) -> float:
        return round((self.close_4h - self.ema50_4h) / max(self.ema50_4h, 1e-12) * 100.0, 2)

    @property
    def mode_label(self) -> str:
        storm_tag = f" ⚡ ШТОРМ (1D ATR {self.atr1d_pct:.2f}% &gt; 4%)" if self.in_storm else f" (1D ATR {self.atr1d_pct:.2f}%)"
        if self.allow_long:
            return f"🟢 LONG-РЕЖИМ (BTC 4H UP + 1D &gt; EMA20){storm_tag}"
        if self.allow_short:
            return f"🔴 SHORT-РЕЖИМ (BTC 4H DN + 1D &lt; EMA20){storm_tag}"
        return f"⚪ НЕЙТРАЛ / СПИМ (4H={('UP' if self.v1_long else ('DN' if self.v1_short else 'FLAT'))}, 1D={('UP' if self.d1_dir == 1 else 'DN')}){storm_tag}"


def evaluate_btc_l2_regime(
    df_btc_4h_raw: pd.DataFrame,
    now_ms: int,
    *,
    storm_atr_pct: float = 4.0,
) -> BtcL2Regime:
    """
    Evaluate strictly closed BTC 4H V1 + strictly closed previous UTC day 1D EMA20 & 1D ATR(14).
    Drops any forming 4H bar AND drops the currently forming UTC day (day_ts + DAY_MS > now_ms).
    """
    if df_btc_4h_raw.empty:
        raise ValueError("Empty BTC 4H dataframe")

    df4 = (
        df_btc_4h_raw.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df4["timestamp"] = df4["timestamp"].astype(np.int64)
    closed_4h = df4[df4["timestamp"] + H4_MS <= now_ms].reset_index(drop=True)
    if len(closed_4h) < 55:
        raise ValueError(f"Insufficient closed BTC 4H bars ({len(closed_4h)} < 55)")

    c4 = closed_4h["close"].to_numpy(float)
    e20_4 = ema_arr(c4, 20)
    e50_4 = ema_arr(c4, 50)
    i4 = len(c4) - 1
    v1_l = bool((c4[i4] > e50_4[i4]) and (e20_4[i4] > e50_4[i4]))
    v1_s = bool((c4[i4] < e50_4[i4]) and (e20_4[i4] < e50_4[i4]))

    # Aggregate closed 4H bars into UTC days and keep STRICTLY CLOSED previous UTC days (day_ts + DAY_MS <= now_ms)
    tmp = closed_4h.copy()
    tmp["day_ts"] = (tmp["timestamp"] // DAY_MS) * DAY_MS
    df1d = (
        tmp.groupby("day_ts", as_index=False)
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"), bars=("timestamp", "count"))
        .sort_values("day_ts")
        .reset_index(drop=True)
    )
    closed_1d = df1d[(df1d["day_ts"] + DAY_MS) <= now_ms].reset_index(drop=True)
    if len(closed_1d) < 22:
        raise ValueError(f"Insufficient closed BTC 1D bars ({len(closed_1d)} < 22; fetch >= 180 4H bars)")

    hd = closed_1d["high"].to_numpy(float)
    ld = closed_1d["low"].to_numpy(float)
    cd = closed_1d["close"].to_numpy(float)
    e20_1d = ema_arr(cd, 20)
    atr1d = atr_arr(hd, ld, cd, 14) / np.maximum(cd, 1e-12) * 100.0
    id1 = len(cd) - 1

    d1_dir = 1 if cd[id1] > e20_1d[id1] else -1
    cur_atr1d_pct = float(atr1d[id1])
    in_storm = bool(cur_atr1d_pct > storm_atr_pct)

    return BtcL2Regime(
        bar_ts_4h=int(closed_4h["timestamp"].iloc[i4]),
        close_4h=float(c4[i4]),
        ema20_4h=float(e20_4[i4]),
        ema50_4h=float(e50_4[i4]),
        v1_long=v1_l,
        v1_short=v1_s,
        day_ts_1d=int(closed_1d["day_ts"].iloc[id1]),
        close_1d=float(cd[id1]),
        ema20_1d=float(e20_1d[id1]),
        d1_dir=d1_dir,
        atr1d_pct=round(cur_atr1d_pct, 2),
        in_storm=in_storm,
    )


@dataclass(frozen=True)
class RetailLsSnapshot:
    symbol: str
    z_score: float
    buy_ratio: float
    bar_ts_1h: int
    available: bool

    def blocks_side(self, side: str, max_abs_z: float = 1.0) -> bool:
        if not self.available:
            return False
        if side == "LONG":
            return self.z_score > max_abs_z
        if side == "SHORT":
            return self.z_score < -max_abs_z
        return False


def fetch_retail_ls_snapshot(
    client: BybitClient,
    symbol: str,
    now_ms: int,
    *,
    rolling_hours: int = 72,
    cache_ttl_sec: float = 300.0,
) -> RetailLsSnapshot:
    """
    Fetch Bybit V5 `/v5/market/account-ratio?period=1h` for `symbol` and compute causal 72h rolling z-score.
    Strictly filters `timestamp + H1_MS <= now_ms` (zero look-ahead).
    """
    now_mono = time.monotonic()
    cur_closed_h = ((now_ms // H1_MS) - 1) * H1_MS
    with _ls_cache_lock:
        cached = _ls_cache.get(symbol)
        if cached is not None:
            fetched_at, ts_h, z_val, br_val = cached
            if (now_mono - fetched_at) < cache_ttl_sec and ts_h >= cur_closed_h:
                return RetailLsSnapshot(symbol=symbol, z_score=z_val, buy_ratio=br_val, bar_ts_1h=ts_h, available=True)

    url = f"{client.public_base_url}/v5/market/account-ratio"
    params = {"category": "linear", "symbol": symbol, "period": "1h", "limit": 100}
    sess = _get_ls_session()
    try:
        resp = sess.get(url, params=params, timeout=(5, 10))
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("retCode") != 0:
            logger.debug("account-ratio non-zero retCode for %s: %s", symbol, payload.get("retMsg"))
            return RetailLsSnapshot(symbol=symbol, z_score=0.0, buy_ratio=0.5, bar_ts_1h=0, available=False)
        rows = payload.get("result", {}).get("list", [])
        if len(rows) < 25:
            return RetailLsSnapshot(symbol=symbol, z_score=0.0, buy_ratio=0.5, bar_ts_1h=0, available=False)

        df = pd.DataFrame(rows)
        df["timestamp"] = df["timestamp"].astype(np.int64)
        df["buyRatio"] = df["buyRatio"].astype(float)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df = df[df["timestamp"] + H1_MS <= now_ms].reset_index(drop=True)
        if len(df) < 24:
            return RetailLsSnapshot(symbol=symbol, z_score=0.0, buy_ratio=0.5, bar_ts_1h=0, available=False)

        vals = df["buyRatio"]
        mu = vals.rolling(rolling_hours, min_periods=24).mean()
        sd = vals.rolling(rolling_hours, min_periods=24).std().replace(0.0, np.nan)
        z_series = ((vals - mu) / sd).fillna(0.0)

        z_val = round(float(z_series.iloc[-1]), 3)
        br_val = round(float(vals.iloc[-1]), 4)
        ts_h = int(df["timestamp"].iloc[-1])
        with _ls_cache_lock:
            _ls_cache[symbol] = (now_mono, ts_h, z_val, br_val)
        return RetailLsSnapshot(symbol=symbol, z_score=z_val, buy_ratio=br_val, bar_ts_1h=ts_h, available=True)
    except Exception as exc:
        logger.debug("Retail L/S fetch failed for %s: %s", symbol, exc)
        return RetailLsSnapshot(symbol=symbol, z_score=0.0, buy_ratio=0.5, bar_ts_1h=0, available=False)
