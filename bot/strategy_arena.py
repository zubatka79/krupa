"""Strategy Arena: 3-Strategy Tournament on Top-100 Pure Crypto Bybit USDT Perpetuals.

Fixed Money Management (uniform across all tests):
- Starting deposit: $1,000 (compounded)
- Margin per trade: 10% of current equity
- Leverage: 10x fixed (position notional = 100% of current equity)
- Max open positions: 3 (max 2 same direction)
- Daily loss limit: -10.5% (max ~3 stops/day)
- Entry: Maker limit (0.02% fee, 0.065% micro-pullback, 2-bar TTL)
- TP exit: Maker limit (0.02%), SL/BE exit: Server Stop Taker (0.055%)
- Common guards: closed 15M candles, ATR/ROC anti-pump/dump, BTC shock >0.8%, 3-bar cooldown
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.bybit_client import BybitClient, INTERVAL_TO_MS, KLINE_COLUMNS
from bot.indicators import (
    compute_adx,
    compute_atr,
    compute_daily_vwap,
    compute_mfi,
    compute_rsi,
    compute_rvol,
    ema,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("strategy_arena")


# ==============================================================================
# 1. Async Rate-Limit-Safe Kline Downloader with Incremental .csv.gz Cache
# ==============================================================================
async def _fetch_kline_page(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> List[List[Any]]:
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": str(int(start_ms)),
        "end": str(int(end_ms)),
        "limit": "1000",
    }
    backoff = 0.5
    for attempt in range(6):
        async with sem:
            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2.0, 6.0)
                        continue
                    data = await resp.json()
                    code = data.get("retCode", -1)
                    if code == 0:
                        return data.get("result", {}).get("list", [])
                    if code in (10006, 10018):
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2.0, 6.0)
                        continue
                    return []
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.8, 5.0)
    return []


async def _fetch_kline_range_async(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    interval_ms = INTERVAL_TO_MS[interval]
    rows: List[List[Any]] = []
    cursor_end = end_ms
    max_pages = int((end_ms - start_ms) / (interval_ms * 990)) + 4

    for _ in range(max_pages):
        if cursor_end < start_ms:
            break
        batch = await _fetch_kline_page(session, sem, symbol, interval, start_ms, cursor_end)
        if not batch:
            break
        rows.extend(batch)
        oldest_ts = int(batch[-1][0])
        if oldest_ts <= start_ms or len(batch) < 1000:
            break
        cursor_end = oldest_ts - interval_ms

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df["timestamp"] = df["timestamp"].astype("int64")
    for col in ("open", "high", "low", "close", "volume", "turnover"):
        df[col] = df[col].astype("float64")
    return df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


async def _sync_symbol_cache_async(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    cache_dir: Path,
    symbol: str,
    interval: str,
    req_start: int,
    req_end: int,
    now_ms: int,
) -> Tuple[str, pd.DataFrame, bool]:
    interval_ms = INTERVAL_TO_MS[interval]
    cache_file = cache_dir / f"{symbol}_{interval}.csv.gz"
    cached_df: Optional[pd.DataFrame] = None

    if cache_file.exists():
        try:
            cached_df = pd.read_csv(cache_file, compression="gzip")
            if not cached_df.empty:
                cached_df["timestamp"] = cached_df["timestamp"].astype("int64")
                for col in ("open", "high", "low", "close", "volume", "turnover"):
                    cached_df[col] = cached_df[col].astype("float64")
                cached_df = cached_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        except Exception:
            cached_df = None

    frames: List[pd.DataFrame] = []
    downloaded = False

    if cached_df is None or cached_df.empty:
        fresh = await _fetch_kline_range_async(session, sem, symbol, interval, req_start, req_end)
        if not fresh.empty:
            frames.append(fresh)
            downloaded = True
    else:
        frames.append(cached_df)
        c_min = int(cached_df["timestamp"].iloc[0])
        c_max = int(cached_df["timestamp"].iloc[-1])

        if req_start < c_min - 4 * interval_ms:
            older = await _fetch_kline_range_async(session, sem, symbol, interval, req_start, c_min - interval_ms)
            if not older.empty:
                frames.append(older)
                downloaded = True
        if req_end > c_max + 2 * interval_ms:
            newer = await _fetch_kline_range_async(session, sem, symbol, interval, c_max + interval_ms, req_end)
            if not newer.empty:
                frames.append(newer)
                downloaded = True

    if not frames:
        return symbol, pd.DataFrame(columns=KLINE_COLUMNS), downloaded

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    merged = merged[merged["timestamp"] + interval_ms <= now_ms].reset_index(drop=True)

    if downloaded and not merged.empty:
        tmp = cache_file.with_suffix(".tmp.gz")
        merged.to_csv(tmp, index=False, compression="gzip")
        os.replace(tmp, cache_file)

    filtered = merged[(merged["timestamp"] >= req_start) & (merged["timestamp"] <= req_end)].copy()
    return symbol, filtered.reset_index(drop=True), downloaded


async def download_all_top100_async(
    symbols: List[str],
    cache_dir: Path,
    req_start: int,
    req_end: int,
    concurrency: int = 10,
) -> Dict[str, pd.DataFrame]:
    now_ms = int(time.time() * 1000)
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    results: Dict[str, pd.DataFrame] = {}

    async with aiohttp.ClientSession(connector=connector, headers={"User-Agent": "bybit-arena/1.0"}) as session:
        tasks = [
            asyncio.create_task(_sync_symbol_cache_async(session, sem, cache_dir, sym, "15", req_start, req_end, now_ms))
            for sym in symbols
        ]
        done = 0
        dl_count = 0
        for fut in asyncio.as_completed(tasks):
            sym, df, was_dl = await fut
            done += 1
            if was_dl:
                dl_count += 1
            if not df.empty:
                results[sym] = df
            if done % 15 == 0 or done == len(symbols):
                logger.info("  Synced %d/%d symbols (fetched from API: %d, reused cache: %d)", done, len(symbols), dl_count, done - dl_count)
    return results


# ==============================================================================
# 2. Common Guards & 3 Distinct Strategy Signal Builders (Strictly Causal)
# ==============================================================================
def compute_all_strategy_signals(
    df_15m: pd.DataFrame,
    df_btc_15m: pd.DataFrame,
    timeline_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute common protective filters and signals for all 3 strategies on closed 15M candles:
      - fibo_trend_29 (wave=29)
      - fibo_trend_36 (wave=36)
      - mean_reversion
      - liquidity_sweep
    Zero lookahead: 1H/4H indicators merge strictly on completed bar close timestamps.
    """
    entry_ms = INTERVAL_TO_MS["15"]
    h1_ms = INTERVAL_TO_MS["60"]
    h4_ms = INTERVAL_TO_MS["240"]

    df = df_15m.copy().sort_values("timestamp").reset_index(drop=True)
    df["close_ts"] = (df["timestamp"] + entry_ms).astype("int64")

    # Core 15M indicators
    df["rsi"] = compute_rsi(df["close"], 14)
    df["mfi"] = compute_mfi(df["high"], df["low"], df["close"], df["volume"], 14)
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)) * 100.0
    df["vwap"] = compute_daily_vwap(df)
    df["rvol"] = compute_rvol(df["volume"], 20)
    df["ema5"] = ema(df["close"], 5)

    c_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    body = (df["close"] - df["open"]).abs()
    body_ratio = (body / c_range).fillna(0.0)
    lower_wick_ratio = ((pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]) / c_range).fillna(0.0)
    upper_wick_ratio = ((df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)) / c_range).fillna(0.0)

    # Common protective filters: anti-pump/dump (ATR/ROC) + 3-bar cooldown + BTC shock > 0.8%
    bar_range = df["high"] - df["low"]
    range_3 = df["high"].rolling(3, min_periods=3).max() - df["low"].rolling(3, min_periods=3).min()
    roc_3_abs = (df["close"] - df["close"].shift(3)).abs() / df["close"].shift(3).replace(0.0, np.nan)
    atr_frac = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)

    is_abnormal_vol = (
        (bar_range > 2.2 * df["atr"])
        | (range_3 > 3.8 * df["atr"])
        | (roc_3_abs > 2.2 * atr_frac)
    ).fillna(False)
    vol_cooldown = (
        is_abnormal_vol
        | is_abnormal_vol.shift(1, fill_value=False)
        | is_abnormal_vol.shift(2, fill_value=False)
        | is_abnormal_vol.shift(3, fill_value=False)
    )

    btc = df_btc_15m[["timestamp", "open", "close"]].copy().sort_values("timestamp")
    btc["btc_move"] = ((btc["close"] - btc["open"]).abs() / btc["open"].replace(0.0, np.nan)) * 100.0
    df = pd.merge_asof(df, btc[["timestamp", "btc_move"]], on="timestamp", direction="backward")
    btc_ok = df["btc_move"].fillna(0.0) <= 0.8
    atr_ok = (df["atr_pct"] >= 0.12) & (df["atr_pct"] <= 1.50)

    common_ok = (~vol_cooldown) & btc_ok & atr_ok

    # Strictly closed 1H and 4H context
    df_1h = BybitClient.resample_klines(df_15m, "60")
    df_4h = BybitClient.resample_klines(df_15m, "240")

    h1_out = pd.DataFrame(index=df_1h.index)
    h1_out["close_ts"] = (df_1h["timestamp"] + h1_ms).astype("int64")
    h1_e20 = ema(df_1h["close"], 20)
    h1_e50 = ema(df_1h["close"], 50)
    h1_adx, _, _ = compute_adx(df_1h["high"], df_1h["low"], df_1h["close"], 14)
    h1_out["h1_bull"] = (df_1h["close"] > h1_e50) & (h1_e20 > h1_e50) & (h1_e50 > h1_e50.shift(2))
    h1_out["h1_bear"] = (df_1h["close"] < h1_e50) & (h1_e20 < h1_e50) & (h1_e50 < h1_e50.shift(2))
    h1_out["h1_adx"] = h1_adx.values
    h1_out["h1_stretch"] = ((df_1h["close"] - h1_e20).abs() / h1_e20.replace(0.0, np.nan)).fillna(0.0).values

    h4_out = pd.DataFrame(index=df_4h.index)
    h4_out["close_ts"] = (df_4h["timestamp"] + h4_ms).astype("int64")
    h4_adx, _, _ = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], 14)
    h4_out["h4_adx"] = h4_adx.values

    df = pd.merge_asof(df, h1_out.sort_values("close_ts"), on="close_ts", direction="backward")
    df = pd.merge_asof(df, h4_out.sort_values("close_ts"), on="close_ts", direction="backward")
    h1_bull = df["h1_bull"].astype("boolean").fillna(False)
    h1_bear = df["h1_bear"].astype("boolean").fillna(False)
    h1_adx_s = df["h1_adx"].fillna(0.0)
    h4_adx_s = df["h4_adx"].fillna(0.0)
    h1_stretch_s = df["h1_stretch"].fillna(0.0)

    # ------------------------------------------------------------------
    # STRATEGY 1: fibo_trend (wave period = 29 and 36 on 15M + 4-6 bar EMA + 1H trend)
    # ------------------------------------------------------------------
    min_low_2 = pd.concat([df["low"], df["low"].shift(1)], axis=1).min(axis=1)
    max_high_2 = pd.concat([df["high"], df["high"].shift(1)], axis=1).max(axis=1)
    short_trend_up = (df["close"] > df["ema5"]) & (df["ema5"] > df["ema5"].shift(4))
    short_trend_dn = (df["close"] < df["ema5"]) & (df["ema5"] < df["ema5"].shift(4))
    bull_confirm = (df["close"] > df["open"]) & (body_ratio >= 0.35) & (df["close"] > df["close"].shift(1))
    bear_confirm = (df["close"] < df["open"]) & (body_ratio >= 0.35) & (df["close"] < df["close"].shift(1))

    for period in (29, 36):
        w_hi = df["high"].shift(1).rolling(period, min_periods=period).max()
        w_lo = df["low"].shift(1).rolling(period, min_periods=period).min()
        w_rng = w_hi - w_lo
        w_ok = w_rng >= (2.2 * df["atr"])

        # Pullback into 0.50 - 0.618 - 0.786 Fibonacci retracement pocket
        fib_long_zone = (
            w_ok
            & (min_low_2 <= w_hi - 0.48 * w_rng)
            & (min_low_2 >= w_hi - 0.80 * w_rng)
            & (df["close"] > w_hi - 0.786 * w_rng)
        )
        fib_short_zone = (
            w_ok
            & (max_high_2 >= w_lo + 0.48 * w_rng)
            & (max_high_2 <= w_lo + 0.80 * w_rng)
            & (df["close"] < w_lo + 0.786 * w_rng)
        )

        df[f"fibo{period}_L"] = common_ok & h1_bull & (h1_adx_s >= 16.0) & short_trend_up & fib_long_zone & bull_confirm
        df[f"fibo{period}_S"] = common_ok & h1_bear & (h1_adx_s >= 16.0) & short_trend_dn & fib_short_zone & bear_confirm
        df[f"fibo{period}_Q"] = (h1_adx_s + body_ratio * 25.0).fillna(0.0)

    # ------------------------------------------------------------------
    # STRATEGY 2: mean_reversion (BB(20,2) / VWAP stretch + RSI/MFI extreme + reversal, no crash trend)
    # ------------------------------------------------------------------
    bb_mid = df["close"].rolling(20, min_periods=20).mean()
    bb_std = df["close"].rolling(20, min_periods=20).std()
    bb_up = bb_mid + 2.0 * bb_std
    bb_lo = bb_mid - 2.0 * bb_std

    no_crash_trend = (h1_adx_s < 32.0) & (h4_adx_s < 32.0) & (h1_stretch_s < 0.035)
    rsi_min2 = pd.concat([df["rsi"], df["rsi"].shift(1)], axis=1).min(axis=1)
    rsi_max2 = pd.concat([df["rsi"], df["rsi"].shift(1)], axis=1).max(axis=1)
    mfi_min2 = pd.concat([df["mfi"], df["mfi"].shift(1)], axis=1).min(axis=1)
    mfi_max2 = pd.concat([df["mfi"], df["mfi"].shift(1)], axis=1).max(axis=1)

    mr_long = (
        common_ok
        & no_crash_trend
        & ((df["low"] < bb_lo) | (df["low"].shift(1) < bb_lo.shift(1)))
        & (df["close"] < df["vwap"] * 0.994)
        & (rsi_min2 < 32.0)
        & (mfi_min2 < 30.0)
        & (df["close"] > df["open"])
        & (df["close"] >= bb_lo * 0.997)
        & (df["rsi"] > df["rsi"].shift(1))
    )
    mr_short = (
        common_ok
        & no_crash_trend
        & ((df["high"] > bb_up) | (df["high"].shift(1) > bb_up.shift(1)))
        & (df["close"] > df["vwap"] * 1.006)
        & (rsi_max2 > 68.0)
        & (mfi_max2 > 70.0)
        & (df["close"] < df["open"])
        & (df["close"] <= bb_up * 1.003)
        & (df["rsi"] < df["rsi"].shift(1))
    )
    df["mr_L"] = mr_long
    df["mr_S"] = mr_short
    df["mr_Q"] = ((50.0 - df["rsi"]).abs() + (50.0 - df["mfi"]).abs() * 0.5).fillna(0.0)

    # ------------------------------------------------------------------
    # STRATEGY 3: liquidity_sweep (24-bar swing pierce on high volume + instant reclaim)
    # ------------------------------------------------------------------
    sw_lo = df["low"].shift(1).rolling(24, min_periods=20).min()
    sw_hi = df["high"].shift(1).rolling(24, min_periods=20).max()

    ls_long = (
        common_ok
        & (df["low"] < sw_lo * 0.999)
        & (df["close"] > sw_lo)
        & (df["rvol"] >= 1.4)
        & (lower_wick_ratio >= 0.35)
        & (df["close"] >= df["open"])
    )
    ls_short = (
        common_ok
        & (df["high"] > sw_hi * 1.001)
        & (df["close"] < sw_hi)
        & (df["rvol"] >= 1.4)
        & (upper_wick_ratio >= 0.35)
        & (df["close"] <= df["open"])
    )
    df["ls_L"] = ls_long
    df["ls_S"] = ls_short
    df["ls_Q"] = (df["rvol"] * 15.0 + pd.concat([lower_wick_ratio, upper_wick_ratio], axis=1).max(axis=1) * 30.0).fillna(0.0)

    cols = [
        "timestamp", "high", "low", "close",
        "fibo29_L", "fibo29_S", "fibo29_Q",
        "fibo36_L", "fibo36_S", "fibo36_Q",
        "mr_L", "mr_S", "mr_Q",
        "ls_L", "ls_S", "ls_Q",
    ]
    aligned = timeline_df.merge(df[cols], on="timestamp", how="left")

    out: Dict[str, np.ndarray] = {
        "high": aligned["high"].to_numpy(dtype=np.float64),
        "low": aligned["low"].to_numpy(dtype=np.float64),
        "close": aligned["close"].to_numpy(dtype=np.float64),
    }
    for prefix in ("fibo29", "fibo36", "mr", "ls"):
        out[f"{prefix}_L"] = aligned[f"{prefix}_L"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        out[f"{prefix}_S"] = aligned[f"{prefix}_S"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        out[f"{prefix}_Q"] = aligned[f"{prefix}_Q"].fillna(0.0).to_numpy(dtype=np.float64)
    return out


# ==============================================================================
# 3. Fixed-Rule Portfolio Simulator ($1,000 start, 10% margin x 10x = 100% notional)
# ==============================================================================
@dataclass
class ArenaConfig:
    strategy_key: str   # "fibo29", "fibo36", "mr", "ls"
    strategy_name: str
    tp_pct: float
    sl_pct: float
    smart_be: bool
    maker_pullback_pct: float = 0.065  # midpoint of 0.05%-0.08%
    maker_ttl_bars: int = 2
    be_trigger_ratio: float = 0.65
    be_offset_pct: float = 0.15

    @property
    def label(self) -> str:
        be = f"BE(+0.15% @ 65%)" if self.smart_be else "NoBE"
        return f"{self.strategy_name} | TP {self.tp_pct:.2f}% / SL {self.sl_pct:.1f}% | {be}"


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
    period: str
    entry_ts: int
    entry_price: float
    sl_price: float
    tp_price: float
    be_trigger_price: Optional[float]
    be_stop_price: Optional[float]
    qty: float
    notional: float
    entry_fee: float
    be_active: bool = False


def simulate_fixed_10x(
    cfg: ArenaConfig,
    symbols: List[str],
    sym_data: Dict[str, Dict[str, np.ndarray]],
    timeline: np.ndarray,
    split_ts: int,
    initial_deposit: float = 1000.0,
    max_open: int = 3,
    max_same_dir: int = 2,
    daily_loss_limit_pct: float = 10.5,
    cooldown_bars: int = 3,
) -> Dict[str, Any]:
    """
    Exact simulation under fixed money rules:
      - Margin = 10% of current equity, Leverage = 10x -> Notional = 100% of current equity
      - Maker entry (0.02% fee, 0.065% pullback, 2-bar TTL)
      - Maker TP (0.02% fee), Taker SL/BE (0.055% fee)
      - Daily loss limit = -10.5%, 3-bar per-coin cooldown
    """
    maker_rate = 0.0002
    taker_rate = 0.00055
    tp_frac = cfg.tp_pct / 100.0
    sl_frac = cfg.sl_pct / 100.0
    pb_frac = cfg.maker_pullback_pct / 100.0
    be_trig_frac = (cfg.tp_pct * cfg.be_trigger_ratio) / 100.0
    be_off_frac = cfg.be_offset_pct / 100.0

    l_key = f"{cfg.strategy_key}_L"
    s_key = f"{cfg.strategy_key}_S"
    q_key = f"{cfg.strategy_key}_Q"

    equity = float(initial_deposit)
    peak_eq = equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0

    eq_at_split: Optional[float] = None
    open_pos: List[_OpenPos] = []
    pending: List[_Pending] = []
    cooldown_until: Dict[str, int] = {}
    trades: List[Dict[str, Any]] = []

    n_bars = len(timeline)
    cur_day = -1
    day_start_eq = equity
    daily_halted = False

    def _close(pos: _OpenPos, exit_p: float, reason: str, exit_maker: bool, ts: int, bar_i: int) -> None:
        nonlocal equity, peak_eq, max_dd_usd, max_dd_pct
        gross = (
            (exit_p - pos.entry_price) * pos.qty
            if pos.direction == "LONG"
            else (pos.entry_price - exit_p) * pos.qty
        )
        exit_fee = abs(pos.qty * exit_p) * (maker_rate if exit_maker else taker_rate)
        fees = pos.entry_fee + exit_fee
        net = gross - fees
        equity = max(1.0, equity + net)

        if equity > peak_eq:
            peak_eq = equity
        dd_u = peak_eq - equity
        dd_p = (dd_u / peak_eq) * 100.0 if peak_eq > 0 else 0.0
        if dd_u > max_dd_usd:
            max_dd_usd = dd_u
        if dd_p > max_dd_pct:
            max_dd_pct = dd_p

        trades.append(
            {
                "symbol": pos.symbol,
                "direction": pos.direction,
                "period": pos.period,
                "entry_ts": pos.entry_ts,
                "exit_ts": ts,
                "reason": reason,
                "net_usd": net,
                "ret_pct": (net / max(pos.notional, 1e-9)) * 100.0,  # notional == equity at entry!
                "fees_usd": fees,
            }
        )
        cooldown_until[pos.symbol] = bar_i + cooldown_bars

    for i in range(n_bars):
        ts = int(timeline[i])
        if eq_at_split is None and ts >= split_ts:
            eq_at_split = equity

        day_key = ts // 86_400_000
        if day_key != cur_day:
            cur_day = day_key
            day_start_eq = equity
            daily_halted = False

        # 1. Manage open positions (strictly conservative intra-bar order)
        if open_pos:
            still_open: List[_OpenPos] = []
            for pos in open_pos:
                sd = sym_data[pos.symbol]
                hi = sd["high"][i]
                lo = sd["low"][i]
                if not np.isfinite(lo):
                    still_open.append(pos)
                    continue

                is_long = pos.direction == "LONG"
                if cfg.smart_be and pos.be_active:
                    be_p = pos.be_stop_price if pos.be_stop_price is not None else pos.sl_price
                    be_hit = (lo <= be_p) if is_long else (hi >= be_p)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if be_hit:
                        _close(pos, be_p, "BE", False, ts, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, ts, i)
                    else:
                        still_open.append(pos)
                else:
                    sl_hit = (lo <= pos.sl_price) if is_long else (hi >= pos.sl_price)
                    tp_hit = (hi >= pos.tp_price) if is_long else (lo <= pos.tp_price)
                    if sl_hit:
                        _close(pos, pos.sl_price, "SL", False, ts, i)
                    elif tp_hit:
                        _close(pos, pos.tp_price, "TP", True, ts, i)
                    else:
                        if cfg.smart_be and pos.be_trigger_price is not None:
                            trig = (hi >= pos.be_trigger_price) if is_long else (lo <= pos.be_trigger_price)
                            if trig:
                                pos.be_active = True
                        still_open.append(pos)
            open_pos = still_open

        # 2. Daily -10.5% circuit breaker (realized + unrealized)
        if not daily_halted:
            unreal = 0.0
            for pos in open_pos:
                cl = sym_data[pos.symbol]["close"][i]
                if np.isfinite(cl):
                    g = (cl - pos.entry_price) * pos.qty if pos.direction == "LONG" else (pos.entry_price - cl) * pos.qty
                    unreal += g - pos.entry_fee
            dd_day = ((day_start_eq - (equity + unreal)) / max(day_start_eq, 1e-9)) * 100.0
            if dd_day >= daily_loss_limit_pct:
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

                sd = sym_data[ord_p.symbol]
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

                # Fixed sizing: 10% margin x 10x leverage = 100% of current equity
                notional = equity
                entry_p = ord_p.limit_price
                qty = notional / entry_p
                entry_fee = notional * maker_rate

                if is_long:
                    sl_p = entry_p * (1.0 - sl_frac)
                    tp_p = entry_p * (1.0 + tp_frac)
                    be_tr = entry_p * (1.0 + be_trig_frac) if cfg.smart_be else None
                    be_st = entry_p * (1.0 + be_off_frac) if cfg.smart_be else None
                else:
                    sl_p = entry_p * (1.0 + sl_frac)
                    tp_p = entry_p * (1.0 - tp_frac)
                    be_tr = entry_p * (1.0 - be_trig_frac) if cfg.smart_be else None
                    be_st = entry_p * (1.0 - be_off_frac) if cfg.smart_be else None

                pos = _OpenPos(
                    symbol=ord_p.symbol,
                    direction=ord_p.direction,
                    period="IS" if ts < split_ts else "OOS",
                    entry_ts=ts,
                    entry_price=entry_p,
                    sl_price=sl_p,
                    tp_price=tp_p,
                    be_trigger_price=be_tr,
                    be_stop_price=be_st,
                    qty=qty,
                    notional=notional,
                    entry_fee=entry_fee,
                )

                # Conservative same-bar SL check on fill bar (no optimistic same-bar TP)
                if (lo <= sl_p) if is_long else (hi >= sl_p):
                    _close(pos, sl_p, "SL", False, ts, i)
                else:
                    open_pos.append(pos)
            pending = next_pend

        if daily_halted or len(open_pos) >= max_open:
            continue

        # 4. Scan closed 15M bar `i` for new signals
        busy = {p.symbol for p in open_pos} | {o.symbol for o in pending}
        cands: List[Tuple[float, str, str, float]] = []
        for sym in symbols:
            if sym in busy or i < cooldown_until.get(sym, -1):
                continue
            sd = sym_data[sym]
            if sd[l_key][i]:
                cands.append((float(sd[q_key][i]), sym, "LONG", float(sd["close"][i])))
            elif sd[s_key][i]:
                cands.append((float(sd[q_key][i]), sym, "SHORT", float(sd["close"][i])))

        if not cands:
            continue
        cands.sort(key=lambda x: x[0], reverse=True)

        for _, sym, direction, cl_p in cands:
            if len(open_pos) + len(pending) >= max_open + 2:
                break
            if sum(1 for p in open_pos if p.direction == direction) >= max_same_dir:
                continue
            limit_p = cl_p * (1.0 - pb_frac) if direction == "LONG" else cl_p * (1.0 + pb_frac)
            pending.append(_Pending(symbol=sym, direction=direction, limit_price=limit_p, expires_idx=i + cfg.maker_ttl_bars))

    # Close remaining positions at end of dataset
    if open_pos and n_bars > 0:
        last_ts = int(timeline[-1])
        for pos in open_pos:
            cl = sym_data[pos.symbol]["close"][-1]
            if not np.isfinite(cl):
                cl = pos.entry_price
            _close(pos, float(cl), "EOD", False, last_ts, n_bars - 1)

    if eq_at_split is None:
        eq_at_split = equity

    n_tr = len(trades)
    wins = sum(1 for t in trades if t["net_usd"] > 0)
    is_trades = [t for t in trades if t["period"] == "IS"]
    oos_trades = [t for t in trades if t["period"] == "OOS"]

    # Dollar PnL cleanly split across continuous compounded equity curve
    is_net_usd = sum(t["net_usd"] for t in is_trades)
    oos_net_usd = sum(t["net_usd"] for t in oos_trades)
    total_net_usd = equity - initial_deposit

    # Standalone normalized % returns for IS and OOS stability verification
    eq_is_norm = 100.0
    for t in is_trades:
        eq_is_norm *= 1.0 + t["ret_pct"] / 100.0
    eq_oos_norm = 100.0
    for t in oos_trades:
        eq_oos_norm *= 1.0 + t["ret_pct"] / 100.0

    return {
        "start_usd": round(initial_deposit, 2),
        "end_usd": round(equity, 2),
        "total_net_usd": round(total_net_usd, 2),
        "is_net_usd": round(is_net_usd, 2),
        "oos_net_usd": round(oos_net_usd, 2),
        "is_net_pct": round(eq_is_norm - 100.0, 2),
        "oos_net_pct": round(eq_oos_norm - 100.0, 2),
        "total_net_pct": round((total_net_usd / initial_deposit) * 100.0, 2),
        "trades": n_tr,
        "trades_per_day": round(n_tr / 180.0, 2),
        "is_trades": len(is_trades),
        "oos_trades": len(oos_trades),
        "win_rate_pct": round((wins / n_tr) * 100.0, 2) if n_tr > 0 else 0.0,
        "tp_count": sum(1 for t in trades if t["reason"] == "TP"),
        "be_count": sum(1 for t in trades if t["reason"] == "BE"),
        "sl_count": sum(1 for t in trades if t["reason"] == "SL"),
        "fees_usd": round(sum(t["fees_usd"] for t in trades), 2),
        "max_dd_usd": round(max_dd_usd, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "trades_list": trades,
    }


# ==============================================================================
# 4. Main Tournament Orchestrator
# ==============================================================================
def run_arena(days: int = 180, top_n: int = 100, is_days: int = 120, oos_days: int = 60) -> Dict[str, Any]:
    t0 = time.time()
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = BybitClient(mode="testnet", cache_dir=cache_dir, public_mainnet=True)

    logger.info("Fetching Top-%d PURE CRYPTO Bybit USDT perpetuals (min_order <= 50 USDT)...", top_n)
    raw_top = client.get_top_usdt_perpetuals(top_n=top_n + 15, max_min_order_usdt=50.0, crypto_only=True)
    universe = [s for s in raw_top if s.symbol != "BTCUSDT"][:top_n]
    symbols_100 = [s.symbol for s in universe]
    logger.info("Selected %d pure crypto coins (rank #1 %s .. #%d %s).", len(symbols_100), symbols_100[0], len(symbols_100), symbols_100[-1])

    entry_ms = INTERVAL_TO_MS["15"]
    now_ms = int(time.time() * 1000)
    sim_end_ms = ((now_ms // entry_ms) - 1) * entry_ms
    sim_start_ms = sim_end_ms - days * 86_400_000
    warmup_days = 10
    fetch_start_ms = sim_start_ms - warmup_days * 86_400_000
    split_ms = sim_start_ms + is_days * 86_400_000

    all_to_sync = ["BTCUSDT"] + symbols_100
    logger.info("Syncing 180d (+%dd warmup) 15M klines for %d symbols via async aiohttp...", warmup_days, len(all_to_sync))
    raw_frames = asyncio.run(download_all_top100_async(all_to_sync, cache_dir, fetch_start_ms, sim_end_ms, concurrency=10))

    df_btc = raw_frames["BTCUSDT"]
    btc_sim_ts = df_btc[(df_btc["timestamp"] >= sim_start_ms) & (df_btc["timestamp"] <= sim_end_ms)]["timestamp"].to_numpy(dtype=np.int64)
    timeline = np.sort(np.unique(btc_sim_ts))
    timeline_df = pd.DataFrame({"timestamp": timeline})

    logger.info("Computing signals for all 3 strategies across %d coins...", len(symbols_100))
    sym_data: Dict[str, Dict[str, np.ndarray]] = {}
    active_symbols: List[str] = []
    for sym in symbols_100:
        df_s = raw_frames.get(sym)
        if df_s is None or len(df_s) < 400:
            continue
        sym_data[sym] = compute_all_strategy_signals(df_s, df_btc, timeline_df)
        active_symbols.append(sym)
    logger.info("Prepared %d active coins with valid historical data.", len(active_symbols))

    tp_sl_grid = [
        (1.25, 3.5),
        (1.50, 3.0),
        (1.75, 3.5),
        (2.00, 3.5),
    ]

    strategy_families = [
        ("fibo_trend", [("fibo29", "Фибоначчи по тренду (волна 29)"), ("fibo36", "Фибоначчи по тренду (волна 36)")]),
        ("mean_reversion", [("mr", "Перегрев и возврат к средней (BB/VWAP+RSI/MFI)")]),
        ("liquidity_sweep", [("ls", "Снятие ликвидности / ложный пробой (24 свечи)")]),
    ]

    arena_summary: Dict[str, Any] = {}

    for fam_id, sub_strats in strategy_families:
        logger.info("=== Evaluating Strategy Family: %s ===", fam_id)
        best_variant: Optional[Dict[str, Any]] = None
        all_variants: List[Dict[str, Any]] = []

        for s_key, s_name in sub_strats:
            for tp, sl in tp_sl_grid:
                for smart_be in (False, True):
                    cfg = ArenaConfig(
                        strategy_key=s_key,
                        strategy_name=s_name,
                        tp_pct=tp,
                        sl_pct=sl,
                        smart_be=smart_be,
                    )

                    # Stage A: Standalone scan of all 100 coins -> find stable coins positive in BOTH IS (4m) & OOS (2m)
                    stable_coins: List[Dict[str, Any]] = []
                    for sym in active_symbols:
                        res_c = simulate_fixed_10x(
                            cfg=cfg,
                            symbols=[sym],
                            sym_data={sym: sym_data[sym]},
                            timeline=timeline,
                            split_ts=split_ms,
                            initial_deposit=1000.0,
                            max_open=1,
                            max_same_dir=1,
                            daily_loss_limit_pct=10.5,
                            cooldown_bars=3,
                        )
                        if res_c["is_net_pct"] > 0.0 and res_c["oos_net_pct"] > 0.0 and res_c["is_trades"] >= 3 and res_c["oos_trades"] >= 2:
                            stable_coins.append(
                                {
                                    "symbol": sym,
                                    "trades": res_c["trades"],
                                    "is_trades": res_c["is_trades"],
                                    "oos_trades": res_c["oos_trades"],
                                    "win_rate_pct": res_c["win_rate_pct"],
                                    "is_net_pct": res_c["is_net_pct"],
                                    "oos_net_pct": res_c["oos_net_pct"],
                                    "total_net_pct": res_c["total_net_pct"],
                                    "total_net_usd": res_c["total_net_usd"],
                                }
                            )

                    stable_coins.sort(key=lambda c: (c["total_net_pct"], c["oos_net_pct"]), reverse=True)
                    top_basket = stable_coins[:12]  # Top-12 stable coins (within requested Top 10-15)
                    basket_syms = [c["symbol"] for c in top_basket]

                    # Stage B: Portfolio simulation on the selected Top-12 stable coins starting at $1,000
                    if basket_syms:
                        port = simulate_fixed_10x(
                            cfg=cfg,
                            symbols=basket_syms,
                            sym_data=sym_data,
                            timeline=timeline,
                            split_ts=split_ms,
                            initial_deposit=1000.0,
                            max_open=3,
                            max_same_dir=2,
                            daily_loss_limit_pct=10.5,
                            cooldown_bars=3,
                        )
                        port.pop("trades_list", None)
                    else:
                        port = {
                            "start_usd": 1000.0, "end_usd": 1000.0, "total_net_usd": 0.0,
                            "is_net_usd": 0.0, "oos_net_usd": 0.0, "is_net_pct": 0.0,
                            "oos_net_pct": 0.0, "total_net_pct": 0.0, "trades": 0,
                            "trades_per_day": 0.0, "is_trades": 0, "oos_trades": 0,
                            "win_rate_pct": 0.0, "tp_count": 0, "be_count": 0, "sl_count": 0,
                            "fees_usd": 0.0, "max_dd_usd": 0.0, "max_dd_pct": 0.0,
                        }

                    variant_rec = {
                        "label": cfg.label,
                        "config": asdict(cfg),
                        "stable_coins_count": len(stable_coins),
                        "basket_size": len(basket_syms),
                        "basket_symbols": basket_syms,
                        "top_10_champions": stable_coins[:10],
                        "portfolio": port,
                    }
                    all_variants.append(variant_rec)

        # Pick best variant for this strategy family (require positive IS & OOS, maximize End USD / low DD)
        all_variants.sort(
            key=lambda v: (
                1.0 if (v["portfolio"]["is_net_usd"] > 0 and v["portfolio"]["oos_net_usd"] > 0) else 0.0,
                1.0 if v["stable_coins_count"] >= 10 else 0.0,
                v["portfolio"]["end_usd"],
                -v["portfolio"]["max_dd_pct"],
            ),
            reverse=True,
        )
        best_variant = all_variants[0]
        arena_summary[fam_id] = {
            "winner": best_variant,
            "all_variants_summary": [
                {
                    "label": v["label"],
                    "stable_coins_count": v["stable_coins_count"],
                    "end_usd": v["portfolio"]["end_usd"],
                    "is_net_usd": v["portfolio"]["is_net_usd"],
                    "oos_net_usd": v["portfolio"]["oos_net_usd"],
                    "trades": v["portfolio"]["trades"],
                    "win_rate_pct": v["portfolio"]["win_rate_pct"],
                    "max_dd_pct": v["portfolio"]["max_dd_pct"],
                }
                for v in all_variants
            ],
        }
        p = best_variant["portfolio"]
        logger.info(
            "  Winner [%s]: %s | Stable=%d/100 | $1,000 -> $%.2f (4m: %+.2f$, 2m: %+.2f$) | %d tr | WR=%.1f%% | MaxDD=%.1f%% ($%.2f)",
            fam_id, best_variant["label"], best_variant["stable_coins_count"],
            p["end_usd"], p["is_net_usd"], p["oos_net_usd"],
            p["trades"], p["win_rate_pct"], p["max_dd_pct"], p["max_dd_usd"],
        )

    elapsed = round(time.time() - t0, 2)
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "start_utc": datetime.fromtimestamp(sim_start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "split_utc": datetime.fromtimestamp(split_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "end_utc": datetime.fromtimestamp(sim_end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "universe_requested": len(symbols_100),
            "universe_active": len(active_symbols),
            "elapsed_sec": elapsed,
            "symbols_100": symbols_100,
        },
        "strategies": arena_summary,
    }

    out_path = Path("data/strategy_arena_results.json")
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Tournament finished in %.2fs -> saved to %s", elapsed, out_path)

    _print_simple_dollar_report(report)
    return report


def _print_simple_dollar_report(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 96)
    print(f"STRATEGY ARENA RESULTS | TOP-{report['meta']['universe_active']} PURE CRYPTO | START: $1,000 (10% margin x 10x)")
    print(f"Period: {report['meta']['start_utc']} -> {report['meta']['end_utc']} (IS=4m, OOS=2m)")
    print("=" * 96)

    for fam_id in ("fibo_trend", "mean_reversion", "liquidity_sweep"):
        w = report["strategies"][fam_id]["winner"]
        p = w["portfolio"]
        print(f"\n>>> {w['label']}")
        print(f"    Stable coins (IS>0 & OOS>0): {w['stable_coins_count']} / {report['meta']['universe_active']} (traded Top-{w['basket_size']})")
        print(
            f"    Deposit: $1,000 -> ${p['end_usd']:,.2f} | Net Profit: {p['total_net_usd']:+,.2f}$ "
            f"(First 4m: {p['is_net_usd']:+,.2f}$ | Last 2m: {p['oos_net_usd']:+,.2f}$)"
        )
        print(
            f"    Trades: {p['trades']} ({p['trades_per_day']:.2f}/day) | Win Rate: {p['win_rate_pct']:.2f}% "
            f"(TP={p['tp_count']}, BE={p['be_count']}, SL={p['sl_count']}) | Fees: ${p['fees_usd']:,.2f}"
        )
        print(f"    Max Drawdown: ${p['max_dd_usd']:,.2f} ({p['max_dd_pct']:.2f}%)")
        print("    Top-10 Champion Coins (standalone on $1,000):")
        for rank, c in enumerate(w["top_10_champions"], 1):
            print(
                f"      {rank:>2}. {c['symbol']:<14} | {c['trades']:>3} tr | WR {c['win_rate_pct']:>5.1f}% | "
                f"4m: {c['is_net_pct']:>+6.1f}% | 2m: {c['oos_net_pct']:>+6.1f}% | 6m: {c['total_net_pct']:>+6.1f}% ({c['total_net_usd']:>+7.2f}$)"
            )
    print("\n" + "=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="3-Strategy Tournament on Top-100 Pure Crypto Bybit Perpetuals")
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()
    run_arena(days=args.days, top_n=args.top)


if __name__ == "__main__":
    main()
