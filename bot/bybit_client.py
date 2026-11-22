"""Bybit v5 API client: keyless public data with incremental .csv.gz caching + private pybit execution."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    from pybit.unified_trading import HTTP as PybitHTTP
except ImportError:
    PybitHTTP = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

INTERVAL_TO_MS: Dict[str, int] = {
    "1": 60_000,
    "3": 3 * 60_000,
    "5": 5 * 60_000,
    "15": 15 * 60_000,
    "30": 30 * 60_000,
    "60": 60 * 60_000,
    "120": 120 * 60_000,
    "240": 240 * 60_000,
    "360": 360 * 60_000,
    "720": 720 * 60_000,
    "D": 86_400_000,
}

KLINE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]

# Non-crypto Bybit symbolType values (stocks, ETFs, commodities, forex, indices)
TRADFI_SYMBOL_TYPES = {"stock", "etf", "commodity", "forex", "index", "fx", "equity", "metal"}

# Explicit blocklist of TradFi stocks, ETFs, commodities, tokenized metals, and FX tickers
TRADFI_BASE_COINS = {
    "NVDA", "QQQ", "SOXL", "SOXS", "CL", "XAU", "XAG", "XAUT", "PAXG", "MU",
    "KORU", "SKHYNIX", "SPCX", "SNDK", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL",
    "GOOG", "META", "SPY", "NAS100", "US30", "US500", "SPX", "NDX", "DJI",
    "AMD", "INTC", "COIN", "MSTR", "NFLX", "PLTR", "HOOD", "BABA", "TSM",
    "TQQQ", "SQQQ", "IWM", "DIA", "GLD", "SLV", "USO", "BRENT", "WTI",
    "NATGAS", "COPPER", "XPT", "XPD", "EUR", "GBP", "JPY", "AUD", "CAD",
    "CHF", "NZD", "CNH", "EURT", "XAUUSD", "XAGUSD", "CRCL", "ARM", "AVGO",
    "ORCL", "JPM", "GS", "V", "MA", "WMT", "COST", "DIS", "UBER", "SMCI",
}

# Stablecoin base assets excluded from directional perpetual trading
STABLECOIN_BASE_COINS = {
    "USDC", "USDE", "FDUSD", "TUSD", "DAI", "USDD", "PYUSD", "EURC", "BUSD",
    "USDT", "RLUSD", "USD1", "FRAX", "GUSD", "USDP", "LUSD", "CRVUSD", "GHO",
}


@dataclass
class InstrumentSpec:
    symbol: str
    base_coin: str
    quote_coin: str
    status: str
    min_order_qty: float
    qty_step: float
    min_notional: float
    tick_size: float
    min_leverage: float
    max_leverage: float
    symbol_type: str = ""
    last_price: float = 0.0
    turnover_24h: float = 0.0
    volume_24h: float = 0.0
    funding_rate: float = 0.0
    open_interest: float = 0.0

    @property
    def min_order_usdt(self) -> float:
        by_qty = self.min_order_qty * self.last_price if self.last_price > 0 else 0.0
        return max(by_qty, self.min_notional)

    @property
    def is_pure_crypto(self) -> bool:
        """Return True strictly for pure crypto perpetual contracts (no TradFi or stablecoins)."""
        if self.symbol_type.strip().lower() in TRADFI_SYMBOL_TYPES:
            return False
        base = self.base_coin.upper().strip()
        if base in TRADFI_BASE_COINS or base in STABLECOIN_BASE_COINS:
            return False
        return True


def round_step_down(value: float, step: float) -> float:
    """Round value down to the nearest valid exchange step size."""
    if step <= 0:
        return value
    precision = max(0, int(round(-math.log10(step)))) if step < 1 else 0
    steps = math.floor((value + 1e-12) / step)
    return round(steps * step, precision)


def round_tick(value: float, tick: float) -> float:
    """Round price to the nearest valid tick size."""
    if tick <= 0:
        return value
    precision = max(0, int(round(-math.log10(tick)))) if tick < 1 else 0
    return round(round(value / tick) * tick, precision)


class BybitClient:
    """Hybrid Bybit v5 client for public market data + private futures trading."""

    MAINNET_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"
    DEMO_URL = "https://api-demo.bybit.com"

    def __init__(
        self,
        mode: str = "testnet",
        cache_dir: str | Path = "data/cache",
        public_mainnet: bool = True,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 15,
    ) -> None:
        load_dotenv()
        self.mode = mode.lower().strip()
        self.is_demo = self.mode == "demo"
        self.testnet = self.mode == "testnet"
        if self.is_demo:
            self.private_base_url = self.DEMO_URL
        elif self.testnet:
            self.private_base_url = self.TESTNET_URL
        else:
            self.private_base_url = self.MAINNET_URL

        self.public_base_url = self.MAINNET_URL if public_mainnet else self.private_base_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = max(10, min(int(timeout), 15))

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "bybit-autonomous-engine/1.0"})

        if api_key is not None:
            self.api_key = api_key
        elif self.is_demo:
            self.api_key = (os.getenv("BYBIT_DEMO_API_KEY", "").strip() or os.getenv("BYBIT_API_KEY", "").strip())
        else:
            self.api_key = os.getenv("BYBIT_API_KEY", "").strip()

        if api_secret is not None:
            self.api_secret = api_secret
        elif self.is_demo:
            self.api_secret = (os.getenv("BYBIT_DEMO_API_SECRET", "").strip() or os.getenv("BYBIT_API_SECRET", "").strip())
        else:
            self.api_secret = os.getenv("BYBIT_API_SECRET", "").strip()

        self._private_client: Any = None
        self._init_private_client()

    def _init_private_client(self) -> None:
        if self.has_credentials and PybitHTTP is not None:
            self._private_client = PybitHTTP(
                demo=self.is_demo,
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=10000,
                timeout=self.timeout,
                max_retries=2,
                retry_delay=1,
            )

    def reset_sessions(self) -> None:
        """Close and recreate public and private HTTP sessions after network/socket timeout or sleep wake-up."""
        try:
            self._session.close()
        except Exception:
            pass
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "bybit-autonomous-engine/1.0"})
        if self._private_client is not None:
            try:
                inner = getattr(self._private_client, "client", None)
                if inner is not None:
                    inner.close()
            except Exception:
                pass
        self._init_private_client()

    @property
    def has_credentials(self) -> bool:
        return bool(
            self.api_key
            and self.api_secret
            and "your_bybit" not in self.api_key.lower()
            and "your_bybit" not in self.api_secret.lower()
        )

    # ------------------------------------------------------------------
    # Low-level public REST helper with retry & rate-limit backoff
    # ------------------------------------------------------------------
    def _public_get(self, path: str, params: Optional[Dict[str, Any]] = None, retries: int = 4) -> Dict[str, Any]:
        url = f"{self.public_base_url}{path}"
        backoff = 0.6
        last_err: Optional[Exception] = None
        req_timeout = (10, self.timeout)
        for attempt in range(retries):
            try:
                resp = self._session.get(url, params=params, timeout=req_timeout)
                if resp.status_code == 429:
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 8.0)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                ret_code = payload.get("retCode", -1)
                if ret_code == 0:
                    return payload.get("result", {})
                if ret_code in (10006, 10018):  # rate limit
                    time.sleep(backoff)
                    backoff = min(backoff * 2.0, 8.0)
                    continue
                raise RuntimeError(f"Bybit API error {ret_code}: {payload.get('retMsg')} ({path})")
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_err = exc
                if isinstance(exc, requests.RequestException):
                    try:
                        self._session.close()
                    except Exception:
                        pass
                    self._session = requests.Session()
                    self._session.headers.update({"User-Agent": "bybit-autonomous-engine/1.0"})
                if attempt < retries - 1:
                    time.sleep(backoff)
                    backoff = min(backoff * 1.8, 6.0)
        raise RuntimeError(f"Failed public GET {path} after {retries} retries: {last_err}")

    # ------------------------------------------------------------------
    # Public Market Metadata & Universe Selection
    # ------------------------------------------------------------------
    def get_linear_instruments(self) -> Dict[str, InstrumentSpec]:
        """Fetch all active linear USDT perpetual contracts."""
        specs: Dict[str, InstrumentSpec] = {}
        cursor = ""
        while True:
            params: Dict[str, Any] = {"category": "linear", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            result = self._public_get("/v5/market/instruments-info", params=params)
            for item in result.get("list", []):
                if item.get("status") != "Trading":
                    continue
                if item.get("quoteCoin") != "USDT":
                    continue
                if item.get("contractType") != "LinearPerpetual":
                    continue
                lot = item.get("lotSizeFilter", {})
                price_f = item.get("priceFilter", {})
                lev_f = item.get("leverageFilter", {})
                symbol = item["symbol"]
                specs[symbol] = InstrumentSpec(
                    symbol=symbol,
                    base_coin=item.get("baseCoin", ""),
                    quote_coin="USDT",
                    status=item.get("status", ""),
                    min_order_qty=float(lot.get("minOrderQty", 0.0) or 0.0),
                    qty_step=float(lot.get("qtyStep", 0.001) or 0.001),
                    min_notional=float(lot.get("minNotionalValue", 5.0) or 5.0),
                    tick_size=float(price_f.get("tickSize", 0.0001) or 0.0001),
                    min_leverage=float(lev_f.get("minLeverage", 1.0) or 1.0),
                    max_leverage=float(lev_f.get("maxLeverage", 50.0) or 50.0),
                    symbol_type=str(item.get("symbolType", "") or ""),
                )
            cursor = result.get("nextPageCursor", "")
            if not cursor:
                break
        return specs

    def get_tickers(self) -> Dict[str, Dict[str, float]]:
        """Fetch 24h linear tickers including turnover, price, funding rate, and OI."""
        result = self._public_get("/v5/market/tickers", params={"category": "linear"})
        tickers: Dict[str, Dict[str, float]] = {}
        for item in result.get("list", []):
            sym = item.get("symbol", "")
            if not sym:
                continue
            tickers[sym] = {
                "last_price": float(item.get("lastPrice") or 0.0),
                "price_24h_pcnt": float(item.get("price24hPcnt") or 0.0),
                "turnover_24h": float(item.get("turnover24h") or 0.0),
                "volume_24h": float(item.get("volume24h") or 0.0),
                "funding_rate": float(item.get("fundingRate") or 0.0),
                "open_interest": float(item.get("openInterest") or 0.0),
                "open_interest_value": float(item.get("openInterestValue") or 0.0),
            }
        return tickers

    def get_top_usdt_perpetuals(
        self,
        top_n: int = 100,
        max_min_order_usdt: float = 20.0,
        exclude_stablecoins: bool = True,
        crypto_only: bool = True,
    ) -> List[InstrumentSpec]:
        """Return top-N active USDT perpetuals sorted by 24h turnover passing min-order & crypto_only filters."""
        specs = self.get_linear_instruments()
        tickers = self.get_tickers()

        eligible: List[InstrumentSpec] = []
        for sym, spec in specs.items():
            t = tickers.get(sym)
            if not t or t["last_price"] <= 0 or t["turnover_24h"] <= 0:
                continue
            if crypto_only and not spec.is_pure_crypto:
                continue
            if exclude_stablecoins and spec.base_coin.upper() in STABLECOIN_BASE_COINS:
                continue
            spec.last_price = t["last_price"]
            spec.turnover_24h = t["turnover_24h"]
            spec.volume_24h = t["volume_24h"]
            spec.funding_rate = t["funding_rate"]
            spec.open_interest = t["open_interest"]
            if spec.min_order_usdt <= max_min_order_usdt:
                eligible.append(spec)

        eligible.sort(key=lambda s: s.turnover_24h, reverse=True)
        return eligible[:top_n]

    def get_instrument_specs_for_symbols(self, symbols: List[str]) -> Dict[str, InstrumentSpec]:
        """Fetch specs enriched with live ticker data for a specific list of symbols."""
        specs = self.get_linear_instruments()
        tickers = self.get_tickers()
        out: Dict[str, InstrumentSpec] = {}
        for sym in symbols:
            if sym in specs:
                spec = specs[sym]
                t = tickers.get(sym, {})
                spec.last_price = t.get("last_price", 0.0)
                spec.turnover_24h = t.get("turnover_24h", 0.0)
                spec.volume_24h = t.get("volume_24h", 0.0)
                spec.funding_rate = t.get("funding_rate", 0.0)
                spec.open_interest = t.get("open_interest", 0.0)
                out[sym] = spec
        return out

    # ------------------------------------------------------------------
    # Historical Klines with Incremental .csv.gz Caching
    # ------------------------------------------------------------------
    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol}_{interval}.csv.gz"

    def _fetch_kline_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch raw closed klines in [start_ms, end_ms] via paginated Bybit v5 API."""
        interval_ms = INTERVAL_TO_MS[str(interval)]
        rows: List[List[Any]] = []
        cursor_end = end_ms
        max_pages = int((end_ms - start_ms) / (interval_ms * 990)) + 5

        for _ in range(max_pages):
            if cursor_end < start_ms:
                break
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": str(interval),
                "start": int(start_ms),
                "end": int(cursor_end),
                "limit": 1000,
            }
            result = self._public_get("/v5/market/kline", params=params)
            batch = result.get("list", [])
            if not batch:
                break
            rows.extend(batch)
            # Bybit returns newest first; oldest item is batch[-1]
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
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    def get_klines(
        self,
        symbol: str,
        interval: str = "15",
        days: Optional[int] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        use_cache: bool = True,
        closed_only: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch klines with incremental .csv.gz cache in data/cache/.
        Downloads ONLY missing segments before cached start or after cached end.
        """
        interval_str = str(interval)
        if interval_str not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval: {interval_str}")
        interval_ms = INTERVAL_TO_MS[interval_str]

        now_ms = int(time.time() * 1000)
        last_closed_open_ms = ((now_ms // interval_ms) - 1) * interval_ms

        req_end = min(end_ms, last_closed_open_ms) if end_ms is not None else last_closed_open_ms
        if start_ms is not None:
            req_start = int(start_ms)
        elif days is not None:
            req_start = req_end - int(days * 86_400_000)
        else:
            req_start = req_end - int(3 * 86_400_000)

        cache_file = self._cache_path(symbol, interval_str)
        cached_df: Optional[pd.DataFrame] = None

        if use_cache and cache_file.exists():
            try:
                cached_df = pd.read_csv(cache_file, compression="gzip")
                if not cached_df.empty:
                    cached_df["timestamp"] = cached_df["timestamp"].astype("int64")
                    for col in ("open", "high", "low", "close", "volume", "turnover"):
                        cached_df[col] = cached_df[col].astype("float64")
                    cached_df = cached_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            except Exception as exc:
                logger.warning("Corrupted cache %s (%s), refetching.", cache_file, exc)
                cached_df = None

        frames: List[pd.DataFrame] = []
        updated = False

        if cached_df is None or cached_df.empty:
            fresh = self._fetch_kline_range(symbol, interval_str, req_start, req_end)
            if not fresh.empty:
                frames.append(fresh)
                updated = True
        else:
            frames.append(cached_df)
            cached_min = int(cached_df["timestamp"].iloc[0])
            cached_max = int(cached_df["timestamp"].iloc[-1])

            # Fetch missing older history (only if gap > 2 bars)
            if req_start < cached_min - 2 * interval_ms:
                older = self._fetch_kline_range(
                    symbol, interval_str, req_start, cached_min - interval_ms
                )
                if not older.empty:
                    frames.append(older)
                    updated = True

            # Fetch missing newer candles; the last cached bar is re-fetched in case
            # it was stored right at its close, before the exchange finalized it
            if req_end >= cached_max + interval_ms:
                newer = self._fetch_kline_range(symbol, interval_str, cached_max, req_end)
                if not newer.empty:
                    frames.append(newer)
                    updated = True

        if not frames:
            return pd.DataFrame(columns=KLINE_COLUMNS)

        merged = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if closed_only:
            merged = merged[merged["timestamp"] + interval_ms <= now_ms].reset_index(drop=True)

        if use_cache and updated and not merged.empty:
            tmp_file = cache_file.with_name(f"{cache_file.stem}.{os.getpid()}.tmp.gz")
            try:
                merged.to_csv(tmp_file, index=False, compression="gzip")
                for attempt in range(3):
                    try:
                        os.replace(tmp_file, cache_file)
                        break
                    except PermissionError:
                        if attempt < 2:
                            time.sleep(0.15 * (attempt + 1))
                        else:
                            logger.debug("Cache file %s busy, skipping cache write.", cache_file)
            except Exception as exc:
                logger.debug("Cache write skipped for %s: %s", cache_file, exc)
            finally:
                try:
                    if tmp_file.exists():
                        tmp_file.unlink()
                except OSError:
                    pass

        filtered = merged[(merged["timestamp"] >= req_start) & (merged["timestamp"] <= req_end)].copy()
        return filtered.reset_index(drop=True)

    @staticmethod
    def resample_klines(df_15m: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """Resample UTC-aligned 15m klines into exact higher timeframe candles ('60', '240', 'D')."""
        if df_15m.empty:
            return pd.DataFrame(columns=KLINE_COLUMNS)
        rule_map = {"60": "1h", "120": "2h", "240": "4h", "D": "1D"}
        rule = rule_map.get(str(target_interval))
        if rule is None:
            raise ValueError(f"Unsupported resample target: {target_interval}")

        tmp = df_15m.copy()
        tmp["dt"] = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)
        tmp = tmp.set_index("dt")
        agg = tmp.resample(rule, origin="epoch", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "turnover": "sum",
            }
        ).dropna(subset=["open", "close"])
        agg["timestamp"] = agg.index.as_unit("ms").asi8.astype("int64")
        return agg[KLINE_COLUMNS].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public Open Interest & Funding Rate History (graceful fallback)
    # ------------------------------------------------------------------
    def get_open_interest(self, symbol: str, interval_time: str = "15min", limit: int = 30) -> Optional[pd.DataFrame]:
        """Fetch recent Open Interest series; returns None gracefully if unavailable."""
        try:
            result = self._public_get(
                "/v5/market/open-interest",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "intervalTime": interval_time,
                    "limit": min(limit, 200),
                },
                retries=2,
            )
            items = result.get("list", [])
            if not items:
                return None
            df = pd.DataFrame(items)
            df["timestamp"] = df["timestamp"].astype("int64")
            df["openInterest"] = df["openInterest"].astype("float64")
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            return None

    def get_funding_history(
        self,
        symbol: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 200,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical funding rates (8h intervals); returns None gracefully on error."""
        try:
            params: Dict[str, Any] = {
                "category": "linear",
                "symbol": symbol,
                "limit": min(limit, 200),
            }
            if start_ms is not None:
                params["startTime"] = int(start_ms)
            if end_ms is not None:
                params["endTime"] = int(end_ms)
            result = self._public_get("/v5/market/funding/history", params=params, retries=2)
            items = result.get("list", [])
            if not items:
                return None
            df = pd.DataFrame(items)
            df["timestamp"] = df["fundingRateTimestamp"].astype("int64")
            df["fundingRate"] = df["fundingRate"].astype("float64")
            return df[["timestamp", "fundingRate"]].sort_values("timestamp").reset_index(drop=True)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Private Authenticated Execution Endpoints (pybit v5)
    # ------------------------------------------------------------------
    def _require_private(self) -> Any:
        if self._private_client is None:
            raise RuntimeError(
                "Private Bybit client unavailable. Configure BYBIT_API_KEY and BYBIT_API_SECRET in .env."
            )
        return self._private_client

    def get_wallet_balance(self, coin: str = "USDT") -> Dict[str, float]:
        client = self._require_private()
        resp = client.get_wallet_balance(accountType="UNIFIED", coin=coin)
        if resp.get("retCode") != 0:
            raise RuntimeError(f"get_wallet_balance error: {resp}")
        acct = resp["result"]["list"][0]
        equity = float(acct.get("totalEquity") or 0.0)
        avail = float(acct.get("totalAvailableBalance") or 0.0)
        for c in acct.get("coin", []):
            if c.get("coin") == coin:
                equity = float(c.get("equity") or c.get("walletBalance") or equity)
                wb = float(c.get("walletBalance") or equity)
                locked = float(c.get("totalOrderIM") or 0.0) + float(c.get("totalPositionIM") or 0.0)
                c_avail = float(c.get("availableToWithdraw") or 0.0)
                avail = c_avail if c_avail > 0 else (avail if avail > 0 else max(0.0, wb - locked))
                break
        return {"equity": equity, "available": avail}

    def set_leverage(self, symbol: str, leverage: int) -> None:
        client = self._require_private()
        lev_str = str(int(leverage))
        try:
            resp = client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=lev_str,
                sellLeverage=lev_str,
            )
            if resp.get("retCode") not in (0, 110043):  # 110043 = leverage not modified
                raise RuntimeError(f"set_leverage failed for {symbol}: {resp}")
        except Exception as exc:
            if "110043" not in str(exc):
                raise

    def ensure_isolated_margin_and_leverage(self, symbol: str, leverage: int = 10) -> None:
        """Force Isolated Margin mode (tradeMode=1) and exact leverage (10x) via Bybit v5 API."""
        client = self._require_private()
        lev_str = str(int(leverage))
        # 1. Force Isolated Margin mode on account/symbol (ignore 'already isolated' / UTA codes)
        try:
            resp = client.switch_margin_mode(
                category="linear",
                symbol=symbol,
                tradeMode=1,  # 1 = Isolated margin
                buyLeverage=lev_str,
                sellLeverage=lev_str,
            )
            if resp.get("retCode") not in (0, 110026, 100028):
                logger.warning("switch_margin_mode non-zero for %s: %s", symbol, resp)
        except Exception as exc:
            err = str(exc)
            if "110026" not in err and "100028" not in err:
                # UTA accounts use set_margin_mode(setMarginMode="ISOLATED_MARGIN") at account level
                try:
                    client.set_margin_mode(setMarginMode="ISOLATED_MARGIN")
                except Exception as exc2:
                    logger.warning("Isolated margin setup note for %s: %s / %s", symbol, exc, exc2)

        # 2. Enforce exact leverage
        self.set_leverage(symbol, leverage)

    def place_limit_entry_with_sl(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        stop_loss: float,
    ) -> Dict[str, Any]:
        """Place a GTC Limit entry order with atomic server-side Stop-Loss attached."""
        client = self._require_private()
        resp = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=str(qty),
            price=str(price),
            stopLoss=str(stop_loss),
            slTriggerBy="MarkPrice",
            timeInForce="GTC",
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"place_limit_entry_with_sl error: {resp}")
        return resp["result"]

    def place_market_order_with_sl(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_loss: float,
    ) -> Dict[str, Any]:
        """Atomically open a position with a server-side Stop-Loss attached."""
        client = self._require_private()
        resp = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            stopLoss=str(stop_loss),
            slTriggerBy="MarkPrice",
            timeInForce="IOC",
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"place_market_order_with_sl error: {resp}")
        return resp["result"]

    def place_limit_tp_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> Dict[str, Any]:
        """Place a reduce-only GTC limit take-profit order."""
        client = self._require_private()
        resp = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Limit",
            qty=str(qty),
            price=str(price),
            reduceOnly=True,
            timeInForce="GTC",
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"place_limit_tp_order error: {resp}")
        return resp["result"]

    def close_position_market(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Dict[str, Any]:
        """Force-close an open position at market (reduceOnly IOC) for H1-flip / pre-news shield."""
        client = self._require_private()
        resp = client.place_order(
            category="linear",
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
            timeInForce="IOC",
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"close_position_market error: {resp}")
        return resp["result"]

    def set_trading_stop(self, symbol: str, stop_loss: float, position_idx: int = 0) -> Dict[str, Any]:
        """Update server-side stop-loss (e.g., move to breakeven +0.15%)."""
        client = self._require_private()
        resp = client.set_trading_stop(
            category="linear",
            symbol=symbol,
            stopLoss=str(stop_loss),
            slTriggerBy="MarkPrice",
            positionIdx=position_idx,
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"set_trading_stop error: {resp}")
        return resp["result"]

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        client = self._require_private()
        params: Dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = client.get_positions(**params)
        if resp.get("retCode") != 0:
            raise RuntimeError(f"get_positions error: {resp}")
        return [p for p in resp["result"].get("list", []) if float(p.get("size") or 0) > 0]

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        client = self._require_private()
        params: Dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            params["symbol"] = symbol
        resp = client.get_open_orders(**params)
        if resp.get("retCode") != 0:
            raise RuntimeError(f"get_open_orders error: {resp}")
        return resp["result"].get("list", [])

    def get_order_status(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        client = self._require_private()
        for method in (client.get_open_orders, client.get_order_history):
            resp = method(category="linear", symbol=symbol, orderId=order_id)
            if resp.get("retCode") == 0:
                items = resp["result"].get("list", [])
                if items:
                    return items[0]
        return None

    def get_closed_pnl(self, symbol: str, start_ms: int, limit: int = 50) -> List[Dict[str, Any]]:
        client = self._require_private()
        resp = client.get_closed_pnl(category="linear", symbol=symbol, startTime=int(start_ms), limit=limit)
        if resp.get("retCode") != 0:
            raise RuntimeError(f"get_closed_pnl error: {resp}")
        return resp["result"].get("list", [])

    def cancel_order(self, symbol: str, order_id: str) -> None:
        client = self._require_private()
        try:
            client.cancel_order(category="linear", symbol=symbol, orderId=order_id)
        except Exception as exc:
            if "110001" not in str(exc):  # 110001 = order not exists / already filled
                logger.warning("cancel_order failed for %s (%s): %s", symbol, order_id, exc)

    def cancel_all_orders(self, symbol: str) -> None:
        client = self._require_private()
        client.cancel_all_orders(category="linear", symbol=symbol)
