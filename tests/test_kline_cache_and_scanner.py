import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from bot.bybit_client import BybitClient, INTERVAL_TO_MS, InstrumentSpec
from bot.news_guard import NewsGuard
from bot.risk_manager import RiskManager
from bot.scanner import MarketScanner


def _make_df(timestamps_ms, base_price=100.0):
    rows = []
    for i, ts in enumerate(timestamps_ms):
        p = base_price + i * 0.1
        rows.append(
            {
                "timestamp": int(ts),
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p + 0.1,
                "volume": 1000.0,
                "turnover": 100000.0,
            }
        )
    return pd.DataFrame(rows)


class TestKlineCacheAndScanner(unittest.TestCase):
    def test_kline_cache_refetches_when_missing_exactly_one_closed_bar(self):
        interval_ms = INTERVAL_TO_MS["15"]
        now_ms = 1_700_000_000 * 1000
        # Align to exact 15m boundary + 10s
        now_ms = (now_ms // interval_ms) * interval_ms + 10_000
        req_end = ((now_ms // interval_ms) - 1) * interval_ms  # latest closed bar open ts
        cached_max = req_end - interval_ms  # missing exactly 1 closed bar

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cached_bars = [cached_max - i * interval_ms for i in range(110, -1, -1)]
            cached_df = _make_df(cached_bars)
            cached_df.to_csv(cache_dir / "ADAUSDT_15.csv.gz", index=False, compression="gzip")

            client = BybitClient(mode="demo", cache_dir=cache_dir, api_key="", api_secret="")
            fetched_df = _make_df([cached_max, req_end], base_price=200.0)

            with (
                patch("bot.bybit_client.time.time", return_value=now_ms / 1000.0),
                patch.object(client, "_fetch_kline_range", return_value=fetched_df) as mock_fetch,
            ):
                res = client.get_klines("ADAUSDT", interval="15", days=1, use_cache=True, closed_only=True)

            mock_fetch.assert_called_once()
            self.assertEqual(int(res["timestamp"].iloc[-1]), req_end)
            self.assertEqual(float(res["close"].iloc[-1]), 200.2)

    def test_scanner_marks_incomplete_when_btc_bar_is_stale(self):
        interval_ms = INTERVAL_TO_MS["15"]
        now_sec = 1_700_001_000.0
        now_ms = int(now_sec * 1000)
        expected_last_ts = ((now_ms // interval_ms) - 1) * interval_ms
        stale_ts = expected_last_ts - interval_ms

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ng = NewsGuard(
                config={"macro_shield_enabled": False},
                state_path=tmp_path / "news_state.json",
                macro_events_path=tmp_path / "macro_events.json",
            )
            mock_client = MagicMock()
            btc_bars = [stale_ts - i * interval_ms for i in range(35, -1, -1)]
            mock_client.get_klines.return_value = _make_df(btc_bars)

            cfg = {
                "strategy": "ada_fibo50_15x",
                "whitelist": ["ADAUSDT"],
                "allowed_directions": "long_short",
                "entry_tf": "15",
                "context_tf": "60",
                "trend_tf": "240",
            }
            scanner = MarketScanner(
                client=mock_client,
                risk_manager=RiskManager(cfg),
                config=cfg,
                blacklist_path=tmp_path / "blacklist.json",
                news_state_path=tmp_path / "news_state.json",
                universe_cache_path=tmp_path / "universe.json",
                news_guard=ng,
            )

            with patch("bot.scanner.time.time", return_value=now_sec):
                signals = scanner.scan_once(equity=2000.0)

            self.assertEqual(signals, [])
            self.assertTrue(scanner.last_scan_incomplete)

    def test_scanner_marks_incomplete_when_coin_bar_is_stale(self):
        interval_ms = INTERVAL_TO_MS["15"]
        now_sec = 1_700_001_000.0
        now_ms = int(now_sec * 1000)
        expected_last_ts = ((now_ms // interval_ms) - 1) * interval_ms
        stale_ts = expected_last_ts - interval_ms

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ng = NewsGuard(
                config={"macro_shield_enabled": False},
                state_path=tmp_path / "news_state.json",
                macro_events_path=tmp_path / "macro_events.json",
            )
            btc_df = _make_df([expected_last_ts - i * interval_ms for i in range(35, -1, -1)])
            ada_df = _make_df([stale_ts - i * interval_ms for i in range(130, -1, -1)])

            mock_client = MagicMock()
            mock_client.get_klines.side_effect = lambda sym, **kw: btc_df if sym == "BTCUSDT" else ada_df
            spec = InstrumentSpec(
                symbol="ADAUSDT",
                base_coin="ADA",
                quote_coin="USDT",
                status="Trading",
                min_order_qty=1.0,
                qty_step=1.0,
                min_notional=5.0,
                tick_size=0.0001,
                min_leverage=1.0,
                max_leverage=50.0,
            )

            cfg = {
                "strategy": "ada_fibo50_15x",
                "whitelist": ["ADAUSDT"],
                "allowed_directions": "long_short",
                "entry_tf": "15",
                "context_tf": "60",
                "trend_tf": "240",
            }
            scanner = MarketScanner(
                client=mock_client,
                risk_manager=RiskManager(cfg),
                config=cfg,
                blacklist_path=tmp_path / "blacklist.json",
                news_state_path=tmp_path / "news_state.json",
                universe_cache_path=tmp_path / "universe.json",
                news_guard=ng,
            )
            with (
                patch("bot.scanner.time.time", return_value=now_sec),
                patch.object(scanner, "get_universe", return_value=[spec]),
            ):
                signals = scanner.scan_once(equity=2000.0)

            self.assertEqual(signals, [])
            self.assertTrue(scanner.last_scan_incomplete)

    def test_crv_stoch_10x_generates_limit_signal_in_uptrend_pullback(self):
        interval_ms = INTERVAL_TO_MS["15"]
        now_sec = 1_700_005_000.0
        expected_last_ts = ((int(now_sec * 1000) // interval_ms) - 1) * interval_ms

        n = 280
        ts_list = [expected_last_ts - (n - 1 - i) * interval_ms for i in range(n)]
        rows = []
        for i, ts in enumerate(ts_list):
            if i < n - 3:
                p = 0.45 + i * 0.00025  # steady 1H uptrend within 5% of EMA200
                hi, lo, cl = p + 0.0035, p - 0.0035, p + 0.0004
            else:
                # 15M dip keeps close above EMA20 - 0.70*ATR while Stochastic(14) <= 20
                p = 0.5175 - (i - (n - 3)) * 0.0004
                hi, lo, cl = p + 0.0035, p - 0.0038, p - 0.0023
            rows.append(
                {
                    "timestamp": int(ts),
                    "open": p,
                    "high": hi,
                    "low": lo,
                    "close": cl,
                    "volume": 50000.0,
                    "turnover": 25000.0,
                }
            )
        crv_df = pd.DataFrame(rows)
        btc_df = _make_df([expected_last_ts - i * interval_ms for i in range(40, -1, -1)], base_price=60000.0)

        spec = InstrumentSpec(
            symbol="CRVUSDT",
            base_coin="CRV",
            quote_coin="USDT",
            status="Trading",
            min_order_qty=1.0,
            qty_step=1.0,
            min_notional=5.0,
            tick_size=0.0001,
            min_leverage=1.0,
            max_leverage=25.0,
        )
        cfg = {
            "strategy": "crv_stoch_10x",
            "whitelist": ["CRVUSDT"],
            "allowed_directions": "long_short",
            "entry_tf": "15",
            "context_tf": "60",
            "trend_tf": "240",
            "crv_ema_period": 20,
            "crv_stoch_period": 14,
            "crv_stoch_long_max": 20.0,
            "crv_stoch_short_min": 80.0,
            "crv_pullback_atr": 0.7,
            "min_atr_pct": 0.1,
            "max_atr_pct": 1.8,
            "h1_adx_min": 14.0,
            "max_dist_ema200_pct": 5.0,
            "btc_1h_veto_pct": 1.0,
            "tp_pct": 1.5,
            "sl_pct": 2.5,
            "smart_be1_trigger_pct": 0.825,
            "smart_be1_offset_pct": 0.27,
            "smart_be2_trigger_pct": 1.17,
            "smart_be2_offset_pct": 0.57,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ng = NewsGuard(
                config={"macro_shield_enabled": False},
                state_path=tmp_path / "news_state.json",
                macro_events_path=tmp_path / "macro_events.json",
            )
            mock_client = MagicMock()
            mock_client.get_klines.side_effect = lambda sym, **kw: btc_df if sym == "BTCUSDT" else crv_df
            scanner = MarketScanner(
                client=mock_client,
                risk_manager=RiskManager(cfg),
                config=cfg,
                blacklist_path=tmp_path / "blacklist.json",
                news_state_path=tmp_path / "news_state.json",
                universe_cache_path=tmp_path / "universe.json",
                news_guard=ng,
            )
            with (
                patch("bot.scanner.time.time", return_value=now_sec),
                patch.object(scanner, "get_universe", return_value=[spec]),
            ):
                signals = scanner.scan_once(equity=2000.0)

        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.symbol, "CRVUSDT")
        self.assertEqual(sig.direction, "LONG")
        self.assertLess(sig.entry_price, float(crv_df["close"].iloc[-1]))
        self.assertEqual(sig.tp_pct, 1.5)
        self.assertEqual(sig.sl_pct, 2.5)

    def _build_btc_4h_df(self, n: int, h4_ms: int, last_closed_ts: int, bullish: bool, include_unclosed_spike: bool = False) -> pd.DataFrame:
        rows = []
        for i in range(n):
            ts = last_closed_ts - (n - 1 - i) * h4_ms
            if bullish:
                p = 80000.0 + i * 120.0
            else:
                p = 95000.0 - i * 120.0
            rows.append({
                "timestamp": ts,
                "open": p - 50.0,
                "high": p + 150.0,
                "low": p - 150.0,
                "close": p,
                "volume": 1000.0,
                "turnover": 1000.0 * p,
            })
        if include_unclosed_spike:
            rows.append({
                "timestamp": last_closed_ts + h4_ms,  # unclosed current 4H bar
                "open": 81000.0,
                "high": 120000.0,
                "low": 80500.0,
                "close": 115000.0,
                "volume": 5000.0,
                "turnover": 5000.0 * 115000.0,
            })
        return pd.DataFrame(rows)

    def test_btc_4h_e4_uses_strictly_closed_candle_and_blocks_long_allows_short(self):
        h4_ms = INTERVAL_TO_MS["240"]
        now_sec = 1_700_006_400.0  # exact 4H boundary + 0s -> last closed 4H bar is now - 4h
        now_ms = int(now_sec * 1000)
        last_closed_4h_ts = (now_ms // h4_ms - 1) * h4_ms
        btc_4h_bear_df = self._build_btc_4h_df(
            n=100, h4_ms=h4_ms, last_closed_ts=last_closed_4h_ts, bullish=False, include_unclosed_spike=True
        )
        cfg = {
            "strategy": "crv_stoch_10x",
            "whitelist": ["CRVUSDT"],
            "allowed_directions": "long_short",
            "entry_tf": "15",
            "context_tf": "60",
            "trend_tf": "240",
            "btc_4h_bear_block_long": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mock_client = MagicMock()
            mock_client.get_klines.return_value = btc_4h_bear_df
            scanner = MarketScanner(
                client=mock_client,
                risk_manager=RiskManager(cfg),
                config=cfg,
                blacklist_path=tmp_path / "blacklist.json",
                news_state_path=tmp_path / "news_state.json",
                universe_cache_path=tmp_path / "universe.json",
            )
            with patch("bot.scanner.time.time", return_value=now_sec):
                res = scanner._eval_btc_4h_e4_filter(trend_tf="240")
        self.assertTrue(res["enabled"])
        self.assertTrue(res["block_long"])
        self.assertEqual(res["bar_ts"], last_closed_4h_ts)

        # Verify scan_once blocks LONG signal when BTC 4H is bearish, while SHORT direction remains allowed
        interval_15_ms = INTERVAL_TO_MS["15"]
        expected_last_15_ts = (now_ms // interval_15_ms - 1) * interval_15_ms
        btc_15_df = _make_df([expected_last_15_ts - i * interval_15_ms for i in range(35, -1, -1)])
        n = 280
        rows = []
        for i in range(n):
            ts = expected_last_15_ts - (n - 1 - i) * interval_15_ms
            if i < n - 3:
                p = 0.45 + i * 0.00025
                hi, lo, cl = p + 0.0035, p - 0.0035, p + 0.0004
            else:
                p = 0.5175 - (i - (n - 3)) * 0.0004
                hi, lo, cl = p + 0.0035, p - 0.0038, p - 0.0023
            rows.append({"timestamp": ts, "open": p, "high": hi, "low": lo, "close": cl, "volume": 50000.0, "turnover": 25000.0})
        crv_df = pd.DataFrame(rows)
        spec = InstrumentSpec("CRVUSDT", "CRV", "USDT", 0.0001, 0.1, 0.1, 5.0, 0.51, 1.0, 25.0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ng = NewsGuard(config={"macro_shield_enabled": False}, state_path=tmp_path / "ns.json", macro_events_path=tmp_path / "me.json")
            mock_client = MagicMock()
            def _gk(sym, **kw):
                if sym == "BTCUSDT" and str(kw.get("interval")) == "240":
                    return btc_4h_bear_df
                return btc_15_df if sym == "BTCUSDT" else crv_df
            mock_client.get_klines.side_effect = _gk
            scanner = MarketScanner(
                client=mock_client,
                risk_manager=RiskManager(cfg),
                config=cfg,
                blacklist_path=tmp_path / "bl.json",
                news_state_path=tmp_path / "ns.json",
                universe_cache_path=tmp_path / "u.json",
                news_guard=ng,
            )
            with patch("bot.scanner.time.time", return_value=now_sec), patch.object(scanner, "get_universe", return_value=[spec]):
                signals = scanner.scan_once(equity=2000.0)
        self.assertEqual(signals, [])

    def test_btc_4h_e4_allows_long_when_btc_above_ema50(self):
        h4_ms = INTERVAL_TO_MS["240"]
        interval_15_ms = INTERVAL_TO_MS["15"]
        now_sec = 1_700_005_000.0
        now_ms = int(now_sec * 1000)
        last_closed_4h_ts = (now_ms // h4_ms - 1) * h4_ms
        expected_last_15_ts = (now_ms // interval_15_ms - 1) * interval_15_ms
        btc_4h_bull_df = self._build_btc_4h_df(n=100, h4_ms=h4_ms, last_closed_ts=last_closed_4h_ts, bullish=True)
        btc_15_df = _make_df([expected_last_15_ts - i * interval_15_ms for i in range(35, -1, -1)])
        n = 280
        rows = []
        for i in range(n):
            ts = expected_last_15_ts - (n - 1 - i) * interval_15_ms
            if i < n - 3:
                p = 0.45 + i * 0.00025
                hi, lo, cl = p + 0.0035, p - 0.0035, p + 0.0004
            else:
                p = 0.5175 - (i - (n - 3)) * 0.0004
                hi, lo, cl = p + 0.0035, p - 0.0038, p - 0.0023
            rows.append({"timestamp": ts, "open": p, "high": hi, "low": lo, "close": cl, "volume": 50000.0, "turnover": 25000.0})
        crv_df = pd.DataFrame(rows)
        spec = InstrumentSpec("CRVUSDT", "CRV", "USDT", 0.0001, 0.1, 0.1, 5.0, 0.51, 1.0, 25.0)
        cfg = {
            "strategy": "crv_stoch_10x",
            "whitelist": ["CRVUSDT"],
            "allowed_directions": "long_short",
            "entry_tf": "15",
            "context_tf": "60",
            "trend_tf": "240",
            "btc_4h_bear_block_long": True,
            "tp_pct": 1.5,
            "sl_pct": 2.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ng = NewsGuard(config={"macro_shield_enabled": False}, state_path=tmp_path / "ns.json", macro_events_path=tmp_path / "me.json")
            mock_client = MagicMock()
            mock_client.get_klines.side_effect = lambda sym, **kw: (
                btc_4h_bull_df if (sym == "BTCUSDT" and str(kw.get("interval")) == "240")
                else (btc_15_df if sym == "BTCUSDT" else crv_df)
            )
            scanner = MarketScanner(
                client=mock_client,
                risk_manager=RiskManager(cfg),
                config=cfg,
                blacklist_path=tmp_path / "bl.json",
                news_state_path=tmp_path / "ns.json",
                universe_cache_path=tmp_path / "u.json",
                news_guard=ng,
            )
            with patch("bot.scanner.time.time", return_value=now_sec), patch.object(scanner, "get_universe", return_value=[spec]):
                signals = scanner.scan_once(equity=2000.0)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].direction, "LONG")

    def test_btc_4h_e4_disabled_for_wlfi_and_fail_closed_on_api_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mock_client = MagicMock()
            mock_client.get_klines.side_effect = RuntimeError("API timeout")

            cfg_wlfi = {"strategy": "wlfi_meanrev_15x_v2a", "whitelist": ["WLFIUSDT"], "btc_4h_bear_block_long": False}
            sc_wlfi = MarketScanner(
                client=mock_client, risk_manager=RiskManager(cfg_wlfi), config=cfg_wlfi,
                blacklist_path=tmp_path / "bl.json", news_state_path=tmp_path / "ns.json", universe_cache_path=tmp_path / "u.json",
            )
            res_wlfi = sc_wlfi._eval_btc_4h_e4_filter(trend_tf="240")
            self.assertFalse(res_wlfi["enabled"])
            self.assertFalse(res_wlfi["block_long"])
            mock_client.get_klines.assert_not_called()

            cfg_ada = {"strategy": "ada_fibo50_15x", "whitelist": ["ADAUSDT"], "btc_4h_bear_block_long": True}
            sc_ada = MarketScanner(
                client=mock_client, risk_manager=RiskManager(cfg_ada), config=cfg_ada,
                blacklist_path=tmp_path / "bl.json", news_state_path=tmp_path / "ns.json", universe_cache_path=tmp_path / "u.json",
            )
            res_ada = sc_ada._eval_btc_4h_e4_filter(trend_tf="240")
            self.assertTrue(res_ada["enabled"])
            self.assertTrue(res_ada["block_long"])

    def test_hbar_macd_stoch_10x_generates_limit_signal(self):
        interval_ms = INTERVAL_TO_MS["15"]
        now_sec = 1_700_005_000.0
        expected_last_ts = (int(now_sec * 1000) // interval_ms - 1) * interval_ms
        btc_df = _make_df([expected_last_ts - i * interval_ms for i in range(35, -1, -1)])
        n = 280
        rows = []
        for i in range(n):
            ts = expected_last_ts - (n - 1 - i) * interval_ms
            if i < n - 8:
                p = 0.10 + i * 0.00015
                hi, lo, cl = p + 0.0008, p - 0.0008, p + 0.0002
            elif i < n - 1:
                p = 0.1408 - (i - (n - 8)) * 0.00055
                hi, lo, cl = p + 0.0006, p - 0.0009, p - 0.0004
            else:
                p = 0.1380
                hi, lo, cl = p + 0.0014, p - 0.0004, p + 0.0012
            rows.append({"timestamp": ts, "open": p, "high": hi, "low": lo, "close": cl, "volume": 50000.0, "turnover": 6000.0})
        hbar_df = pd.DataFrame(rows)
        spec = InstrumentSpec("HBARUSDT", "HBAR", "USDT", 0.00001, 1.0, 1.0, 5.0, 0.12, 1.0, 25.0)
        cfg = {
            "strategy": "hbar_macd_stoch_10x",
            "whitelist": ["HBARUSDT"],
            "allowed_directions": "long_short",
            "entry_tf": "15",
            "context_tf": "60",
            "trend_tf": "240",
            "btc_4h_bear_block_long": False,
            "tp_pct": 1.4,
            "sl_pct": 2.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ng = NewsGuard(config={"macro_shield_enabled": False}, state_path=tmp_path / "ns.json", macro_events_path=tmp_path / "me.json")
            mock_client = MagicMock()
            mock_client.get_klines.side_effect = lambda sym, **kw: btc_df if sym == "BTCUSDT" else hbar_df
            scanner = MarketScanner(
                client=mock_client,
                risk_manager=RiskManager(cfg),
                config=cfg,
                blacklist_path=tmp_path / "bl.json",
                news_state_path=tmp_path / "ns.json",
                universe_cache_path=tmp_path / "u.json",
                news_guard=ng,
            )
            with patch("bot.scanner.time.time", return_value=now_sec), patch.object(scanner, "get_universe", return_value=[spec]):
                signals = scanner.scan_once(equity=2000.0)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "HBARUSDT")
        self.assertEqual(signals[0].direction, "LONG")
        self.assertEqual(signals[0].tp_pct, 1.4)
        self.assertEqual(signals[0].sl_pct, 2.5)


if __name__ == "__main__":
    unittest.main()
