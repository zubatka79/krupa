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


if __name__ == "__main__":
    unittest.main()
