import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from bot.bybit_client import InstrumentSpec
from bot.executor import TradeExecutor
from bot.news_guard import NewsGuard, to_iso_z
from bot.risk_manager import RiskManager
from bot.scanner import TradeSignal


def _base_config():
    return {
        "mode": "demo",
        "strategy": "ada_fibo50_15x",
        "leverage": 10,
        "min_leverage": 10,
        "max_leverage": 10,
        "margin_pct_per_trade": 15.0,
        "max_open_positions": 1,
        "max_same_direction": 1,
        "max_portfolio_open_positions": 4,
        "max_portfolio_margin_pct": 60.0,
        "max_daily_stops": 3,
        "daily_loss_limit_pct": 18.0,
        "maker_ttl_bars": 4,
        "tp_pct": 1.2,
        "sl_pct": 2.5,
        "smart_be1_trigger_pct": 0.8,
        "smart_be1_offset_pct": 0.3,
        "smart_be2_trigger_pct": 0.95,
        "smart_be2_offset_pct": 0.5,
        "exit_on_h1_flip": False,
        "macro_shield_enabled": True,
        "macro_pre_window_min": 30,
        "macro_post_window_min": 15,
        "maker_fee_pct": 0.02,
        "taker_fee_pct": 0.10,
    }


def _make_signal(symbol="ADAUSDT", direction="LONG", entry=0.50):
    spec = InstrumentSpec(
        symbol=symbol,
        base_coin=symbol.replace("USDT", ""),
        quote_coin="USDT",
        status="Trading",
        min_order_qty=1.0,
        qty_step=1.0,
        min_notional=5.0,
        tick_size=0.0001,
        min_leverage=1.0,
        max_leverage=50.0,
    )
    return TradeSignal(
        symbol=symbol,
        direction=direction,
        timestamp=1_700_000_000_000,
        entry_price=entry,
        sl_pct=2.5,
        score=80.0,
        spec=spec,
    )


def _write_fresh_macro(macro_p: Path, now: datetime, event_dt: datetime | None = None):
    events = []
    if event_dt is not None:
        events.append(
            {
                "title": "Core PCE Price Index m/m",
                "country": "USD",
                "impact": "High",
                "time_utc": to_iso_z(event_dt),
                "is_core_us_macro": True,
                "forecast": "0.3%",
                "previous": "0.3%",
            }
        )
    macro_p.write_text(
        json.dumps(
            {
                "fetched_at_utc": to_iso_z(now),
                "coverage_from_utc": to_iso_z(now - timedelta(days=2)),
                "coverage_to_utc": to_iso_z(now + timedelta(days=5)),
                "status": "OK",
                "events": events,
            }
        ),
        encoding="utf-8",
    )


class TestExecutor(unittest.TestCase):
    def _build_executor(self, tmp_path: Path, client_mock: MagicMock, dry_run: bool = False):
        cfg = _base_config()
        ng = NewsGuard(
            config=cfg,
            state_path=tmp_path / "news_state.json",
            macro_events_path=tmp_path / "macro_events.json",
        )
        return (
            TradeExecutor(
                client=client_mock,
                risk_manager=RiskManager(cfg),
                config=cfg,
                db_path=tmp_path / "trades.db",
                news_state_path=tmp_path / "news_state.json",
                dry_run=dry_run,
                news_guard=ng,
            ),
            tmp_path / "macro_events.json",
        )

    def test_blocks_entry_when_ttl_overlaps_macro_event(self):
        now = datetime(2026, 11, 25, 12, 15, tzinfo=timezone.utc)
        ev_dt = now + timedelta(minutes=75)  # < 30 + 4*15 = 90 min
        with tempfile.TemporaryDirectory() as tmp:
            cl = MagicMock(has_credentials=True)
            ex, macro_p = self._build_executor(Path(tmp), cl, dry_run=False)
            _write_fresh_macro(macro_p, now, event_dt=ev_dt)
            tid = ex.open_position(_make_signal(), current_equity=2000.0, now=now, allow_network_refresh=False)
            self.assertIsNone(tid)
            cl.place_limit_entry_with_sl.assert_not_called()

    def test_portfolio_margin_cap_and_fail_closed_on_api_error(self):
        now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            cl = MagicMock(has_credentials=True)
            ex, macro_p = self._build_executor(Path(tmp), cl, dry_run=False)
            _write_fresh_macro(macro_p, now, event_dt=None)

            # Fail-closed when get_positions raises
            cl.get_positions.side_effect = RuntimeError("timeout")
            self.assertIsNone(ex.open_position(_make_signal(), current_equity=2000.0, now=now, allow_network_refresh=False))

            # Blocked when account-wide IM >= 60% of equity (e.g. $1,200 out of $2,000)
            cl.get_positions.side_effect = None
            cl.get_positions.return_value = [{"symbol": "LTCUSDT", "size": "10", "positionIM": "950.0"}]
            cl.get_open_orders.return_value = [{"symbol": "MNTUSDT", "orderIM": "300.0", "reduceOnly": False}]
            self.assertIsNone(ex.open_position(_make_signal(), current_equity=2000.0, now=now, allow_network_refresh=False))
            cl.place_limit_entry_with_sl.assert_not_called()

    def test_partial_fill_cancels_remainder_places_tp_and_recalculates_margin(self):
        now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            cl = MagicMock(has_credentials=True)
            cl.get_positions.return_value = []
            cl.get_open_orders.return_value = []
            cl.place_limit_entry_with_sl.return_value = {"orderId": "ORD-ENTRY-1"}
            cl.get_wallet_balance.return_value = {"equity": 2000.0, "available": 2000.0}
            ex, macro_p = self._build_executor(Path(tmp), cl, dry_run=False)
            _write_fresh_macro(macro_p, now, event_dt=None)

            tid = ex.open_position(_make_signal(entry=0.50), current_equity=2000.0, now=now, allow_network_refresh=False)
            self.assertIsNotNone(tid)

            # Simulate partial fill: 2000 qty filled on exchange out of 6000 ordered
            cl.get_positions.side_effect = lambda symbol=None: [
                {"symbol": "ADAUSDT", "size": "2000", "markPrice": "0.501"}
            ]
            cl.get_order_status.return_value = {"orderStatus": "PartiallyFilled"}
            cl.place_limit_tp_order.return_value = {"orderId": "ORD-TP-1"}

            ex.sync_positions(now=now + timedelta(minutes=1), allow_network_refresh=False)

            cl.cancel_order.assert_called_with("ADAUSDT", "ORD-ENTRY-1")
            cl.place_limit_tp_order.assert_called_once()
            _, tp_kw = cl.place_limit_tp_order.call_args
            self.assertEqual(tp_kw["qty"], 2000.0)

            active = ex.get_active_trades()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["status"], "OPEN")
            self.assertEqual(active[0]["qty"], 2000.0)
            self.assertAlmostEqual(active[0]["notional_usdt"], 1000.0, places=2)
            self.assertAlmostEqual(active[0]["margin_usdt"], 100.0, places=2)

    def test_live_position_size_returns_none_on_error_and_prevents_false_cancel(self):
        now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            cl = MagicMock(has_credentials=True)
            cl.get_positions.return_value = []
            cl.get_open_orders.return_value = []
            cl.place_limit_entry_with_sl.return_value = {"orderId": "ORD-1"}
            ex, macro_p = self._build_executor(Path(tmp), cl, dry_run=False)
            _write_fresh_macro(macro_p, now, event_dt=None)

            tid = ex.open_position(_make_signal(), current_equity=2000.0, now=now, allow_network_refresh=False)
            self.assertIsNotNone(tid)

            # On TTL expiry: get_positions() succeeds once in sync_positions, then fails inside _live_position_size(symbol=sym)
            def _pos_side_effect(symbol=None):
                if symbol is not None:
                    raise RuntimeError("network blip")
                return []

            cl.get_positions.side_effect = _pos_side_effect
            cl.get_order_status.return_value = {"orderStatus": "New"}

            self.assertIsNone(ex._live_position_size("ADAUSDT"))
            ex.sync_positions(now=now + timedelta(hours=2), allow_network_refresh=False)

            # Trade must remain PENDING (NOT falsely marked CANCELLED)
            active = ex.get_active_trades()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["status"], "PENDING")

    def test_ensure_tp_order_market_close_and_replacement(self):
        now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            cl = MagicMock(has_credentials=True)
            ex, _ = self._build_executor(Path(tmp), cl, dry_run=False)
            with ex._connect() as conn:
                conn.execute(
                    "INSERT INTO trades (symbol, direction, status, mode, entry_time, entry_price, sl_price, tp1_price, tp2_price, be_price, "
                    "qty, tp1_qty, tp2_qty, remaining_qty, leverage, margin_usdt, notional_usdt, score, tp1_order_id, fees_usdt) "
                    "VALUES ('ADAUSDT', 'LONG', 'OPEN', 'demo', ?, 0.50, 0.4875, 0.506, 0.506, 0.5015, 1000.0, 1000.0, 0.0, 1000.0, 10, 50.0, 500.0, 80.0, 'TP-OLD', 0.1)",
                    (to_iso_z(now),),
                )
                conn.commit()
            trade = ex.get_active_trades()[0]

            # Case A: TP cancelled on exchange & price still below TP -> re-places TP for full pos_size
            cl.get_order_status.return_value = {"orderStatus": "Cancelled"}
            cl.place_limit_tp_order.return_value = {"orderId": "TP-NEW"}
            closed = ex._ensure_tp_order(trade, pos_size=1000.0, price=0.502, now=now)
            self.assertFalse(closed)
            cl.place_limit_tp_order.assert_called_once_with(symbol="ADAUSDT", side="Sell", qty=1000.0, price=0.506)

            # Case B: TP missing & price already beyond TP -> closes at market (TP_MARKET)
            cl.get_order_status.return_value = {"orderStatus": "Cancelled"}
            closed_mkt = ex._ensure_tp_order(trade, pos_size=1000.0, price=0.507, now=now)
            self.assertTrue(closed_mkt)
            cl.close_position_market.assert_called_once_with(symbol="ADAUSDT", side="Sell", qty=1000.0)

    def test_finalize_live_closed_uses_exchange_closed_pnl(self):
        now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            cl = MagicMock(has_credentials=True)
            cl.get_wallet_balance.return_value = {"equity": 2031.25, "available": 2031.25}
            cl.get_order_status.return_value = {"orderStatus": "Filled"}
            cl.get_closed_pnl.return_value = [
                {
                    "createdTime": str(int((now + timedelta(minutes=5)).timestamp() * 1000)),
                    "closedPnl": "31.25",
                    "closedSize": "6000",
                    "cumExitValue": "3036.0",
                    "openFee": "0.60",
                    "closeFee": "0.61",
                }
            ]
            ex, _ = self._build_executor(Path(tmp), cl, dry_run=False)
            with ex._connect() as conn:
                conn.execute(
                    "INSERT INTO trades (symbol, direction, status, mode, entry_time, entry_price, sl_price, tp1_price, tp2_price, be_price, "
                    "qty, tp1_qty, tp2_qty, remaining_qty, leverage, margin_usdt, notional_usdt, score, tp1_order_id, fees_usdt) "
                    "VALUES ('ADAUSDT', 'LONG', 'OPEN', 'demo', ?, 0.50, 0.4875, 0.506, 0.506, 0.5015, 6000.0, 6000.0, 0.0, 6000.0, 10, 300.0, 3000.0, 80.0, 'TP-1', 0.6)",
                    (to_iso_z(now),),
                )
                conn.commit()
            trade = ex.get_active_trades()[0]
            ex._finalize_live_closed(trade, now=now + timedelta(minutes=6))

            with ex._connect() as conn:
                row = dict(conn.execute("SELECT * FROM trades WHERE id = ?", (trade["id"],)).fetchone())
            self.assertEqual(row["status"], "CLOSED")
            self.assertEqual(row["exit_reason"], "TP")
            self.assertAlmostEqual(row["net_pnl_usdt"], 31.25, places=2)
            self.assertAlmostEqual(row["fees_usdt"], 1.21, places=2)


if __name__ == "__main__":
    unittest.main()
