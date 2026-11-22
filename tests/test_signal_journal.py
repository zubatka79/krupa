import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from bot.journal import SignalJournal
from bot.risk_manager import RiskManager
from bot.scanner import MarketScanner
from stats_report import report_db, trade_stats
from test_executor import TestExecutor, _base_config, _make_signal, _write_fresh_macro


class TestSignalJournal(unittest.TestCase):
    def _executor(self, tmp: Path, client: MagicMock, dry_run: bool = False):
        return TestExecutor._build_executor(self, tmp, client, dry_run=dry_run)

    def test_executor_reports_reject_reason(self):
        now = datetime(2026, 11, 25, 12, 15, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            cl = MagicMock(has_credentials=True)
            ex, macro_p = self._executor(Path(tmp), cl)
            _write_fresh_macro(macro_p, now, event_dt=now + timedelta(minutes=75))
            self.assertIsNone(ex.open_position(_make_signal(), current_equity=2000.0, now=now, allow_network_refresh=False))
            self.assertEqual(ex.last_reject_reason, "macro_event_soon")

            _write_fresh_macro(macro_p, now, event_dt=None)
            cl.get_positions.side_effect = RuntimeError("timeout")
            self.assertIsNone(ex.open_position(_make_signal(), current_equity=2000.0, now=now, allow_network_refresh=False))
            self.assertEqual(ex.last_reject_reason, "portfolio_check_failed")

    def test_scanner_collects_filtered_setups(self):
        cfg = _base_config()
        scanner = MarketScanner(client=MagicMock(), risk_manager=RiskManager(cfg), config=cfg)
        scanner._journal_bar_ts = 1_700_000_000_000
        spec = _make_signal().spec
        scanner._blocked("btc_1h_veto", spec, "LONG", "BTC 1H Veto blocked %s LONG (BTC 1H=%.2f%%).", "ADAUSDT", -1.25)
        self.assertEqual(len(scanner.journal_events), 1)
        ev = scanner.journal_events[0]
        self.assertEqual((ev["symbol"], ev["direction"], ev["outcome"], ev["reason"]), ("ADAUSDT", "LONG", "FILTERED", "btc_1h_veto"))
        self.assertIn("-1.25", ev["details"]["message"])

    def test_journal_and_report_end_to_end(self):
        now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ex, macro_p = self._executor(tmp_p, MagicMock(has_credentials=False), dry_run=True)
            _write_fresh_macro(macro_p, now, event_dt=None)
            tid = ex.open_position(_make_signal(), current_equity=2000.0, now=now, allow_network_refresh=False)
            self.assertIsNotNone(tid)

            journal = SignalJournal(ex.db_path, strategy="ada_fibo50_15x")
            journal.record(bar_ts_ms=1_700_000_000_000, symbol="ADAUSDT", direction="LONG", outcome="OPENED", trade_id=tid)
            journal.record(bar_ts_ms=1_700_000_900_000, symbol="ADAUSDT", direction="SHORT", outcome="FILTERED", reason="btc_1h_veto")
            self.assertTrue(journal.entry_exists(1_700_000_000_000, "ADAUSDT"))
            self.assertFalse(journal.entry_exists(1_700_000_900_000, "ADAUSDT"))

            with sqlite3.connect(ex.db_path) as conn:
                conn.execute(
                    "UPDATE trades SET status='CLOSED', exit_reason='TP', net_pnl_usdt=12.5, fees_usdt=0.4 WHERE id=?",
                    (tid,),
                )
            db = tmp_p / "trades_ada.db"
            ex.db_path.rename(db)

            lines, trades = report_db(db, since=None)
            text = "\n".join(lines)
            self.assertIn("Net PnL: +12.50", text)
            self.assertIn("FILTERED:btc_1h_veto - 1", text)
            self.assertEqual(trade_stats(trades)["win_rate"], 100.0)


if __name__ == "__main__":
    unittest.main()
