import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from bot.news_guard import (
    NewsGuard,
    is_core_us_macro_title,
    minutes_to_next_event,
    to_iso_z,
)


class TestNewsGuard(unittest.TestCase):
    def test_core_us_macro_title_patterns(self):
        expected_core = [
            "Core PCE Price Index m/m",
            "Advance GDP q/q",
            "Prelim GDP q/q",
            "FOMC Statement",
            "CPI y/y",
            "Non-Farm Employment Change",
            "Core PPI m/m",
            "Fed Chair Powell Speaks",
        ]
        for title in expected_core:
            self.assertTrue(is_core_us_macro_title(title), f"Expected core macro match for: {title}")

        self.assertFalse(is_core_us_macro_title("Retail Sales m/m"))
        self.assertFalse(is_core_us_macro_title("Unemployment Claims"))

    def test_macro_windows_pre_and_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            macro_p = tmp_path / "macro_events.json"
            state_p = tmp_path / "news_state.json"

            ev_time = datetime(2026, 11, 25, 13, 30, tzinfo=timezone.utc)
            payload = {
                "fetched_at_utc": to_iso_z(ev_time - timedelta(hours=2)),
                "coverage_from_utc": to_iso_z(ev_time - timedelta(days=2)),
                "coverage_to_utc": to_iso_z(ev_time + timedelta(days=5)),
                "status": "OK",
                "core_count": 1,
                "usd_high_count": 1,
                "events": [
                    {
                        "title": "Core PCE Price Index m/m",
                        "country": "USD",
                        "impact": "High",
                        "time_utc": to_iso_z(ev_time),
                        "is_core_us_macro": True,
                        "forecast": "0.3%",
                        "previous": "0.3%",
                    }
                ],
                "all_usd_high_events": [],
            }
            macro_p.write_text(json.dumps(payload), encoding="utf-8")

            ng = NewsGuard(
                config={"macro_shield_enabled": True, "macro_pre_window_min": 30, "macro_post_window_min": 15},
                state_path=state_p,
                macro_events_path=macro_p,
            )

            # 15 min before event: entry blocked, cancel limits, protect open positions
            pre = ng.evaluate_macro_shield(now=ev_time - timedelta(minutes=15), allow_network_refresh=False)
            self.assertFalse(pre["entry_allowed"])
            self.assertTrue(pre["cancel_pending_limits"])
            self.assertTrue(pre["protect_open_positions"])

            # 10 min after event: entry blocked, cancel limits, protect_open_positions=False
            post = ng.evaluate_macro_shield(now=ev_time + timedelta(minutes=10), allow_network_refresh=False)
            self.assertFalse(post["entry_allowed"])
            self.assertTrue(post["cancel_pending_limits"])
            self.assertFalse(post["protect_open_positions"])

            # Outside window (60 min before): entry allowed and minutes_to_next_event == 60
            before = ng.evaluate_macro_shield(now=ev_time - timedelta(minutes=60), allow_network_refresh=False)
            self.assertTrue(before["entry_allowed"])
            self.assertAlmostEqual(minutes_to_next_event(before, now=ev_time - timedelta(minutes=60)), 60.0, places=2)

    def test_fail_closed_on_missing_corrupt_or_stale_calendar(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            macro_p = tmp_path / "macro_events.json"
            state_p = tmp_path / "news_state.json"
            ng = NewsGuard(
                config={"macro_shield_enabled": True, "macro_max_stale_hours": 36},
                state_path=state_p,
                macro_events_path=macro_p,
            )

            # Missing calendar
            res_missing = ng.evaluate_macro_shield(now=datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc), allow_network_refresh=False)
            self.assertTrue(res_missing["fail_safe_active"])
            self.assertFalse(res_missing["entry_allowed"])

            # Corrupt JSON
            macro_p.write_text("{bad json", encoding="utf-8")
            res_corrupt = ng.evaluate_macro_shield(now=datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc), allow_network_refresh=False)
            self.assertTrue(res_corrupt["fail_safe_active"])
            self.assertFalse(res_corrupt["entry_allowed"])

            # Stale calendar (fetched 40h ago > 36h)
            now = datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc)
            macro_p.write_text(
                json.dumps({"fetched_at_utc": to_iso_z(now - timedelta(hours=40)), "events": []}),
                encoding="utf-8",
            )
            res_stale = ng.evaluate_macro_shield(now=now, allow_network_refresh=False)
            self.assertTrue(res_stale["fail_safe_active"])
            self.assertFalse(res_stale["entry_allowed"])

            # Calendar fetched 2 days ago (not covering today UTC) even if within stale window
            macro_p.write_text(
                json.dumps({"fetched_at_utc": to_iso_z(now - timedelta(days=2)), "events": []}),
                encoding="utf-8",
            )
            res_old_day = ng.evaluate_macro_shield(now=now, allow_network_refresh=False)
            self.assertTrue(res_old_day["fail_safe_active"])
            self.assertFalse(res_old_day["entry_allowed"])


if __name__ == "__main__":
    unittest.main()
