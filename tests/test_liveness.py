import tempfile
import threading
import unittest
from pathlib import Path

from ai_trader_v2.liveness import RunState, StallGuard, git_version


class TestGitVersion(unittest.TestCase):
    def test_loose_ref_packed_ref_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(git_version(root), "unknown")

            git = root / ".git"
            (git / "refs/heads").mkdir(parents=True)
            (git / "HEAD").write_text("ref: refs/heads/main\n")
            (git / "packed-refs").write_text("# pack-refs\n83914a39a56b9148defbe0c3e578af0acf9afa01 refs/heads/main\n")
            self.assertEqual(git_version(root), "83914a3")

            (git / "refs/heads/main").write_text("0dff565aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
            self.assertEqual(git_version(root), "0dff565")


class TestRunState(unittest.TestCase):
    def test_clean_then_crash_then_crash_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_state.json"

            st = RunState(path)
            st.start("abc1234", now=1000.0)
            self.assertEqual(st.startup_note(), "")
            st.mark_clean_exit()

            st = RunState(path)
            st.start("abc1234", now=2000.0)
            self.assertEqual(st.startup_note(), "")  # previous run exited cleanly
            # no mark_clean_exit: simulates a crash

            st = RunState(path)
            st.start("abc1234", now=2100.0)
            self.assertIn("аварийно", st.startup_note())

            for t in (2200.0, 2300.0):
                st = RunState(path)
                st.start("abc1234", now=t)
            self.assertIn("по кругу", st.startup_note())

            st = RunState(path)
            st.start("abc1234", now=2300.0 + 31 * 60)
            self.assertNotIn("по кругу", st.startup_note())

    def test_fatal_alert_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = RunState(Path(tmp) / "run_state.json")
            st.start("abc1234", now=1000.0)
            self.assertTrue(st.fatal_alert_allowed(now=1000.0))
            self.assertFalse(st.fatal_alert_allowed(now=1300.0))
            st2 = RunState(Path(tmp) / "run_state.json")
            st2.start("abc1234", now=1400.0)
            self.assertFalse(st2.fatal_alert_allowed(now=1400.0))
            self.assertTrue(st2.fatal_alert_allowed(now=1000.0 + 601))


class TestStallGuard(unittest.TestCase):
    def test_exits_when_loop_stops_beating(self):
        exited = threading.Event()
        alerts = []
        guard = StallGuard(0.2, on_stall=alerts.append, exit_fn=lambda code: exited.set(), check_every=0.05)
        guard.start()
        self.assertTrue(exited.wait(2))
        self.assertEqual(len(alerts), 1)

    def test_no_exit_while_beating(self):
        exited = threading.Event()
        guard = StallGuard(0.3, on_stall=lambda age: None, exit_fn=lambda code: exited.set(), check_every=0.05)
        guard.start()
        for _ in range(10):
            guard.beat()
            if exited.wait(0.05):
                break
        guard.stop()
        self.assertFalse(exited.is_set())


if __name__ == "__main__":
    unittest.main()
