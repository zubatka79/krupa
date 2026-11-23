import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ai_supervisor
import run_bot
import watchdog
from bot.supervisor_tools import redact_secrets


class TestMonitoringAndSecurity(unittest.TestCase):
    def test_watchdog_keeps_alerts_on_api_error_and_resolves_after_two_clean_checks(self):
        state = {"active_alerts": {"no_sl_ADAUSDT": 100.0, "proc_dead_LTCUSDT": 100.0}, "missing_checks": {}}
        with patch("watchdog.send_telegram", return_value=True) as mock_tg:
            # Round 1: Bybit API failed -> unchecked_prefixes=('no_sl_', 'no_tp_', ...)
            # no_sl_ADAUSDT must stay; proc_dead_LTCUSDT has 1 clean check (< 2), must NOT resolve yet
            watchdog.dispatch_alerts(
                state,
                current_issues={"bybit_api_err": "API timeout"},
                now_ts=2000.0,
                cooldown_sec=1800.0,
                unchecked_prefixes=("no_sl_", "no_tp_", "ttl_"),
            )
            self.assertIn("no_sl_ADAUSDT", state["active_alerts"])
            self.assertIn("proc_dead_LTCUSDT", state["active_alerts"])
            self.assertEqual(state["missing_checks"].get("proc_dead_LTCUSDT"), 1)

            # Round 2: clean check for all -> proc_dead_LTCUSDT hits 2 clean checks and resolves; no_sl_ADAUSDT hits 1
            watchdog.dispatch_alerts(
                state,
                current_issues={},
                now_ts=2200.0,
                cooldown_sec=1800.0,
                unchecked_prefixes=(),
            )
            self.assertNotIn("proc_dead_LTCUSDT", state["active_alerts"])
            self.assertIn("no_sl_ADAUSDT", state["active_alerts"])
            sent_texts = [c.args[0] for c in mock_tg.call_args_list]
            self.assertTrue(any("proc_dead_LTCUSDT" in t and "устранена" in t for t in sent_texts))

    def test_telegram_supervisor_authorization(self):
        with (
            patch("ai_supervisor.load_dotenv"),
            patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "", "TELEGRAM_ALLOWED_USER_IDS": ""}),
        ):
            sup = ai_supervisor.TelegramSupervisor()
            self.assertFalse(sup.access_configured)
            self.assertFalse(sup.is_authorized({"id": 999, "type": "private"}, {"id": 999}))

        with (
            patch("ai_supervisor.load_dotenv"),
            patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123:ABC", "TELEGRAM_CHAT_ID": "111", "TELEGRAM_ALLOWED_USER_IDS": "111"}),
        ):
            sup = ai_supervisor.TelegramSupervisor()
            self.assertTrue(sup.is_authorized({"id": 111, "type": "private"}, {"id": 111}))
            self.assertFalse(sup.is_authorized({"id": 222, "type": "private"}, {"id": 222}))
            self.assertFalse(sup.is_authorized({"id": 111, "type": "private"}, {"id": 999}))

    def test_pause_intent_and_symbol_extraction(self):
        self.assertEqual(ai_supervisor.parse_pause_intent("сними паузу с ada"), "unpause")
        self.assertEqual(ai_supervisor.parse_pause_intent("поставь wlfi на паузу на 12 часов"), "pause")
        self.assertIsNone(ai_supervisor.parse_pause_intent("покажи баланс"))

        syms, unknown, all_flag = ai_supervisor.extract_pause_symbols("поставь wlfi, crv и doge на паузу")
        self.assertEqual(syms, ["WLFIUSDT", "CRVUSDT"])
        self.assertEqual(unknown, ["DOGE"])
        self.assertFalse(all_flag)

        syms2, unknown2, all_flag2 = ai_supervisor.extract_pause_symbols("сними все паузы")
        self.assertEqual(syms2, [])
        self.assertTrue(all_flag2)

    def test_redact_secrets_strips_all_keys_and_tokens(self):
        fake_env = {
            "TELEGRAM_BOT_TOKEN": "8961559973:SECRET_TG_TOKEN",
            "ANTHROPIC_API_KEY": "sk-ant-secret-key-999",
            "GEMINI_API_KEY": "AIzaSecretGeminiKey",
            "AI_API_KEY": "sk-proxy-secret",
            "BYBIT_DEMO_API_KEY": "bybitDemoKey123",
            "BYBIT_DEMO_API_SECRET": "bybitDemoSecret456",
        }
        with patch.dict(os.environ, fake_env):
            raw = (
                "Error calling https://api.telegram.org/bot8961559973:SECRET_TG_TOKEN/getMe "
                "keys: sk-ant-secret-key-999 AIzaSecretGeminiKey sk-proxy-secret bybitDemoKey123 bybitDemoSecret456"
            )
            cleaned = redact_secrets(raw)
            for secret in fake_env.values():
                self.assertNotIn(secret, cleaned)
            self.assertNotIn("SECRET_TG_TOKEN", cleaned)

    def test_run_bot_load_config_validates_live_and_rejects_bad(self):
        root = Path(__file__).resolve().parent.parent
        expected_caps = {
            "config_ada.json": (40000.0, True),
            "config_wlfi.json": (8000.0, False),
            "config_ltc.json": (50000.0, True),
            "config_mnt.json": (20000.0, True),
            "config_crv.json": (10000.0, True),
            "config_hbar.json": (25000.0, True),
        }
        for fname, (exp_cap, exp_e4) in expected_caps.items():
            cfg = run_bot.load_config(root / fname)
            self.assertIn(cfg["mode"], ("demo", "testnet", "live", "mainnet"))
            self.assertEqual(cfg["max_portfolio_open_positions"], 5)
            self.assertEqual(cfg["max_portfolio_margin_pct"], 78.0)
            self.assertEqual(cfg["margin_pct_per_trade"], 15.0)
            self.assertEqual(cfg["leverage"], 10)
            self.assertEqual(cfg["max_notional_usdt"], exp_cap)
            self.assertEqual(cfg["btc_4h_bear_block_long"], exp_e4)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{broken", encoding="utf-8")
            with self.assertRaises(Exception):
                run_bot.load_config(p)

            p.write_text(json.dumps({"mode": "demo", "strategy": "ada_fibo50_15x", "whitelist": ["ADAUSDT"], "leverage": 30, "margin_pct_per_trade": 15}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_bot.load_config(p)

            p.write_text(json.dumps({"mode": "invalid_mode", "strategy": "ada_fibo50_15x", "whitelist": ["ADAUSDT"], "leverage": 10, "margin_pct_per_trade": 15}), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_bot.load_config(p)


if __name__ == "__main__":
    unittest.main()
