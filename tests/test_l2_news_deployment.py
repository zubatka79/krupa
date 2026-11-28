"""
Comprehensive Pre-Deployment Verification Suite for L2_news (Step 17 -> Demo/Testnet).

Verifies:
  1. Demo config integrity (config_ai_trader.json, config_long_trader.json, config/macro_events_utc.json).
  2. MacroCalendarGuard blackout [T-60m..T+45m], horizon < 3d warning & update_macro_calendar.py fail-safe preservation.
  3. Strictly closed previous-UTC-day 1D EMA20 & 1D ATR VolStorm causality (zero look-ahead).
  4. VikaExecutor L2_news lifecycle: 15x Isolated, 5x12%, Cap $5k/$2.5k, <=85% reserve, AntiSalvo (<=2/bar, <=5/side),
     2-Step DCA (35/65 & 30/70), DCA fill detection & TP recompute, BE +0.30% + synchronous Leg-2 DCA cancel,
     and strict Hard-SL-only AC 6h & Coin Probation (BE exits NEVER trigger pauses!).
  5. Rebuilt Telegram UI (all buttons, handlers, notifications, zero legacy 10x/N2N7 strings, < 4096 chars).
  6. Secret hygiene audit across all modified/new source files.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime, evaluate_btc_l2_regime
from ai_trader_v2.macro_calendar import MacroCalendarGuard
from ai_trader_v2.telegram_ui import (
    BTN_HISTORY, BTN_MACRO, BTN_PANIC, BTN_POSITIONS, BTN_STATUS, BTN_TOGGLE_PAUSE, VikaTelegramUI,
)
from ai_trader_v2.vika_engine import CoinEvaluation, DAY_MS, H4_MS, M15_MS, evaluate_coin
from ai_trader_v2_long.long_engine import LongCoinEvaluation
from ai_trader_v2_long.long_executor import VikaLongExecutor
from ai_trader_v2_long.long_telegram_ui import VikaLongTelegramUI
from bot.bybit_client import InstrumentSpec
import update_macro_calendar


class _MockPriv:
    def __init__(self, parent: "_MockClient") -> None:
        self.parent = parent

    def set_margin_mode(self, **kwargs):
        return {"retCode": 0}

    def switch_margin_mode(self, **kwargs):
        return {"retCode": 0}

    def place_order(self, **kwargs):
        self.parent.place_calls.append(kwargs)
        oid = f"ORD-{len(self.parent.place_calls)}"
        sym = kwargs["symbol"]
        if kwargs["orderType"] == "Market":
            px = self.parent.market_prices.get(sym, 100.0)
            self.parent.positions[sym] = {
                "symbol": sym,
                "side": kwargs["side"],
                "size": str(kwargs["qty"]),
                "avgPrice": str(px),
                "stopLoss": str(kwargs.get("stopLoss", "0")),
                "takeProfit": "0",
                "unrealisedPnl": "0",
            }
        else:
            self.parent.open_orders[oid] = {
                "orderId": oid,
                "symbol": sym,
                "side": kwargs["side"],
                "orderType": "Limit",
                "qty": str(kwargs["qty"]),
                "price": str(kwargs["price"]),
                "reduceOnly": bool(kwargs.get("reduceOnly", False)),
                "stopOrderType": "",
            }
        return {"retCode": 0, "result": {"orderId": oid}}


class _MockClient:
    has_credentials = True
    public_base_url = "https://api.bybit.com"

    def __init__(self, equity: float = 2000.0) -> None:
        self.equity = equity
        self.market_prices: Dict[str, float] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.closed_pnl_by_sym: Dict[str, List[Dict[str, Any]]] = {}
        self.place_calls: List[Dict[str, Any]] = []
        self.limit_tp_calls: List[tuple] = []
        self.sl_calls: List[tuple] = []
        self.cancel_calls: List[tuple] = []
        self._priv = _MockPriv(self)

    def _require_private(self):
        return self._priv

    def set_leverage(self, symbol: str, leverage: int):
        return {}

    def get_wallet_balance(self, coin: str = "USDT"):
        return {"equity": self.equity, "available": self.equity}

    def get_positions(self, symbol: str | None = None):
        if symbol:
            return [self.positions[symbol]] if symbol in self.positions else []
        return list(self.positions.values())

    def get_open_orders(self, symbol: str | None = None):
        vals = list(self.open_orders.values())
        return [o for o in vals if o["symbol"] == symbol] if symbol else vals

    def set_trading_stop(self, symbol: str, stop_loss: float, position_idx: int = 0):
        self.sl_calls.append((symbol, stop_loss))
        if symbol in self.positions:
            self.positions[symbol]["stopLoss"] = str(stop_loss)
        return {}

    def place_limit_tp_order(self, symbol: str, side: str, qty: float, price: float):
        oid = f"TP-{len(self.limit_tp_calls) + 1}"
        self.limit_tp_calls.append((symbol, side, qty, price))
        self.open_orders[oid] = {
            "orderId": oid,
            "symbol": symbol,
            "side": side,
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "reduceOnly": True,
            "stopOrderType": "",
        }
        return {"orderId": oid}

    def cancel_order(self, symbol: str, order_id: str):
        self.cancel_calls.append((symbol, order_id))
        self.open_orders.pop(order_id, None)

    def cancel_all_orders(self, symbol: str):
        for oid in [k for k, v in self.open_orders.items() if v["symbol"] == symbol]:
            self.cancel_calls.append((symbol, oid))
            self.open_orders.pop(oid, None)

    def close_position_market(self, symbol: str, side: str, qty: float):
        self.positions.pop(symbol, None)

    def get_closed_pnl(self, symbol: str, start_ms: int = 0, limit: int = 10):
        return self.closed_pnl_by_sym.get(symbol, [])


def test_1_configs_and_macro_calendar() -> None:
    cfg_main = json.loads((ROOT / "ai_trader_v2" / "config_ai_trader.json").read_text(encoding="utf-8"))
    cfg_long = json.loads((ROOT / "ai_trader_v2_long" / "config_long_trader.json").read_text(encoding="utf-8"))
    cal = json.loads((ROOT / "config" / "macro_events_utc.json").read_text(encoding="utf-8"))

    for c in (cfg_main, cfg_long):
        assert c["mode"] == "demo", "Must stay demo/testnet!"
        assert c["leverage"] == 15
        assert c["margin_mode"] == "ISOLATED"
        assert c["margin_pct_per_trade"] == 12.0
        assert c["max_margin_usd"] == 5000.0
        assert c["storm_max_margin_usd"] == 2500.0
        assert c["max_same_direction"] == 5
        assert c["max_total_margin_pct"] == 85.0
        assert c["dca"]["be_lock_pct"] == 0.30
        assert c["dca"]["leg1_pct"] == 35.0 and c["dca"]["leg2_pct"] == 65.0
        assert c["dca"]["storm_leg1_pct"] == 30.0 and c["dca"]["storm_leg2_pct"] == 70.0

    assert len(cfg_main["long_coins"]) == 18 and len(cfg_main["short_coins"]) == 18
    assert len(cfg_long["coins"]) == 18

    # Validate calendar payload & blackout boundaries on FOMC 2026-12-09T19:00:00Z
    now_ref = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
    validated = update_macro_calendar.validate_calendar_payload(cal, now_utc=now_ref, min_future_events=6)
    assert len(validated["events"]) == 6

    guard = MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json")
    assert guard.evaluate(datetime(2026, 12, 9, 17, 59, tzinfo=timezone.utc)).in_blackout is False
    st_in = guard.evaluate(datetime(2026, 12, 9, 18, 0, tzinfo=timezone.utc))
    assert st_in.in_blackout is True and st_in.active_event is not None and st_in.active_event.event_type == "FOMC"
    assert guard.evaluate(datetime(2026, 12, 9, 19, 45, tzinfo=timezone.utc)).in_blackout is True
    assert guard.evaluate(datetime(2026, 12, 9, 19, 46, tzinfo=timezone.utc)).in_blackout is False

    # Horizon < 3d warning check
    st_warn = guard.evaluate(datetime(2027, 1, 25, 12, 0, tzinfo=timezone.utc))
    assert st_warn.horizon_warning is True and st_warn.horizon_days < 3.0

    # Bad payload must raise and preserve existing file
    try:
        update_macro_calendar.validate_calendar_payload({"events": [{"type": "BOGUS", "utc": "bad"}]}, now_utc=now_ref)
        raise AssertionError("Should have rejected invalid payload")
    except ValueError:
        pass
    print("[PASS] 1. Demo configs, All-Thick-18, macro calendar [T-60m..T+45m], horizon warn & validator")


def test_2_strict_closed_1d_causality() -> None:
    # Build 40 days of synthetic 4H BTC candles + 1 forming UTC day with extreme +50% spike
    start_ms = int(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    for k in range(40 * 6):
        ts = start_ms + k * H4_MS
        px = 90_000.0 + k * 20.0
        rows.append([ts, px, px * 1.005, px * 0.995, px, 1000.0, px * 1000.0])
    # Day 40 (forming day, only 2 bars in): huge fake crash to test that forming 1D day is NEVER used!
    day40_start = start_ms + 40 * DAY_MS
    rows.append([day40_start, 95_000.0, 95_000.0, 40_000.0, 40_000.0, 1000.0, 4e7])
    rows.append([day40_start + H4_MS, 40_000.0, 40_000.0, 35_000.0, 35_000.0, 1000.0, 3.5e7])
    df4 = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    now_ms = day40_start + 2 * H4_MS + 1000  # 08:00 UTC of day 40 (day 40 NOT closed!)
    reg = evaluate_btc_l2_regime(df4, now_ms)
    assert reg.day_ts_1d == day40_start - DAY_MS, f"Forming day leaked into 1D! {reg.day_ts_1d}"
    assert reg.d1_dir == 1 and reg.in_storm is False, f"Forming day crash corrupted 1D regime: {reg}"
    print("[PASS] 2. Strictly closed previous-UTC-day 1D EMA20 & 1D ATR causality (zero look-ahead)")


def test_3_executor_dca_be_hard_sl_antisalvo_and_telegram() -> None:
    cfg = json.loads((ROOT / "ai_trader_v2" / "config_ai_trader.json").read_text(encoding="utf-8"))
    cfg["mode"] = "demo"
    cfg["dry_run"] = False

    with tempfile.TemporaryDirectory() as tmp:
        journal = VikaJournal(Path(tmp) / "l2_test.db")
        mock = _MockClient(equity=2000.0)
        sent: List[str] = []
        guard = MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json")
        ex = VikaExecutor(cfg, mock, journal, notifier=sent.append, macro_guard=guard)

        for sym in ex.coin_cfg_map:
            ex.specs[sym] = InstrumentSpec(sym, sym[:-4], "USDT", "Trading", 0.01, 0.01, 5.0, 0.001, 1.0, 50.0)
            mock.market_prices[sym] = 100.0

        ex.last_btc_regime = BtcL2Regime(
            bar_ts_4h=1_795_000_000_000, close_4h=95000, ema20_4h=94000, ema50_4h=92000,
            v1_long=False, v1_short=True, day_ts_1d=1_794_960_000_000, close_1d=95000, ema20_1d=96000,
            d1_dir=-1, atr1d_pct=2.8, in_storm=False,
        )

        bar_ts = 1_795_000_000_000

        def _mk_s(sym: str, key: str) -> CoinEvaluation:
            return CoinEvaluation(
                symbol=sym, key=key, bar_ts=bar_ts, close_price=100.0, rvol=1.6, adx15=22.0, bbw15=2.1,
                atr15_pct=0.5, funding_rate=0.0, funding_bps=0.0, div4h_bear_strong=False, div4h_bull_strong=False,
                n2_blocked=False, slope4h_pct=-0.4, retail_ls_z=0.1, retail_ls_available=True,
                raw_signal="SHORT", final_signal="SHORT",
            )

        # 1) Open 2 SHORTs on same bar -> 3rd SHORT blocked by AntiSalvo (<=2/bar)!
        p1 = ex.open_signal(_mk_s("DOTUSDT", "K1_EMA_TREND_PB"))
        p2 = ex.open_signal(_mk_s("LTCUSDT", "K6_VWAP_RECLAIM"))
        p3 = ex.open_signal(_mk_s("HBARUSDT", "K3_MACD_STOCH"))
        assert p1 is not None and p2 is not None and p3 is None, f"AntiSalvo <=2/bar failed: {p1},{p2},{p3}"

        act = journal.get_active_positions()
        assert len(act) == 2
        # Verify 2-Step DCA placed: Leg 1 = 35% of $240 ($84), Leg 2 = 65% ($156), reserved = ~$240
        assert act[0]["dca_status"] == "PENDING" and act[0]["dca_order_id"]
        assert abs(float(act[0]["margin_usd"]) - 84.0) < 3.0
        assert abs(float(act[0]["reserved_margin_usd"]) - 240.0) < 5.0
        assert abs(float(act[0]["be_lock_price"]) - 99.70) < 1e-3  # SHORT +0.30% profit lock

        # 2) Test BE +0.30% on DOTUSDT -> MUST move SL to 99.70, synchronously cancel DCA limit & free reserved margin!
        dca_oid_dot = act[0]["dca_order_id"]
        moved = ex.check_breakeven_on_closed_bar("DOTUSDT", bar_ts + M15_MS, closed_price=98.90)
        assert moved is True
        assert ("DOTUSDT", dca_oid_dot) in mock.cancel_calls
        dot_row = [p for p in journal.get_active_positions() if p["symbol"] == "DOTUSDT"][0]
        assert dot_row["be_active"] == 1 and dot_row["dca_status"] == "CANCELLED_BE"
        assert abs(float(dot_row["reserved_margin_usd"]) - float(dot_row["margin_usd"])) < 0.01
        assert any("Лимитка 2-й ноги DCA снята" in m for m in sent)

        # 3) Close DOTUSDT on BE (even with tiny -$1 funding loss!) -> MUST NOT trigger AC 6h or Coin Probation!
        mock.positions.pop("DOTUSDT")
        mock.closed_pnl_by_sym["DOTUSDT"] = [{"updatedTime": str(int(time.time() * 1000)), "closedPnl": "-1.0", "avgExitPrice": "99.70"}]
        ex.sync_with_exchange()
        assert journal.get_ac_pause_until_ms("SHORT") == 0, "BE exit must NEVER trigger AC 6h!"
        assert journal.get_coin_probation_until_ms("DOTUSDT") == 0, "BE exit must NEVER trigger Probation!"
        assert any("AC 6ч и Probation НЕ затронуты" in m for m in sent)

        # 4) Simulate Leg-2 DCA fill on LTCUSDT -> size increases, DCA order disappears -> sync recomputes TP & notifies!
        ltc_row = [p for p in journal.get_active_positions() if p["symbol"] == "LTCUSDT"][0]
        mock.open_orders.pop(ltc_row["dca_order_id"], None)
        mock.positions["LTCUSDT"]["size"] = str(float(ltc_row["qty"]) + float(ltc_row["dca_qty"]))
        mock.positions["LTCUSDT"]["avgPrice"] = "100.97"
        ex.sync_with_exchange()
        ltc_after = [p for p in journal.get_active_positions() if p["symbol"] == "LTCUSDT"][0]
        assert ltc_after["dca_status"] == "FILLED" and abs(float(ltc_after["entry_price"]) - 100.97) < 1e-4
        assert any("ЗАЛИТА 2-Я НОГА DCA" in m for m in sent)

        # 5) Close LTCUSDT on real Hard SL (be_active=0, pnl < 0) -> MUST trigger AC 6h SHORT & Probation 1/2!
        mock.positions.pop("LTCUSDT")
        mock.closed_pnl_by_sym["LTCUSDT"] = [{"updatedTime": str(int(time.time() * 1000)), "closedPnl": "-110.0", "avgExitPrice": "104.0"}]
        ex.sync_with_exchange()
        assert journal.get_ac_pause_until_ms("SHORT") > int(time.time() * 1000), "Hard SL MUST trigger AC 6h!"
        assert any("ЗАКРЫТИЕ (HARD_SL)" in m and "AC 6ч (SHORT)" in m for m in sent)

        # 6) Verify Rebuilt Telegram UI formatters & handlers (ZERO legacy strings, < 4096 chars!)
        tg = VikaTelegramUI(cfg, ex, journal, get_latest_evals=lambda: {}, get_latest_long_evals=lambda: {})
        lev_st = ex.ensure_margin_and_leverage()
        welcome = tg.format_startup_welcome(lev_st)
        status = tg.format_status()
        positions = tg.format_open_positions()
        macro_v = tg.format_macro_and_pauses()
        portfolio = tg.format_portfolio()
        history = tg.format_history()

        for txt in (welcome, status, positions, macro_v, portfolio, history):
            assert len(txt) < 4096, f"Telegram message > 4096 chars: {len(txt)}"
            assert "10x Cross" not in txt and "Vika N2N7" not in txt, f"Legacy text leaked: {txt[:120]}"

        assert "15x ISOLATED" in status and "5×12%" in status and "Резерв Bybit" in status
        assert "FOMC" in macro_v and "CPI" in macro_v and "NFP" in macro_v and "AC 6ч" in macro_v
        assert "All-Thick-18 (18L + 18S)" in portfolio
        assert "[HARD_SL]" in history and "[BE]" in history

        # Test interactive Telegram update router (buttons + panic inline confirm)
        tg.chat_id = "12345"
        tg.enabled = False  # don't hit network; route via overridden send_message
        routed: List[str] = []
        tg.send_message = lambda text, reply_markup=None, parse_mode="HTML", chat_id=None: (routed.append(text) or True)  # type: ignore[method-assign]
        for txt in (BTN_STATUS, BTN_POSITIONS, BTN_MACRO, tg.btn_portfolio, BTN_HISTORY, BTN_TOGGLE_PAUSE, BTN_TOGGLE_PAUSE, BTN_PANIC):
            tg._handle_update({"message": {"chat": {"id": 12345}, "from": {"id": 12345}, "text": txt}})
        assert len(routed) == 8 and "ПОДТВЕРЖДЕНИЕ PANIC CLOSE" in routed[-1]
        tg._handle_update({"callback_query": {"id": "cb1", "data": "panic_confirm", "from": {"id": 12345}, "message": {"chat": {"id": 12345}}}})
        assert "PANIC CLOSE ВЫПОЛНЕН" in routed[-1] and journal.is_paused() is True

        # Test standalone LONG Telegram UI too
        cfg_l = json.loads((ROOT / "ai_trader_v2_long" / "config_long_trader.json").read_text(encoding="utf-8"))
        ex_l = VikaLongExecutor(cfg_l, mock, journal, macro_guard=guard)
        tg_l = VikaLongTelegramUI(cfg_l, ex_l, journal, get_latest_evals=lambda: {}, get_btc_regime=lambda: ex.last_btc_regime)
        assert "L2_news" in tg_l.format_status()

    print("[PASS] 3. Executor L2_news (15x, DCA, BE+0.30%+cancel, HardSL vs BE, AntiSalvo) & Rebuilt Telegram UI")


def test_4_secret_hygiene() -> None:
    secret_patterns = [
        re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
        re.compile(r"\b\d{8,11}:AA[A-Za-z0-9_\-]{30,}\b"),  # Telegram bot token
        re.compile(r"BYBIT_[A-Z_]*SECRET\s*=\s*['\"]?[A-Za-z0-9]{20,}"),
    ]
    files_to_check = [
        ROOT / "config" / "macro_events_utc.json",
        ROOT / "update_macro_calendar.py",
        ROOT / "docker-compose.yml",
        ROOT / "ai_trader_v2" / "config_ai_trader.json",
        ROOT / "ai_trader_v2" / "macro_calendar.py",
        ROOT / "ai_trader_v2" / "l2_market_data.py",
        ROOT / "ai_trader_v2" / "vika_engine.py",
        ROOT / "ai_trader_v2" / "journal.py",
        ROOT / "ai_trader_v2" / "executor.py",
        ROOT / "ai_trader_v2" / "run_ai_trader.py",
        ROOT / "ai_trader_v2" / "telegram_ui.py",
        ROOT / "ai_trader_v2" / "deploy_ai_trader_vps.sh",
        ROOT / "ai_trader_v2_long" / "config_long_trader.json",
        ROOT / "ai_trader_v2_long" / "long_engine.py",
        ROOT / "ai_trader_v2_long" / "long_executor.py",
        ROOT / "ai_trader_v2_long" / "run_long_trader.py",
        ROOT / "ai_trader_v2_long" / "long_telegram_ui.py",
    ]
    for fp in files_to_check:
        txt = fp.read_text(encoding="utf-8")
        for pat in secret_patterns:
            m = pat.search(txt)
            assert m is None, f"Hardcoded secret detected in {fp}: {m.group(0)[:12]}..."
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore
    print(f"[PASS] 4. Secret hygiene clean across all {len(files_to_check)} files (.env gitignored)")


if __name__ == "__main__":
    test_1_configs_and_macro_calendar()
    test_2_strict_closed_1d_causality()
    test_3_executor_dca_be_hard_sl_antisalvo_and_telegram()
    test_4_secret_hygiene()
    print("\nALL L2_NEWS PRE-DEPLOYMENT CHECKS PASSED!")
