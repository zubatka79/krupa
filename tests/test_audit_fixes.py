"""
Unit & Integration Regression Suite for the 10 Vika Hybrid Audit Fixes (`fix/audit-v3-hardening`).

Covers:
  1. [CRITICAL-1] Partial Leg-2 DCA fill tracking, TP qty/price sync, and preservation on remainder cancel.
  2. [CRITICAL-2] Partial Limit TP reduction cancels pending DCA; closedPnl sums ALL partial records.
  3. [HIGH-1] Atomic PENDING_OPEN persistence BEFORE Market entry + crash/restart recovery.
  4. [HIGH-2] BE vs DCA race (Bybit 110001) re-queries live position before SL move or CANCELLED_BE.
  5. [HIGH-3] VikaExecutor RLock serialization + isolated pybit.HTTP session for Telegram poll thread.
  6. [HIGH-4] walletBalance vs totalEquity separation, single negative uPnL deduction, and sizing from walletBalance.
  7. [MEDIUM-1] nextPageCursor pagination in BybitClient.get_positions() & get_open_orders() (>50 items).
  8. [MEDIUM-2] SQLite PRAGMA journal_mode=WAL + single atomic transaction for exit + cooldown + AC + Probation.
  9. [LOW-1] Directional Retail L/S z labels in Telegram UI + real closedPnl booking in panic_close_all().
 10. [Data Drift & N7] limit=600 for 4H/15m klines + 8h-normalized N7 funding filter for 4h coins.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime
from ai_trader_v2.macro_calendar import MacroCalendarGuard
from ai_trader_v2.run_ai_trader import VikaRunner
from ai_trader_v2.telegram_ui import VikaTelegramUI
from ai_trader_v2.vika_engine import CoinEvaluation, H1_MS, H4_MS, M15_MS, evaluate_coin
from bot.bybit_client import BybitClient, InstrumentSpec


def _load_cfg() -> Dict[str, Any]:
    cfg = json.loads((ROOT / "ai_trader_v2" / "config_ai_trader.json").read_text(encoding="utf-8"))
    cfg["mode"] = "demo"
    cfg["dry_run"] = False
    return cfg


def _attach_specs(ex: VikaExecutor) -> None:
    for sym in ex.coin_cfg_map:
        ex.specs[sym] = InstrumentSpec(
            sym, sym[:-4], "USDT", "Trading", 0.01, 0.01, 5.0, 0.001, 1.0, 50.0
        )


def _regime(in_storm: bool = False) -> BtcL2Regime:
    return BtcL2Regime(
        bar_ts_4h=1_795_000_000_000,
        close_4h=95000.0,
        ema20_4h=94000.0,
        ema50_4h=92000.0,
        v1_long=False,
        v1_short=True,
        day_ts_1d=1_794_960_000_000,
        close_1d=95000.0,
        ema20_1d=96000.0,
        d1_dir=-1,
        atr1d_pct=5.0 if in_storm else 2.5,
        in_storm=in_storm,
    )


def _short_ev(sym: str = "DOTUSDT", key: str = "K1_EMA_TREND_PB", bar_ts: int = 1_795_000_000_000, px: float = 100.0) -> CoinEvaluation:
    return CoinEvaluation(
        symbol=sym,
        key=key,
        bar_ts=bar_ts,
        close_price=px,
        rvol=1.6,
        adx15=22.0,
        bbw15=2.1,
        atr15_pct=0.5,
        funding_rate=0.0,
        funding_bps=0.0,
        div4h_bear_strong=False,
        div4h_bull_strong=False,
        n2_blocked=False,
        slope4h_pct=-0.4,
        retail_ls_z=0.1,
        retail_ls_available=True,
        raw_signal="SHORT",
        final_signal="SHORT",
    )


class _AuditMockPriv:
    def __init__(self, parent: "_AuditMockClient") -> None:
        self.parent = parent

    def set_margin_mode(self, **kwargs):
        return {"retCode": 0}

    def switch_margin_mode(self, **kwargs):
        return {"retCode": 0}

    def place_order(self, **kwargs):
        if self.parent.on_before_market_place is not None and kwargs.get("orderType") == "Market":
            self.parent.on_before_market_place(kwargs)
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
                "leavesQty": str(kwargs["qty"]),
                "cumExecQty": "0",
                "price": str(kwargs["price"]),
                "reduceOnly": bool(kwargs.get("reduceOnly", False)),
                "stopOrderType": "",
                "createdTime": str(int(time.time() * 1000)),
            }
        return {"retCode": 0, "result": {"orderId": oid}}


class _AuditMockClient:
    has_credentials = True
    public_base_url = "https://api.bybit.com"

    def __init__(self, wallet_balance: float = 2000.0, total_equity: Optional[float] = None) -> None:
        self.wallet_balance = wallet_balance
        self.total_equity = wallet_balance if total_equity is None else total_equity
        self.market_prices: Dict[str, float] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.open_orders: Dict[str, Dict[str, Any]] = {}
        self.closed_pnl_by_sym: Dict[str, List[Dict[str, Any]]] = {}
        self.place_calls: List[Dict[str, Any]] = []
        self.limit_tp_calls: List[tuple] = []
        self.sl_calls: List[tuple] = []
        self.cancel_calls: List[tuple] = []
        self.cancel_110001_ids: set[str] = set()
        self.on_cancel_hook: Any = None
        self.on_before_market_place: Any = None
        self._priv = _AuditMockPriv(self)

    def _require_private(self):
        return self._priv

    def set_leverage(self, symbol: str, leverage: int):
        return {}

    def get_wallet_balance(self, coin: str = "USDT") -> Dict[str, float]:
        return {
            "wallet_balance": self.wallet_balance,
            "total_equity": self.total_equity,
            "equity": self.total_equity,
            "available": min(self.wallet_balance, self.total_equity),
        }

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        if symbol:
            return [dict(self.positions[symbol])] if symbol in self.positions else []
        return [dict(p) for p in self.positions.values()]

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        vals = [dict(o) for o in self.open_orders.values()]
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

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        self.cancel_calls.append((symbol, order_id))
        if self.on_cancel_hook is not None:
            self.on_cancel_hook(symbol, order_id)
        self.open_orders.pop(order_id, None)
        if order_id in self.cancel_110001_ids:
            return False
        return True

    def cancel_all_orders(self, symbol: str):
        for oid in [k for k, v in self.open_orders.items() if v["symbol"] == symbol]:
            self.cancel_calls.append((symbol, oid))
            self.open_orders.pop(oid, None)

    def close_position_market(self, symbol: str, side: str, qty: float):
        self.positions.pop(symbol, None)

    def get_closed_pnl(self, symbol: str, start_ms: int = 0, limit: int = 50):
        return list(self.closed_pnl_by_sym.get(symbol, []))


class TestAuditFixes(unittest.TestCase):
    def test_01_critical1_partial_dca_fill_updates_tp_and_preserves_fill_on_cancel(self) -> None:
        cfg = _load_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            journal = VikaJournal(Path(tmp) / "t1.db")
            mock = _AuditMockClient(wallet_balance=2000.0)
            mock.market_prices["DOTUSDT"] = 100.0
            ex = VikaExecutor(cfg, mock, journal, macro_guard=MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json"))
            _attach_specs(ex)
            ex.last_btc_regime = _regime(in_storm=False)

            pid = ex.open_signal(_short_ev("DOTUSDT", "K1_EMA_TREND_PB"))
            self.assertIsNotNone(pid)
            pos = journal.get_active_positions()[0]
            dca_oid = pos["dca_order_id"]
            leg1_qty = float(pos["qty"])
            dca_qty = float(pos["dca_qty"])

            # Simulate 50% partial fill of Leg-2 DCA while remainder stays resting in orderbook
            part_fill = round(dca_qty * 0.5, 2)
            rem_qty = round(dca_qty - part_fill, 2)
            new_sz = round(leg1_qty + part_fill, 2)
            new_avg = 100.715
            mock.positions["DOTUSDT"]["size"] = str(new_sz)
            mock.positions["DOTUSDT"]["avgPrice"] = str(new_avg)
            mock.open_orders[dca_oid]["leavesQty"] = str(rem_qty)
            mock.open_orders[dca_oid]["cumExecQty"] = str(part_fill)

            ex.sync_with_exchange()

            after_partial = journal.get_active_positions()[0]
            self.assertEqual(after_partial["dca_status"], "PENDING")
            self.assertAlmostEqual(float(after_partial["qty"]), new_sz, places=4)
            self.assertAlmostEqual(float(after_partial["entry_price"]), new_avg, places=4)
            self.assertAlmostEqual(float(after_partial["dca_filled_qty"]), part_fill, places=4)
            expected_tp = round(new_avg * 0.985, 3)
            self.assertAlmostEqual(float(after_partial["tp_price"]), expected_tp, places=3)
            # Old Leg-1 TP cancelled, new TP placed for partial-filled full qty
            tp_orders = [o for o in mock.open_orders.values() if o.get("reduceOnly")]
            self.assertEqual(len(tp_orders), 1)
            self.assertAlmostEqual(float(tp_orders[0]["qty"]), new_sz, places=4)
            self.assertAlmostEqual(float(tp_orders[0]["price"]), expected_tp, places=3)

            # Now cancel remaining DCA externally -> partial fill MUST be preserved (PARTIAL_FILLED), remainder margin freed
            mock.open_orders.pop(dca_oid)
            ex.sync_with_exchange()
            after_cancel = journal.get_active_positions()[0]
            self.assertEqual(after_cancel["dca_status"], "PARTIAL_FILLED")
            self.assertAlmostEqual(float(after_cancel["qty"]), new_sz, places=4)
            self.assertAlmostEqual(float(after_cancel["entry_price"]), new_avg, places=4)
            self.assertAlmostEqual(float(after_cancel["reserved_margin_usd"]), float(after_cancel["margin_usd"]), places=2)

    def test_02_critical2_partial_tp_cancels_dca_and_sums_all_closed_pnl(self) -> None:
        cfg = _load_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            journal = VikaJournal(Path(tmp) / "t2.db")
            mock = _AuditMockClient(wallet_balance=2000.0)
            mock.market_prices["LTCUSDT"] = 100.0
            ex = VikaExecutor(cfg, mock, journal, macro_guard=MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json"))
            _attach_specs(ex)
            ex.last_btc_regime = _regime(in_storm=False)

            ex.open_signal(_short_ev("LTCUSDT", "K6_VWAP_RECLAIM"))
            pos = journal.get_active_positions()[0]
            dca_oid = pos["dca_order_id"]

            # 1) Partial Limit TP reduction -> pending DCA MUST be cancelled immediately
            half_qty = round(float(pos["qty"]) * 0.5, 2)
            mock.positions["LTCUSDT"]["size"] = str(half_qty)
            ex.sync_with_exchange()
            self.assertIn(("LTCUSDT", dca_oid), mock.cancel_calls)
            pos_after_part = journal.get_active_positions()[0]
            self.assertEqual(pos_after_part["dca_status"], "CANCELLED_TP_PARTIAL")
            self.assertAlmostEqual(float(pos_after_part["qty"]), half_qty, places=4)

            # 2) Position fully closes across 3 partial closedPnl entries where closed_list[0] is negative!
            now_ms = int(time.time() * 1000)
            mock.positions.pop("LTCUSDT")
            mock.closed_pnl_by_sym["LTCUSDT"] = [
                {"updatedTime": str(now_ms + 3), "closedPnl": "-0.50", "closedSize": "1.0", "avgExitPrice": "100.5"},
                {"updatedTime": str(now_ms + 2), "closedPnl": "9.00", "closedSize": "5.3", "avgExitPrice": "98.5"},
                {"updatedTime": str(now_ms + 1), "closedPnl": "14.00", "closedSize": "6.3", "avgExitPrice": "98.5"},
            ]
            ex.sync_with_exchange()

            closed = journal.get_recent_closed(limit=5)
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0]["exit_reason"], "TP")
            self.assertAlmostEqual(float(closed[0]["pnl_usd"]), 22.50, places=2)
            self.assertEqual(journal.get_ac_pause_until_ms("SHORT"), 0)
            self.assertEqual(journal.get_coin_probation_until_ms("LTCUSDT"), 0)

    def test_03_high1_atomic_pending_open_before_market_and_restart_recovery(self) -> None:
        cfg = _load_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            journal = VikaJournal(Path(tmp) / "t3.db")
            mock = _AuditMockClient(wallet_balance=2000.0)
            mock.market_prices["DOTUSDT"] = 100.0
            ex = VikaExecutor(cfg, mock, journal, macro_guard=MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json"))
            _attach_specs(ex)
            ex.last_btc_regime = _regime(in_storm=False)

            seen_pending_before_market: List[int] = []

            def _check_draft(kwargs):
                rows = journal.get_pending_open_positions()
                seen_pending_before_market.append(len(rows))

            mock.on_before_market_place = _check_draft
            pid = ex.open_signal(_short_ev("DOTUSDT", "K1_EMA_TREND_PB"))
            self.assertIsNotNone(pid)
            self.assertEqual(seen_pending_before_market, [1])
            self.assertEqual(journal.get_pending_open_positions(), [])
            self.assertEqual(journal.get_active_positions()[0]["status"], "OPEN")

            # Simulate crash right after Market + DCA placement while DB still has PENDING_OPEN
            old_ts_ms = int(time.time() * 1000) - 60 * 60_000  # 60m old (>45m TTL!)
            pid2 = journal.add_position(
                symbol="LTCUSDT",
                side="SHORT",
                key_name="K6_VWAP_RECLAIM",
                status="PENDING_OPEN",
                order_id="",
                entry_price=100.0,
                qty=12.6,
                margin_usd=84.0,
                reserved_margin_usd=240.0,
                leverage=15,
                tp_price=98.5,
                sl_price=104.0,
                be_trigger_price=99.0,
                be_lock_price=99.7,
                entry_bar_ts=1_795_000_000_000,
                expire_ts=0,
                filters={},
                dca_order_id="",
                dca_price=101.5,
                dca_qty=23.05,
                dca_margin_usd=156.0,
                dca_status="PENDING",
                in_storm=False,
                tp_pct=1.5,
            )
            mock.positions["LTCUSDT"] = {
                "symbol": "LTCUSDT",
                "side": "Sell",
                "size": "12.6",
                "avgPrice": "100.0",
                "stopLoss": "104.0",
                "takeProfit": "0",
                "unrealisedPnl": "0",
            }
            mock.open_orders["DCA-RECOVER-60M"] = {
                "orderId": "DCA-RECOVER-60M",
                "symbol": "LTCUSDT",
                "side": "Sell",
                "orderType": "Limit",
                "qty": "23.05",
                "price": "101.5",
                "reduceOnly": False,
                "stopOrderType": "",
                "createdTime": str(old_ts_ms),
            }
            cancelled = ex.cleanup_stale_entry_orders()
            self.assertEqual(cancelled, [])
            self.assertIn("DCA-RECOVER-60M", mock.open_orders)

            ltc = [p for p in journal.get_active_positions() if p["id"] == pid2][0]
            self.assertEqual(ltc["status"], "OPEN")
            self.assertEqual(ltc["dca_status"], "PENDING")
            self.assertEqual(ltc["dca_order_id"], "DCA-RECOVER-60M")

            # Unfilled PENDING_OPEN recovery cleans up cleanly
            pid3 = journal.add_position(
                symbol="HBARUSDT",
                side="SHORT",
                key_name="N2_BB_SQZ_REJECT_S",
                status="PENDING_OPEN",
                order_id="",
                entry_price=0.1,
                qty=100.0,
                margin_usd=10.0,
                leverage=15,
                tp_price=0.0985,
                sl_price=0.104,
                be_trigger_price=0.099,
                be_lock_price=0.0997,
                entry_bar_ts=1_795_000_000_000,
                expire_ts=0,
                filters={},
            )
            ex.sync_with_exchange()
            self.assertTrue(all(p["id"] != pid3 for p in journal.get_active_positions()))

    def test_04_high2_race_be_vs_dca_110001_rechecks_exchange_position(self) -> None:
        # Part A: BybitClient.cancel_order returns False ONLY on 110001 and raises on real errors
        cli = BybitClient(mode="demo", api_key="k", api_secret="s")
        fake_priv = MagicMock()
        fake_priv.cancel_order.side_effect = [
            {"retCode": 0},
            {"retCode": 110001, "retMsg": "order not exists"},
            RuntimeError("ErrCode: 110001"),
            RuntimeError("ErrCode: 10001 network"),
        ]
        cli._private_client = fake_priv
        cli._auto_private_client = False
        self.assertTrue(cli.cancel_order("DOTUSDT", "O1"))
        self.assertFalse(cli.cancel_order("DOTUSDT", "O2"))
        self.assertFalse(cli.cancel_order("DOTUSDT", "O3"))
        with self.assertRaises(RuntimeError):
            cli.cancel_order("DOTUSDT", "O4")

        # Part B: BE vs DCA race in check_breakeven_on_closed_bar
        cfg = _load_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            journal = VikaJournal(Path(tmp) / "t4.db")
            mock = _AuditMockClient(wallet_balance=2000.0)
            mock.market_prices["DOTUSDT"] = 100.0
            ex = VikaExecutor(cfg, mock, journal, macro_guard=MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json"))
            _attach_specs(ex)
            ex.last_btc_regime = _regime(in_storm=False)

            bar_ts = 1_795_000_000_000
            ex.open_signal(_short_ev("DOTUSDT", "K1_EMA_TREND_PB", bar_ts=bar_ts))
            pos = journal.get_active_positions()[0]
            dca_oid = pos["dca_order_id"]
            full_qty = round(float(pos["qty"]) + float(pos["dca_qty"]), 2)

            # Simulate race: DCA fills right as cancel_order fires -> returns 110001 (False) and exchange avg/size update
            mock.cancel_110001_ids.add(dca_oid)

            def _race_fill(sym, oid):
                if oid == dca_oid:
                    mock.positions["DOTUSDT"]["size"] = str(full_qty)
                    mock.positions["DOTUSDT"]["avgPrice"] = "100.97"

            mock.on_cancel_hook = _race_fill
            moved = ex.check_breakeven_on_closed_bar("DOTUSDT", bar_ts + M15_MS, closed_price=98.90)
            self.assertTrue(moved)

            # Must NOT move SL to old 99.70 and must NOT mark CANCELLED_BE!
            expected_new_lock = round(100.97 * 0.997, 3)  # 100.667
            self.assertIn(("DOTUSDT", expected_new_lock), mock.sl_calls)
            self.assertNotIn(("DOTUSDT", 99.7), mock.sl_calls)
            pos_after = journal.get_active_positions()[0]
            self.assertEqual(pos_after["dca_status"], "FILLED")
            self.assertEqual(pos_after["be_active"], 1)
            self.assertAlmostEqual(float(pos_after["sl_price"]), expected_new_lock, places=3)

    def test_05_high3_rlock_and_isolated_telegram_pybit_session(self) -> None:
        # Part A: Distinct pybit.HTTP per thread (including l2-tg-poll)
        created_sessions: List[MagicMock] = []

        def _fake_pybit(**kwargs):
            m = MagicMock()
            m.kwargs = kwargs
            created_sessions.append(m)
            return m

        with patch("bot.bybit_client.PybitHTTP", side_effect=_fake_pybit):
            cli = BybitClient(mode="demo", api_key="k", api_secret="s")
            main_priv = cli._require_private()
            worker_privs: List[Any] = []

            def _worker():
                worker_privs.append(cli.ensure_thread_private_client())

            t = threading.Thread(target=_worker, name="l2-tg-poll")
            t.start()
            t.join(timeout=2)

        self.assertEqual(len(created_sessions), 2)
        self.assertIsNot(main_priv, worker_privs[0])

        # Part B: Concurrent open_signal() and panic_close_all() serialized by VikaExecutor RLock
        cfg = _load_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            journal = VikaJournal(Path(tmp) / "t5.db")
            mock = _AuditMockClient(wallet_balance=2000.0)
            mock.market_prices["DOTUSDT"] = 100.0
            ex = VikaExecutor(cfg, mock, journal, macro_guard=MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json"))
            _attach_specs(ex)
            ex.last_btc_regime = _regime(in_storm=False)

            entered_mkt = threading.Event()
            orig_place_mkt = ex._place_market_entry_with_sl

            def _slow_mkt(sym, side, qty, sl_price):
                entered_mkt.set()
                time.sleep(0.08)
                return orig_place_mkt(sym, side, qty, sl_price)

            ex._place_market_entry_with_sl = _slow_mkt  # type: ignore[method-assign]
            res_holder: List[Any] = []

            t_open = threading.Thread(target=lambda: res_holder.append(("open", ex.open_signal(_short_ev("DOTUSDT", "K1_EMA_TREND_PB")))))
            t_open.start()
            self.assertTrue(entered_mkt.wait(timeout=2.0))
            t_panic = threading.Thread(target=lambda: res_holder.append(("panic", ex.panic_close_all())), name="l2-tg-poll")
            t_panic.start()
            t_open.join(timeout=5.0)
            t_panic.join(timeout=5.0)

            self.assertTrue(journal.is_paused())
            self.assertEqual(journal.get_active_positions(), [])
            self.assertEqual(mock.get_positions(), [])

    def test_06_high4_wallet_balance_vs_total_equity_margin_and_sizing(self) -> None:
        # Part A: BybitClient separates walletBalance and totalEquity
        cli = BybitClient(mode="demo", api_key="k", api_secret="s")
        fake_priv = MagicMock()
        fake_priv.get_wallet_balance.return_value = {
            "retCode": 0,
            "result": {
                "list": [{
                    "totalEquity": "1850.0",
                    "totalWalletBalance": "2000.0",
                    "totalAvailableBalance": "1600.0",
                    "coin": [{
                        "coin": "USDT",
                        "walletBalance": "2000.0",
                        "equity": "1850.0",
                        "availableToWithdraw": "1600.0",
                        "totalOrderIM": "100.0",
                        "totalPositionIM": "150.0",
                    }],
                }]
            },
        }
        cli._private_client = fake_priv
        cli._auto_private_client = False
        bal = cli.get_wallet_balance("USDT")
        self.assertAlmostEqual(bal["wallet_balance"], 2000.0, places=2)
        self.assertAlmostEqual(bal["total_equity"], 1850.0, places=2)

        # Part B: No double deduction of negative unrealisedPnl from free_base
        cfg = _load_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            journal = VikaJournal(Path(tmp) / "t6.db")
            mock = _AuditMockClient(wallet_balance=2000.0, total_equity=1850.0)
            ex = VikaExecutor(cfg, mock, journal, macro_guard=MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json"))
            _attach_specs(ex)
            ex.last_btc_regime = _regime(in_storm=False)
            ex.last_exchange_positions = {"DOTUSDT": {"symbol": "DOTUSDT", "unrealisedPnl": "-150.0"}}
            ex.refresh_balance()
            snap = ex.get_margin_reserve_snapshot()
            self.assertAlmostEqual(snap["wallet_balance"], 2000.0, places=2)
            self.assertAlmostEqual(snap["total_equity"], 1850.0, places=2)
            self.assertAlmostEqual(snap["free_base"], 1850.0, places=2)

            # Part C: Sizing uses realized walletBalance ($2000 -> $240 full margin), NOT inflated totalEquity ($2500 -> $300)
            mock.wallet_balance = 2000.0
            mock.total_equity = 2500.0
            mock.market_prices["DOTUSDT"] = 100.0
            ex.last_exchange_positions = {}
            pid = ex.open_signal(_short_ev("DOTUSDT", "K1_EMA_TREND_PB"))
            self.assertIsNotNone(pid)
            pos = journal.get_active_positions()[0]
            self.assertAlmostEqual(float(pos["reserved_margin_usd"]), 240.0, delta=3.0)

    def test_07_medium1_cursor_pagination_positions_and_open_orders(self) -> None:
        cli = BybitClient(mode="demo", api_key="k", api_secret="s")
        fake_priv = MagicMock()

        def _make_pages(prefix: str):
            p1 = [{"symbol": f"{prefix}{i}USDT", "size": "1"} for i in range(50)]
            p2 = [{"symbol": f"{prefix}{i}USDT", "size": "1"} for i in range(50, 100)]
            p3 = [{"symbol": f"{prefix}{i}USDT", "size": "1"} for i in range(100, 115)]
            return [
                {"retCode": 0, "result": {"list": p1, "nextPageCursor": "cur2"}},
                {"retCode": 0, "result": {"list": p2, "nextPageCursor": "cur3"}},
                {"retCode": 0, "result": {"list": p3, "nextPageCursor": ""}},
            ]

        fake_priv.get_positions.side_effect = _make_pages("POS")
        fake_priv.get_open_orders.side_effect = _make_pages("ORD")
        cli._private_client = fake_priv
        cli._auto_private_client = False

        pos = cli.get_positions()
        ords = cli.get_open_orders()
        self.assertEqual(len(pos), 115)
        self.assertEqual(len(ords), 115)
        self.assertEqual(fake_priv.get_positions.call_count, 3)
        self.assertEqual(fake_priv.get_open_orders.call_count, 3)

    def test_08_medium2_wal_mode_and_atomic_close_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t8.db"
            journal = VikaJournal(db_path)
            with contextlib.closing(sqlite3.connect(str(db_path))) as raw_conn:
                mode = raw_conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(str(mode).lower(), "wal")

            now_ms = int(time.time() * 1000)
            p1 = journal.add_position(
                symbol="DOTUSDT", side="SHORT", key_name="K1_EMA_TREND_PB", status="OPEN", order_id="",
                entry_price=100.0, qty=10.0, margin_usd=80.0, leverage=15, tp_price=98.5, sl_price=104.0,
                be_trigger_price=99.0, be_lock_price=99.7, entry_bar_ts=1_795_000_000_000, expire_ts=0, filters={},
            )
            # Atomic rollback check: force an OperationalError during kv_state write via SQLite trigger
            with contextlib.closing(sqlite3.connect(str(db_path))) as raw_conn:
                raw_conn.execute(
                    """
                    CREATE TRIGGER fail_ac_write BEFORE INSERT ON kv_state
                    WHEN NEW.k LIKE 'ac_pause_%'
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated_mid_tx_failure');
                    END;
                    """
                )
                raw_conn.commit()

            with self.assertRaises(sqlite3.DatabaseError):
                journal.record_position_exit_atomic(
                    pos_id=p1, symbol="DOTUSDT", side="SHORT", now_ms=now_ms,
                    exit_price=104.0, exit_reason="HARD_SL", pnl_usd=-50.0, pnl_pct=-62.5,
                    cooldown_until_ms=now_ms + 3_600_000, is_hard_sl=True,
                    ac_pause_until_ms=now_ms + 6 * 3_600_000,
                )

            # Position MUST still be OPEN and cooldown MUST NOT be persisted!
            self.assertEqual(len(journal.get_open_positions()), 1)
            self.assertEqual(journal.get_cooldown_until_ms("DOTUSDT", "SHORT"), 0)
            self.assertEqual(journal.get_ac_pause_until_ms("SHORT"), 0)

            with contextlib.closing(sqlite3.connect(str(db_path))) as raw_conn:
                raw_conn.execute("DROP TRIGGER fail_ac_write")
                raw_conn.commit()

            # Now succeed atomically (2nd Hard SL triggers Probation)
            hit1, cnt1, _ = journal.record_position_exit_atomic(
                pos_id=p1, symbol="DOTUSDT", side="SHORT", now_ms=now_ms,
                exit_price=104.0, exit_reason="HARD_SL", pnl_usd=-50.0, pnl_pct=-62.5,
                cooldown_until_ms=now_ms + 3_600_000, is_hard_sl=True,
                ac_pause_until_ms=now_ms + 6 * 3_600_000,
            )
            self.assertFalse(hit1)
            self.assertEqual(cnt1, 1)
            self.assertGreater(journal.get_cooldown_until_ms("DOTUSDT", "SHORT"), now_ms)
            self.assertGreater(journal.get_ac_pause_until_ms("SHORT"), now_ms)

            p2 = journal.add_position(
                symbol="DOTUSDT", side="SHORT", key_name="K1_EMA_TREND_PB", status="OPEN", order_id="",
                entry_price=100.0, qty=10.0, margin_usd=80.0, leverage=15, tp_price=98.5, sl_price=104.0,
                be_trigger_price=99.0, be_lock_price=99.7, entry_bar_ts=1_795_001_000_000, expire_ts=0, filters={},
            )
            hit2, cnt2, prob_until = journal.record_position_exit_atomic(
                pos_id=p2, symbol="DOTUSDT", side="SHORT", now_ms=now_ms + 5000,
                exit_price=104.0, exit_reason="HARD_SL", pnl_usd=-50.0, pnl_pct=-62.5,
                cooldown_until_ms=now_ms + 3_600_000, is_hard_sl=True,
                ac_pause_until_ms=now_ms + 6 * 3_600_000,
            )
            self.assertTrue(hit2)
            self.assertEqual(cnt2, 2)
            self.assertEqual(journal.get_coin_probation_until_ms("DOTUSDT"), prob_until)

    def test_09_low1_telegram_directional_z_label_and_panic_close_real_pnl(self) -> None:
        cfg = _load_cfg()
        with tempfile.TemporaryDirectory() as tmp:
            journal = VikaJournal(Path(tmp) / "t9.db")
            mock = _AuditMockClient(wallet_balance=2000.0)
            mock.market_prices["DOTUSDT"] = 100.0
            ex = VikaExecutor(cfg, mock, journal, macro_guard=MacroCalendarGuard(ROOT / "config" / "macro_events_utc.json"))
            _attach_specs(ex)
            ex.last_btc_regime = _regime(in_storm=False)

            tg = VikaTelegramUI(cfg, ex, journal, get_latest_evals=lambda: {})
            port_txt = tg.format_portfolio()
            welc_txt = tg.format_startup_welcome({"DOTUSDT": "OK (15x Isolated)"})
            self.assertIn("z &gt; +1.0", port_txt)
            self.assertIn("z &lt; -1.0", port_txt)
            self.assertNotIn("|z|≥1.0", port_txt)
            self.assertIn("z &gt; +1.0", welc_txt)
            self.assertIn("z &lt; -1.0", welc_txt)
            self.assertNotIn("|z|", welc_txt)

            # panic_close_all records actual closedPnl instead of 0.0
            ex.open_signal(_short_ev("DOTUSDT", "K1_EMA_TREND_PB"))
            now_ms = int(time.time() * 1000)
            mock.closed_pnl_by_sym["DOTUSDT"] = [
                {"updatedTime": str(now_ms + 1), "closedPnl": "10.25", "closedSize": "6.0", "avgExitPrice": "98.8"},
                {"updatedTime": str(now_ms + 2), "closedPnl": "8.50", "closedSize": "6.6", "avgExitPrice": "99.0"},
            ]
            summary = ex.panic_close_all()
            self.assertIn("PANIC CLOSE ВЫПОЛНЕН", summary)
            closed = journal.get_recent_closed(limit=5)
            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0]["exit_reason"], "PANIC_CLOSE")
            self.assertAlmostEqual(float(closed[0]["pnl_usd"]), 18.75, places=2)
            self.assertGreater(float(closed[0]["pnl_pct"]), 0.0)

    def test_10_kline_limit_600_and_n7_8h_funding_normalization(self) -> None:
        # Part A: scan_all_coins requests limit=600 for BTC 4H, coin 4H, and coin 15m
        cfg = _load_cfg()
        cfg["db_path"] = "data/ai_trader_v2/unused.db"
        fetched_limits: List[tuple] = []

        now_ms = 1_795_000_000_000
        # Build clean synthetic klines
        def _fake_fetch(client, sym, interval, limit=600):
            fetched_limits.append((sym, str(interval), int(limit)))
            iv_ms = {"15": M15_MS, "60": H1_MS, "240": H4_MS}[str(interval)]
            n = max(100, int(limit))
            rows = []
            for k in range(n):
                ts = now_ms - (n - k) * iv_ms
                px = 100.0 + k * 0.02
                rows.append([ts, px, px * 1.003, px * 0.997, px, 1000.0, px * 1000.0])
            return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

        with tempfile.TemporaryDirectory() as tmp:
            cfg_tmp = dict(cfg)
            cfg_tmp["db_path"] = str(Path(tmp) / "scan.db")
            cfg_tmp["cache_dir"] = str(Path(tmp) / "cache")
            cfg_tmp["filters"] = dict(cfg["filters"], retail_ls_enabled=False)
            with patch("ai_trader_v2.run_ai_trader.fetch_recent_klines", side_effect=_fake_fetch):
                runner = VikaRunner(cfg_tmp, dry_run_override=True)
                try:
                    runner.client.get_tickers = MagicMock(return_value={})  # type: ignore[method-assign]
                    runner.scan_all_coins()
                finally:
                    runner.close()

        btc_4h_calls = [x for x in fetched_limits if x[0] == "BTCUSDT" and x[1] == "240"]
        coin_15m_calls = [x for x in fetched_limits if x[0] != "BTCUSDT" and x[1] == "15"]
        coin_4h_calls = [x for x in fetched_limits if x[0] != "BTCUSDT" and x[1] == "240"]
        self.assertTrue(all(x[2] == 600 for x in btc_4h_calls))
        self.assertTrue(all(x[2] == 600 for x in coin_15m_calls) and len(coin_15m_calls) > 0)
        self.assertTrue(all(x[2] == 600 for x in coin_4h_calls) and len(coin_4h_calls) > 0)

        # Part B: N7 8h normalization for 4h-interval coins (ENA, PENGU, PUMPFUN, TAO)
        # Construct downtrending 15m/1H/4H data with a K1_EMA_TREND_PB SHORT trigger on the last closed bar
        n15, n1h, n4h = 260, 120, 120
        base_now = (1_795_000_000_000 // M15_MS) * M15_MS + 1000
        rows4 = [[base_now - (n4h - k) * H4_MS, 200.0 - k * 0.5, 200.5 - k * 0.5, 199.5 - k * 0.5, 200.0 - k * 0.5, 1000.0, 1e5] for k in range(n4h)]
        rows1 = [[base_now - (n1h - k) * H1_MS, 200.0 - k * 0.25, 200.3 - k * 0.25, 199.7 - k * 0.25, 200.0 - k * 0.25, 1000.0, 1e5] for k in range(n1h)]
        rows15 = []
        for k in range(n15 - 3):
            p = 150.0 - k * 0.04 + (0.4 if k % 2 == 0 else -0.4)
            rows15.append([base_now - (n15 - k) * M15_MS, p + 0.05, p + 0.35, p - 0.35, p, 1000.0, 1e5])
        # Push RSI(14) >= 52 on i-1 via 2 strong green bars, then bearish pullback rejecting off EMA21 on last closed bar
        p_prev = rows15[-1][4]
        rows15.append([base_now - 3 * M15_MS, p_prev, p_prev + 1.2, p_prev - 0.1, p_prev + 1.1, 1000.0, 1e5])
        rows15.append([base_now - 2 * M15_MS, p_prev + 1.1, p_prev + 2.5, p_prev + 1.0, p_prev + 2.4, 1000.0, 1e5])
        # Last closed bar: high touches EMA21/50 zone, red close below EMA21
        rows15.append([base_now - 1 * M15_MS, p_prev + 2.3, p_prev + 2.4, p_prev - 0.6, p_prev - 0.4, 1500.0, 1e5])

        cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
        df15 = pd.DataFrame(rows15, columns=cols)
        df1h = pd.DataFrame(rows1, columns=cols)
        df4h = pd.DataFrame(rows4, columns=cols)
        fcfg = dict(cfg["filters"], retail_ls_enabled=False, allowed_directions="only_short", div_4h_strong=False)

        # -1.0 bps per 4h = -2.0 bps per 8h -> MUST block SHORT for 4h coin (ENAUSDT), pass for 8h coin (DOTUSDT)
        ev_4h = evaluate_coin(
            "ENAUSDT", "K1_EMA_TREND_PB", df15, df1h, df4h, -0.00010, base_now, fcfg,
            btc_allow_short=True, funding_interval_hours=4.0,
        )
        ev_8h = evaluate_coin(
            "DOTUSDT", "K1_EMA_TREND_PB", df15, df1h, df4h, -0.00010, base_now, fcfg,
            btc_allow_short=True, funding_interval_hours=8.0,
        )
        self.assertEqual(ev_4h.raw_signal, "SHORT")
        self.assertTrue(ev_4h.n7_blocked_short)
        self.assertAlmostEqual(ev_4h.funding_bps_8h, -2.0, places=2)
        self.assertIsNone(ev_4h.final_signal)

        self.assertEqual(ev_8h.raw_signal, "SHORT")
        self.assertFalse(ev_8h.n7_blocked_short)
        self.assertAlmostEqual(ev_8h.funding_bps_8h, -1.0, places=2)
        self.assertEqual(ev_8h.final_signal, "SHORT")


if __name__ == "__main__":
    unittest.main(verbosity=2)
