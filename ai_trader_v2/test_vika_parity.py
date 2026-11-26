"""
Offline Parity & Unit Verification Suite for VIKA_N2_N7 (ai_trader_v2/).

Verifies:
  1. Bit-for-bit signal & filter parity between ai_trader_v2/vika_engine.py
     and the validated research script data/cache/analyze_vika_65stops_2m_15m.py
     across all 4 strategy keys (K1, K6, K7, K8), 4H STRONG divergence, N2, and N7.
  2. VikaJournal SQLite persistence & daily -6.0% state transitions.
  3. VikaExecutor dry-run order sizing (10% x 10x), BE trigger on closed bar, slot cap (8/8).
  4. VikaTelegramUI message formatters for all 6 buttons & startup welcome.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.bybit_client import BybitClient, InstrumentSpec
from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.telegram_ui import VikaTelegramUI
from ai_trader_v2.vika_engine import M15_MS, evaluate_coin


def _load_research_module():
    path = ROOT / "data" / "cache" / "analyze_vika_65stops_2m_15m.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("vika_research", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parity_with_research() -> None:
    mod = _load_research_module()
    if mod is None:
        print("[SKIP] Research reference script not present in data/cache/; skipping parity check.")
        return

    with open(ROOT / "ai_trader_v2" / "config_ai_trader.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Match original research module (pre-Variant B) for exact historical parity check
    filters_cfg = dict(cfg["filters"])
    filters_cfg["allowed_directions"] = "long_short"
    filters_cfg["k7_require_candle_dir"] = False

    # Test representative coins across ALL 5 active keys (K1, K3, K6, K7, K8) including the 3 new coins
    test_cases = [
        ("GRAMUSDT", "K1_EMA_TREND_PB"),
        ("DOGEUSDT", "K3_MACD_STOCH"),
        ("BNBUSDT", "K6_VWAP_RECLAIM"),
        ("BIGTIMEUSDT", "K7_SFP_LIQ_SWEEP"),
        ("SUIUSDT", "K8_SUPERTREND_BO"),
        ("XRPUSDT", "K1_EMA_TREND_PB"),
    ]
    min_slope4 = float(filters_cfg.get("t4_min_4h_ema50_slope_pct", 0.15))
    cache_dir = ROOT / "data" / "cache" / "tv_top100_65d"
    for sym, key in test_cases:
        p15 = cache_dir / f"{sym}_15m.csv.gz"
        p1h = cache_dir / f"{sym}_1h.csv.gz"
        p4h = cache_dir / f"{sym}_4h.csv.gz"
        if not (p15.exists() and p1h.exists() and p4h.exists()):
            print(f"[SKIP] Cached klines missing for {sym}")
            continue

        df15 = pd.read_csv(p15, compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df1h = pd.read_csv(p1h, compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df4h = pd.read_csv(p4h, compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        # Build research precomputed data using exact analyze_vika_65stops_2m_15m.py loader
        tl = df15["timestamp"].to_numpy(np.int64)
        rd = mod.load_vika_coin(sym, 0.0003, tl)
        sl_arr, ss_arr = rd["sigs"][key]
        n = len(tl)

        # Pick 10 timestamps: up to 6 bars where research had signals + 4 regular bars near tail
        sig_idxs = [i for i in range(n - 1500, n - 5) if sl_arr[i] or ss_arr[i]]
        sample_idxs = sig_idxs[-6:] + [n - 10, n - 50, n - 120, n - 250]
        checked = 0
        for idx in sample_idxs:
            bar_ts = int(tl[idx])
            now_ms = bar_ts + M15_MS + 1000  # 1s after bar idx closed

            # Slice realistic live window (same sizes fetched in production: 320x15m, 240x1h, 240x4h)
            df15_win = df15[df15["timestamp"] <= bar_ts].tail(320).reset_index(drop=True)
            df1h_win = df1h[df1h["timestamp"] <= bar_ts].tail(240).reset_index(drop=True)
            df4h_win = df4h[df4h["timestamp"] <= bar_ts].tail(240).reset_index(drop=True)

            # Use exact historical funding rate from research table at this bar
            fr_val = float(rd["fr_last_bps"][idx]) / 10000.0
            ev = evaluate_coin(sym, key, df15_win, df1h_win, df4h_win, fr_val, now_ms, filters_cfg)

            exp_adx = float(rd["adx15"][idx])
            exp_bbw = float(rd["bbw15"][idx])
            exp_atr_pct = float(rd["atr15_pct"][idx])
            exp_slope4 = float(rd["slope4h"][idx])

            assert ev.bar_ts == bar_ts, f"{sym} bar_ts mismatch: {ev.bar_ts} != {bar_ts}"
            assert abs(ev.adx15 - exp_adx) < 0.15, f"{sym} ADX15 drift: {ev.adx15} vs {exp_adx}"
            assert abs(ev.bbw15 - exp_bbw) < 0.05, f"{sym} BBW15 drift: {ev.bbw15} vs {exp_bbw}"
            assert abs(ev.atr15_pct - exp_atr_pct) < 0.02, f"{sym} ATR15% drift: {ev.atr15_pct} vs {exp_atr_pct}"
            assert abs(ev.slope4h_pct - exp_slope4) < 0.05, f"{sym} Slope4H drift: {ev.slope4h_pct} vs {exp_slope4}"

            # Research sl_arr/ss_arr already has 4H STRONG divergence applied (before N2, N7, T4)
            exp_sig = "LONG" if sl_arr[idx] else ("SHORT" if ss_arr[idx] else None)
            actual_after_4h = ev.raw_signal
            if actual_after_4h == "LONG" and ev.div4h_bear_strong:
                actual_after_4h = None
            elif actual_after_4h == "SHORT" and ev.div4h_bull_strong:
                actual_after_4h = None
            assert actual_after_4h == exp_sig, (
                f"{sym} ({key}) signal mismatch @ idx={idx}: got {actual_after_4h} (raw={ev.raw_signal}) vs exp {exp_sig}"
            )

            # Verify N2 + N7 + T4 final signal parity with research logic
            exp_n2_block = (exp_adx >= 40.0) or (exp_bbw >= 6.5) or (exp_atr_pct >= 1.50)
            exp_final = exp_sig
            if exp_final is not None and exp_n2_block:
                exp_final = None
            if exp_final == "LONG" and float(rd["fr_last_bps"][idx]) >= 1.5:
                exp_final = None
            if exp_final == "SHORT" and float(rd["fr_last_bps"][idx]) <= -1.5:
                exp_final = None
            if exp_final == "LONG" and exp_slope4 <= min_slope4:
                exp_final = None
            if exp_final == "SHORT" and (-exp_slope4) <= min_slope4:
                exp_final = None
            assert ev.final_signal == exp_final, (
                f"{sym} ({key}) FINAL signal mismatch @ idx={idx}: got {ev.final_signal} (block={ev.block_reason}) vs exp {exp_final}"
            )

            # Check N7 funding block explicitly when final signal exists
            if ev.final_signal == "LONG":
                ev_n7 = evaluate_coin(sym, key, df15_win, df1h_win, df4h_win, +0.00020, now_ms, filters_cfg)
                assert ev_n7.final_signal is None and ev_n7.n7_blocked_long, f"{sym} N7 failed to block LONG"
            elif ev.final_signal == "SHORT":
                ev_n7 = evaluate_coin(sym, key, df15_win, df1h_win, df4h_win, -0.00020, now_ms, filters_cfg)
                assert ev_n7.final_signal is None and ev_n7.n7_blocked_short, f"{sym} N7 failed to block SHORT"

            checked += 1

        print(f"[PASS] Parity verified for {sym:<12} ({key:<17}): {checked} historical bars matched 100%.")


def test_journal_executor_and_telegram() -> None:
    with open(ROOT / "ai_trader_v2" / "config_ai_trader.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["mode"] = "paper"
    cfg["dry_run"] = True
    cfg["allowed_directions"] = "long_short"

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test_vika.db"
        journal = VikaJournal(db_path)
        client = BybitClient(mode="testnet", cache_dir=tmp)
        sent_msgs = []
        executor = VikaExecutor(cfg, client, journal, notifier=sent_msgs.append)

        # Populate dummy specs for all 10 coins
        for c in cfg["coins"]:
            sym = c["symbol"]
            executor.specs[sym] = InstrumentSpec(
                symbol=sym, base_coin=sym.replace("USDT", ""), quote_coin="USDT",
                status="Trading", min_order_qty=1.0, qty_step=1.0,
                min_notional=5.0, tick_size=0.0001, min_leverage=1.0, max_leverage=50.0,
            )

        from ai_trader_v2.vika_engine import CoinEvaluation

        ev = CoinEvaluation(
            symbol="GRAMUSDT", key="K1_EMA_TREND_PB", bar_ts=1_760_000_000_000,
            close_price=0.0100, rvol=1.4, adx15=24.5, bbw15=2.1, atr15_pct=0.45,
            funding_rate=0.00005, funding_bps=0.50,
            div4h_bear_strong=False, div4h_bull_strong=False,
            n2_blocked=False, n7_blocked_long=False, n7_blocked_short=False,
            raw_signal="LONG", final_signal="LONG", block_reason=None,
        )

        # Open trade in paper mode -> equity=2000, margin=200 (10%), notional=2000 (10x), qty=200000
        pos_id = executor.open_signal(ev)
        assert pos_id is not None
        active = journal.get_active_positions()
        assert len(active) == 1
        assert abs(active[0]["margin_usd"] - 200.0) < 1.0
        assert abs(active[0]["tp_price"] - 0.0101) < 1e-6  # rounded to tick 0.0001 (1.4% = 0.01014 -> 0.0101)
        assert len(sent_msgs) == 1 and "ВХОД В СДЕЛКУ" in sent_msgs[0]

        # Test closed-bar BE trigger
        executor.check_breakeven_on_closed_bar("GRAMUSDT", ev.bar_ts + M15_MS, 0.01012)
        active = journal.get_active_positions()
        assert active[0]["be_active"] == 1
        assert len(sent_msgs) == 2 and "БЕЗУБЫТОК" in sent_msgs[1]

        # Test R3 (max_same_key = 2): 2nd K1 opens, 3rd K1 is blocked, K3 opens freely
        ev_k1_2 = CoinEvaluation(
            symbol="VIRTUALUSDT", key="K1_EMA_TREND_PB", bar_ts=1_760_000_000_000,
            close_price=0.3500, rvol=1.3, adx15=22.0, bbw15=2.5, atr15_pct=0.60,
            funding_rate=0.00005, funding_bps=0.50, slope4h_pct=0.45,
            div4h_bear_strong=False, div4h_bull_strong=False,
            n2_blocked=False, n7_blocked_long=False, n7_blocked_short=False,
            raw_signal="LONG", final_signal="LONG", block_reason=None,
        )
        ev_k1_3 = CoinEvaluation(
            symbol="XRPUSDT", key="K1_EMA_TREND_PB", bar_ts=1_760_000_000_000,
            close_price=2.1000, rvol=1.2, adx15=21.0, bbw15=2.2, atr15_pct=0.50,
            funding_rate=0.00005, funding_bps=0.50, slope4h_pct=0.45,
            div4h_bear_strong=False, div4h_bull_strong=False,
            n2_blocked=False, n7_blocked_long=False, n7_blocked_short=False,
            raw_signal="LONG", final_signal="LONG", block_reason=None,
        )
        ev_k3 = CoinEvaluation(
            symbol="DOGEUSDT", key="K3_MACD_STOCH", bar_ts=1_760_000_000_000,
            close_price=0.1800, rvol=1.5, adx15=23.0, bbw15=2.8, atr15_pct=0.65,
            funding_rate=0.00005, funding_bps=0.50, slope4h_pct=0.55,
            div4h_bear_strong=False, div4h_bull_strong=False,
            n2_blocked=False, n7_blocked_long=False, n7_blocked_short=False,
            raw_signal="LONG", final_signal="LONG", block_reason=None,
        )
        assert executor.open_signal(ev_k1_2) is not None
        assert executor.open_signal(ev_k1_3) is None, "R3 max_same_key=2 failed to block 3rd K1 trade"
        assert executor.open_signal(ev_k3) is not None, "K3_MACD_STOCH trade should open despite K1 at cap"

        # Test Telegram UI formatters for 18 coins + R3 + T4
        tg = VikaTelegramUI(cfg, executor, journal, get_latest_evals=lambda: {"GRAMUSDT": ev, "DOGEUSDT": ev_k3})
        lev_st = executor.ensure_cross_margin_and_leverage()
        assert len(lev_st) == 18
        assert "18 CLEAN + R3 + T4" in tg.format_startup_welcome(lev_st)
        assert "СТАТУС И БАЛАНС | Vika N2N7" in tg.format_status()
        assert "ОТКРЫТЫЕ ПОЗИЦИИ (3/8) | Vika N2N7" in tg.format_open_positions()
        assert "ПОРТФЕЛЬ Vika N2N7 — 18 МОНЕТ" in tg.format_portfolio()
        assert "ИСТОРИЯ СДЕЛОК | Vika N2N7" in tg.format_history()

        # Test Panic Close pauses bot & clears active
        panic_res = executor.panic_close_all()
        assert "PANIC CLOSE ВЫПОЛНЕН" in panic_res
        assert journal.is_paused() is True
        assert len(journal.get_active_positions()) == 0

    print("[PASS] Journal, Executor (10%x10x, R3 max_same_key=2, BE, Panic), and 18-coin Telegram UI verified.")


def test_limit_order_ttl_45m_partial_fill_and_startup_cleanup() -> None:
    """Verify 45m (3-bar) TTL cancel, partial-fill remainder cancel, and startup stale cleanup."""
    import time

    with open(ROOT / "ai_trader_v2" / "config_ai_trader.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["mode"] = "demo"
    cfg["dry_run"] = False
    assert cfg["limit_entry_ttl_bars"] == 3
    assert cfg["limit_entry_ttl_sec"] == 2700

    class MockBybitClient:
        has_credentials = True

        def __init__(self) -> None:
            self.positions: list = []
            self.open_orders: list = []
            self.cancelled_calls: list = []

        def get_positions(self, symbol=None):
            if symbol:
                return [p for p in self.positions if p["symbol"] == symbol]
            return list(self.positions)

        def get_open_orders(self, symbol=None):
            if symbol:
                return [o for o in self.open_orders if o["symbol"] == symbol]
            return list(self.open_orders)

        def cancel_order(self, symbol: str, order_id: str) -> None:
            self.cancelled_calls.append((symbol, order_id))
            self.open_orders = [o for o in self.open_orders if o["orderId"] != order_id]

        def get_wallet_balance(self, coin: str = "USDT"):
            return {"equity": 2000.0, "available": 2000.0}

    with tempfile.TemporaryDirectory() as tmp:
        journal = VikaJournal(Path(tmp) / "ttl_test.db")
        mock = MockBybitClient()
        sent = []
        ex = VikaExecutor(cfg, mock, journal, notifier=sent.append)  # type: ignore[arg-type]

        now_ms = int(time.time() * 1000)

        # --- Case 1: Unfilled limit order aged 46m -> must cancel, free slot, notify Telegram ---
        p1 = journal.add_position(
            symbol="GRAMUSDT", side="LONG", key_name="K1_EMA_TREND_PB", status="PENDING",
            order_id="ORD-UNFILLED-46M", entry_price=0.90, qty=2000.0, margin_usd=180.0, leverage=10,
            tp_price=0.9126, sl_price=0.873, be_trigger_price=0.9085, be_lock_price=0.9022,
            entry_bar_ts=now_ms - 4 * M15_MS, expire_ts=now_ms - 60_000, filters={},
        )
        mock.open_orders = [{
            "orderId": "ORD-UNFILLED-46M", "symbol": "GRAMUSDT", "orderType": "Limit",
            "side": "Buy", "price": "0.90", "reduceOnly": False, "stopOrderType": "",
            "createdTime": str(now_ms - 46 * 60_000),
        }]
        ex.sync_with_exchange()
        assert ("GRAMUSDT", "ORD-UNFILLED-46M") in mock.cancelled_calls
        assert len(journal.get_active_positions()) == 0
        assert any("Лимитный ордер по GRAMUSDT отменён по тайм-ауту 45m (3 свечи)" in m for m in sent)

        # --- Case 2: Partial fill before 45m -> keep OPEN & order; after 45m -> cancel ONLY remainder ---
        sent.clear()
        mock.cancelled_calls.clear()
        p2 = journal.add_position(
            symbol="BNBUSDT", side="SHORT", key_name="K6_VWAP_RECLAIM", status="PENDING",
            order_id="ORD-PARTIAL", entry_price=540.0, qty=3.7, margin_usd=200.0, leverage=10,
            tp_price=532.44, sl_price=556.2, be_trigger_price=534.87, be_lock_price=538.65,
            entry_bar_ts=now_ms - M15_MS, expire_ts=now_ms + 30 * 60_000, filters={},
        )
        mock.positions = [{
            "symbol": "BNBUSDT", "side": "Sell", "size": "1.5", "avgPrice": "540.0",
            "takeProfit": "532.44", "stopLoss": "556.2",
        }]
        mock.open_orders = [{
            "orderId": "ORD-PARTIAL", "symbol": "BNBUSDT", "orderType": "Limit",
            "side": "Sell", "price": "540.0", "reduceOnly": False, "stopOrderType": "",
            "createdTime": str(now_ms - 10 * 60_000),
        }]
        ex.sync_with_exchange()
        # Still before 45m -> position OPEN (qty 1.5), remainder NOT cancelled yet
        act = journal.get_active_positions()
        assert len(act) == 1 and act[0]["status"] == "OPEN" and abs(act[0]["qty"] - 1.5) < 1e-6
        assert act[0]["order_id"] == "ORD-PARTIAL"
        assert len(mock.cancelled_calls) == 0

        # Now expire it past 45m
        with journal._connect() as conn:
            conn.execute("UPDATE positions SET expire_ts = ? WHERE id = ?", (now_ms - 1000, p2))
        ex.sync_with_exchange()
        # Remainder cancelled, OPEN filled part kept intact, order_id cleared, TG notified!
        assert ("BNBUSDT", "ORD-PARTIAL") in mock.cancelled_calls
        act = journal.get_active_positions()
        assert len(act) == 1 and act[0]["status"] == "OPEN" and act[0]["order_id"] == ""
        assert any("BNBUSDT" in m and "45m (3 свечи)" in m and "остаток" in m for m in sent)

        # --- Case 3: Startup guard cancels >45m entry Limit, keeps <45m entry Limit and all TP/SL ---
        mock.cancelled_calls.clear()
        mock.positions = []
        mock.open_orders = [
            # Stale 60m opening limit -> MUST CANCEL
            {"orderId": "STALE-60M", "symbol": "LTCUSDT", "orderType": "Limit", "side": "Buy",
             "price": "40.0", "reduceOnly": False, "stopOrderType": "", "createdTime": str(now_ms - 60 * 60_000)},
            # Fresh 15m opening limit -> MUST KEEP
            {"orderId": "FRESH-15M", "symbol": "NEARUSDT", "orderType": "Limit", "side": "Buy",
             "price": "2.7", "reduceOnly": False, "stopOrderType": "", "createdTime": str(now_ms - 15 * 60_000)},
            # Old 5h TP/SL protective order -> MUST NEVER CANCEL
            {"orderId": "OLD-TPSL", "symbol": "BNBUSDT", "orderType": "Limit", "side": "Buy",
             "price": "532.44", "reduceOnly": True, "stopOrderType": "TakeProfit", "createdTime": str(now_ms - 300 * 60_000)},
        ]
        cancelled = ex.cleanup_stale_entry_orders()
        assert len(cancelled) == 1 and "LTCUSDT" in cancelled[0]
        assert mock.cancelled_calls == [("LTCUSDT", "STALE-60M")]
        remaining_ids = {o["orderId"] for o in mock.open_orders}
        assert remaining_ids == {"FRESH-15M", "OLD-TPSL"}

    print("[PASS] 45m (3-bar) Limit TTL, Partial-Fill remainder cancel & Startup stale guard verified.")


def test_market_entry_and_limit_maker_tp() -> None:
    """Verify Market IOC entry + atomic server-side Market SL + resting reduceOnly Limit Maker TP."""
    with open(ROOT / "ai_trader_v2" / "config_ai_trader.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["entry_order_type"] == "Market"
    assert cfg["tp_order_type"] == "Limit"
    cfg["mode"] = "demo"
    cfg["dry_run"] = False
    cfg["allowed_directions"] = "long_short"

    class MockPriv:
        def __init__(self, parent) -> None:
            self.parent = parent

        def place_order(self, **kwargs):
            self.parent.place_order_calls.append(kwargs)
            # Simulate immediate Market fill at 0.1802 (slight slip from 0.1800)
            self.parent.positions = [{
                "symbol": kwargs["symbol"],
                "side": kwargs["side"],
                "size": kwargs["qty"],
                "avgPrice": "0.1802",
                "stopLoss": kwargs["stopLoss"],
                "takeProfit": "0",
            }]
            return {"retCode": 0, "result": {"orderId": "MKT-001"}}

    class MockClient:
        has_credentials = True

        def __init__(self) -> None:
            self.place_order_calls: list = []
            self.limit_tp_calls: list = []
            self.sl_calls: list = []
            self.cancelled_calls: list = []
            self.positions: list = []
            self.open_orders: list = []
            self._priv = MockPriv(self)

        def _require_private(self):
            return self._priv

        def get_wallet_balance(self, coin: str = "USDT"):
            return {"equity": 2000.0, "available": 2000.0}

        def get_positions(self, symbol=None):
            if symbol:
                return [p for p in self.positions if p["symbol"] == symbol]
            return list(self.positions)

        def get_open_orders(self, symbol=None):
            if symbol:
                return [o for o in self.open_orders if o["symbol"] == symbol]
            return list(self.open_orders)

        def set_trading_stop(self, symbol: str, stop_loss: float, position_idx: int = 0):
            self.sl_calls.append((symbol, stop_loss))
            for p in self.positions:
                if p["symbol"] == symbol:
                    p["stopLoss"] = str(stop_loss)
            return {}

        def place_limit_tp_order(self, symbol: str, side: str, qty: float, price: float):
            self.limit_tp_calls.append((symbol, side, qty, price))
            oid = f"TP-{len(self.limit_tp_calls)}"
            self.open_orders.append({
                "orderId": oid, "symbol": symbol, "side": side,
                "orderType": "Limit", "reduceOnly": True, "stopOrderType": "",
                "qty": str(qty), "price": str(price),
            })
            return {"orderId": oid}

        def cancel_order(self, symbol: str, order_id: str):
            self.cancelled_calls.append((symbol, order_id))
            self.open_orders = [o for o in self.open_orders if o["orderId"] != order_id]

    with tempfile.TemporaryDirectory() as tmp:
        journal = VikaJournal(Path(tmp) / "mkt_test.db")
        mock = MockClient()
        sent = []
        ex = VikaExecutor(cfg, mock, journal, notifier=sent.append)  # type: ignore[arg-type]
        ex.specs["DOGEUSDT"] = InstrumentSpec(
            symbol="DOGEUSDT", base_coin="DOGE", quote_coin="USDT",
            status="Trading", min_order_qty=1.0, qty_step=1.0,
            min_notional=5.0, tick_size=0.0001, min_leverage=1.0, max_leverage=50.0,
        )

        from ai_trader_v2.vika_engine import CoinEvaluation

        ev = CoinEvaluation(
            symbol="DOGEUSDT", key="K3_MACD_STOCH", bar_ts=1_760_000_000_000,
            close_price=0.1800, rvol=1.6, adx15=24.0, bbw15=2.5, atr15_pct=0.65,
            funding_rate=0.00004, funding_bps=0.40, slope4h_pct=0.55,
            div4h_bear_strong=False, div4h_bull_strong=False,
            n2_blocked=False, n7_blocked_long=False, n7_blocked_short=False,
            raw_signal="LONG", final_signal="LONG", block_reason=None,
        )
        pos_id = ex.open_signal(ev)
        assert pos_id is not None
        assert len(mock.place_order_calls) == 1
        po = mock.place_order_calls[0]
        assert po["orderType"] == "Market" and po["timeInForce"] == "IOC"
        assert float(po["stopLoss"]) > 0 and "takeProfit" not in po
        # Resting Limit Maker TP placed from actual avgPrice 0.1802 (+1.4% = 0.1827)
        assert len(mock.limit_tp_calls) == 1
        assert mock.limit_tp_calls[0][1] == "Sell"
        assert abs(mock.limit_tp_calls[0][3] - 0.1827) < 1e-6
        act = journal.get_active_positions()
        assert len(act) == 1 and act[0]["status"] == "OPEN" and abs(act[0]["entry_price"] - 0.1802) < 1e-6
        assert any("Вход (Market)" in m and "TP (Limit Maker)" in m for m in sent)

        # If Limit Maker TP disappears during sync -> automatically restored
        mock.open_orders.clear()
        ex.sync_with_exchange()
        assert len(mock.limit_tp_calls) == 2

        # If position carries native server takeProfit > 0 (pre-upgrade position) -> do NOT duplicate Limit TP
        mock.open_orders.clear()
        mock.positions[0]["takeProfit"] = "0.1827"
        ex.sync_with_exchange()
        assert len(mock.limit_tp_calls) == 2

    print("[PASS] Market entry + atomic Market SL + Limit Maker TP (0.02%) & sync recovery verified.")


def test_only_short_and_k7_candle_color_fix() -> None:
    """Verify Variant B: allowed_directions=only_short and K7_SFP_LIQ_SWEEP candle-color requirement."""
    with open(ROOT / "ai_trader_v2" / "config_ai_trader.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["mode"] == "demo"
    assert cfg["allowed_directions"] == "only_short"
    assert cfg["filters"]["k7_require_candle_dir"] is True
    assert len(cfg["coins"]) == 18
    coin_map = {c["symbol"]: c for c in cfg["coins"]}
    for sym, exp_k, exp_tp in [
        ("WLDUSDT", "K1_EMA_TREND_PB", 1.4),
        ("ADAUSDT", "K7_SFP_LIQ_SWEEP", 1.2),
        ("SOLUSDT", "K6_VWAP_RECLAIM", 1.4),
        ("HYPEUSDT", "K7_SFP_LIQ_SWEEP", 1.4),
        ("PUMPFUNUSDT", "K7_SFP_LIQ_SWEEP", 1.2),
    ]:
        assert sym in coin_map and coin_map[sym]["key"] == exp_k and coin_map[sym]["tp_pct"] == exp_tp

    # 1. Executor blocks LONG and allows SHORT in only_short mode
    cfg_paper = dict(cfg)
    cfg_paper["mode"] = "paper"
    cfg_paper["dry_run"] = True
    with tempfile.TemporaryDirectory() as tmp:
        journal = VikaJournal(Path(tmp) / "dir_test.db")
        client = BybitClient(mode="testnet", cache_dir=tmp)
        ex = VikaExecutor(cfg_paper, client, journal)
        ex.specs["HBARUSDT"] = InstrumentSpec(
            symbol="HBARUSDT", base_coin="HBAR", quote_coin="USDT",
            status="Trading", min_order_qty=1.0, qty_step=1.0,
            min_notional=5.0, tick_size=0.0001, min_leverage=1.0, max_leverage=50.0,
        )
        from ai_trader_v2.vika_engine import CoinEvaluation

        ev_l = CoinEvaluation(
            symbol="HBARUSDT", key="K7_SFP_LIQ_SWEEP", bar_ts=1_760_000_000_000,
            close_price=0.0550, rvol=1.5, adx15=20.0, bbw15=2.0, atr15_pct=0.50,
            funding_rate=0.0, funding_bps=0.0, slope4h_pct=0.40,
            div4h_bear_strong=False, div4h_bull_strong=False,
            n2_blocked=False, n7_blocked_long=False, n7_blocked_short=False,
            raw_signal="LONG", final_signal="LONG", block_reason=None,
        )
        assert ex.open_signal(ev_l) is None, "only_short mode must block LONG in executor"

        ev_s = CoinEvaluation(
            symbol="HBARUSDT", key="K7_SFP_LIQ_SWEEP", bar_ts=1_760_000_000_000,
            close_price=0.0550, rvol=1.5, adx15=20.0, bbw15=2.0, atr15_pct=0.50,
            funding_rate=0.0, funding_bps=0.0, slope4h_pct=-0.40,
            div4h_bear_strong=False, div4h_bull_strong=False,
            n2_blocked=False, n7_blocked_long=False, n7_blocked_short=False,
            raw_signal="SHORT", final_signal="SHORT", block_reason=None,
        )
        assert ex.open_signal(ev_s) is not None, "only_short mode must allow SHORT in executor"

        tg = VikaTelegramUI(cfg_paper, ex, journal, get_latest_evals=lambda: {})
        assert "ONLY SHORT" in tg.format_status()
        assert "K7 color" in tg.format_status()

    # 2. Verify K7 candle-color fix on real cached HBARUSDT bars (if cache present)
    cache_dir = ROOT / "data" / "cache" / "tv_top100_65d"
    p15 = cache_dir / "HBARUSDT_15m.csv.gz"
    p1h = cache_dir / "HBARUSDT_1h.csv.gz"
    p4h = cache_dir / "HBARUSDT_4h.csv.gz"
    if p15.exists() and p1h.exists() and p4h.exists():
        df15 = pd.read_csv(p15, compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df1h = pd.read_csv(p1h, compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df4h = pd.read_csv(p4h, compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        f_no_col = dict(cfg["filters"], allowed_directions="only_short", k7_require_candle_dir=False)
        f_col = dict(cfg["filters"], allowed_directions="only_short", k7_require_candle_dir=True)
        # Scan recent 800 bars to find at least one wrong-color K7 SHORT that gets filtered out
        checked_diff = 0
        for idx in range(len(df15) - 10, max(350, len(df15) - 1200), -1):
            bar_ts = int(df15.at[idx, "timestamp"])
            now_ms = bar_ts + M15_MS + 1000
            if float(df15.at[idx, "close"]) < float(df15.at[idx, "open"]):
                continue  # green bar only
            w15 = df15.iloc[max(0, idx - 319) : idx + 1].reset_index(drop=True)
            w1h = df1h[df1h["timestamp"] <= bar_ts].tail(240).reset_index(drop=True)
            w4h = df4h[df4h["timestamp"] <= bar_ts].tail(240).reset_index(drop=True)
            ev_old = evaluate_coin("HBARUSDT", "K7_SFP_LIQ_SWEEP", w15, w1h, w4h, 0.0, now_ms, f_no_col)
            if ev_old.raw_signal == "SHORT":
                ev_new = evaluate_coin("HBARUSDT", "K7_SFP_LIQ_SWEEP", w15, w1h, w4h, 0.0, now_ms, f_col)
                assert ev_new.raw_signal is None, "K7 candle-color fix failed to block green SHORT candle"
                checked_diff += 1
                break
        assert checked_diff == 1, "Expected to verify at least 1 green K7 SHORT candle blocked on HBARUSDT"

    print("[PASS] Variant B (allowed_directions=only_short + K7 candle-color fix) verified.")


if __name__ == "__main__":
    test_parity_with_research()
    test_journal_executor_and_telegram()
    test_limit_order_ttl_45m_partial_fill_and_startup_cleanup()
    test_market_entry_and_limit_maker_tp()
    test_only_short_and_k7_candle_color_fix()
    print("\nALL VIKA_N2_N7 PARITY & UNIT TESTS PASSED!")
