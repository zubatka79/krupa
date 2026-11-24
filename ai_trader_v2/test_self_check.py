"""
Comprehensive offline end-to-end verification for AI-Trader v2:
  1) Config bounds (29 coins, 12 schools, Demo mode, 8/5 limits, 1H shorts, RVOL S12, 3-stage BE, AI provider)
  2) AI Guardrails, configurable provider priority (Gemini vs Anthropic), and accurate model pricing rates
  3) Shorts on 1H without requiring 4H + S12 RVOL >= 1.2 volume gate
  4) 3-Stage Smart Break-Even (closed >=1.3R -> -0.5R; TP1 -> -0.35R; closed >=2.3R -> +0.3R + 4.0*ATR trailing)
  5) Daily -6.0% stop WITHOUT double-counting uPnL on Bybit Demo/Exchange
  6) Partial fill in PENDING_LIMIT cancels remainder limit order and resizes TP1/TP2 to exact filled pos_size
  7) Headless in-memory PNG chart rendering (valid PNG signature, zero temp files, safe fallback)
"""
from __future__ import annotations

import base64
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_trader_v2.ai_judge import AITradeJudge, get_model_rates_per_m
from ai_trader_v2.chart_renderer import render_15m_chart_png_b64
from ai_trader_v2.executor import AITradeExecutor, TelegramNotifier
from ai_trader_v2.journal import AITradeJournal
from ai_trader_v2.run_ai_trader import load_config
from ai_trader_v2.ta_schools import ALL_12_SCHOOLS, BtcMarketContext, CandidateSetup, evaluate_symbol_12_schools
from bot.bybit_client import BybitClient, InstrumentSpec


def _make_synth_klines(n: int, start_price: float, trend_step: float, interval_ms: int, vol_last_mult: float = 1.6) -> pd.DataFrame:
    base_ts = int(time.time() * 1000) - n * interval_ms
    ts = np.array([base_ts + i * interval_ms for i in range(n)], dtype=np.int64)
    cl = np.array([start_price + i * trend_step for i in range(n)], dtype=float)
    op = np.roll(cl, 1)
    op[0] = start_price
    hi = np.maximum(op, cl) * 1.004
    lo = np.minimum(op, cl) * 0.996
    vol = np.full(n, 1000.0, dtype=float)
    vol[-1] = 1000.0 * vol_last_mult
    return pd.DataFrame({"timestamp": ts, "open": op, "high": hi, "low": lo, "close": cl, "volume": vol, "turnover": vol * cl})


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Config & Limits (8 / 5, Demo, 1H shorts, RVOL S12, 3-stage BE)
    # ------------------------------------------------------------------
    cfg = load_config()
    assert len(cfg["symbols"]) == 29
    assert len(cfg["schools_enabled"]) == 12 and set(cfg["schools_enabled"]) == set(ALL_12_SCHOOLS)
    assert cfg["mode"] == "demo"
    assert cfg["max_open_positions"] == 8 and cfg["max_same_direction"] == 5
    assert cfg["long_require_4h"] is True and cfg["short_require_4h"] is False
    assert cfg["min_rvol_s12"] == 1.2 and cfg["min_schools_non_s12"] == 2 and cfg["min_rvol_non_s12"] == 1.2
    assert cfg["be_stage1_close_r"] == 1.3 and cfg["be_stage1_sl_r"] == -0.5
    assert cfg["be_stage2_tp1_sl_r"] == -0.35
    assert cfg["be_stage3_close_r"] == 2.3 and cfg["be_stage3_sl_r"] == 0.3
    assert cfg["trailing_atr_mult"] == 4.0
    assert cfg["entry_order_type"] == "limit" and cfg["limit_entry_ttl_minutes"] == 45
    assert "ai_provider" in cfg and "ai_model" in cfg
    print("[PASS] 1. Config validated (Demo, 29 coins, 12 schools, max 8/5, 1H shorts, 3-stage BE, RVOL S12)")

    # ------------------------------------------------------------------
    # 2. AI Guardrails + Provider Selector + Accurate Billing Rates
    # ------------------------------------------------------------------
    j_gem = AITradeJudge(monthly_limit_usd=50.0, provider="gemini", model="gemini-2.5-flash")
    j_gem.anthropic_key = "sk-ant-fake-key-for-order-test"
    j_gem.gemini_key = "AIzaFakeGeminiKeyForOrderTest123456"
    assert j_gem._provider_order()[0] == "gemini"
    assert j_gem.active_provider_label.startswith("Gemini (gemini-2.5-flash")
    assert get_model_rates_per_m("anthropic", "claude-opus-5-5") == (5.0, 25.0)
    assert get_model_rates_per_m("anthropic", "claude-sonnet-4-20250514") == (3.0, 15.0)
    assert get_model_rates_per_m("gemini", "gemini-2.5-flash") == (0.15, 0.60)
    assert get_model_rates_per_m("gemini", "gemini-2.5-pro") == (1.25, 10.0)

    cand = CandidateSetup(
        symbol="ETHUSDT", direction="LONG", bar_ts=1764000000000, close_price=3000.0,
        atr_14=32.0, atr_pct=1.066, rvol=1.6, adx_15m=26.0, rsi_15m=58.0,
        fired_schools=["S12_Deriv_OI_Funding", "S07_Ichimoku_Kumo"],
        primary_school="S12_Deriv_OI_Funding", priority_score=58.0, regime_label="STRONG_TREND",
        sl_candidates_pct={"swing_10_sl_pct": 1.35}, default_sl_pct=1.35,
        recommended_tp1_r=1.8, recommended_tp2_r=3.5,
        orderbook={"spread_bps": 0.2, "depth_03_usdt": 5000000.0, "best_bid": 2999.9, "best_ask": 3000.1},
    )
    dec_clamped = AITradeJudge.validate_and_clamp(
        {"decision": "ENTER", "confidence": 85, "sl_pct": 4.5, "tp1_r": 1.8, "tp2_r": 3.5, "use_trailing_after_tp1": False, "reason_ru": "OK"},
        cand, cfg, "test",
    )
    assert dec_clamped.decision == "ENTER" and dec_clamped.sl_pct == 1.6
    dec_low = AITradeJudge.validate_and_clamp(
        {"decision": "ENTER", "confidence": 60, "sl_pct": 1.4, "tp1_r": 1.8, "tp2_r": 3.2, "reason_ru": "Weak"},
        cand, cfg, "test",
    )
    assert dec_low.decision == "SKIP"
    print("[PASS] 2. AI Guardrails, provider switching (Gemini/Claude) & per-model billing rates verified")

    # ------------------------------------------------------------------
    # 3. Shorts on 1H WITHOUT 4H (4H=BULL, 1H=BEAR) + S12 RVOL >= 1.2 Gate
    # ------------------------------------------------------------------
    df15_dn = _make_synth_klines(240, 200.0, -0.25, 900_000, vol_last_mult=1.5)
    df15_dn.loc[239, "close"] = df15_dn["low"].iloc[231:239].min() * 0.995  # clear 8-bar breakdown for S12
    df15_dn.loc[239, "low"] = df15_dn.loc[239, "close"] * 0.998
    df1h_bear = _make_synth_klines(80, 200.0, -0.5, 3_600_000)
    df4h_bull = _make_synth_klines(80, 100.0, +0.8, 14_400_000)  # 4H is BULL while 1H is BEAR!

    mock_cli = MagicMock()
    mock_cli.get_klines.side_effect = lambda sym, interval="15", **kw: (
        df15_dn if str(interval) == "15" else (df1h_bear if str(interval) == "60" else df4h_bull)
    )
    mock_cli.get_open_interest.return_value = pd.DataFrame({
        "timestamp": np.arange(30) * 3_600_000,
        "openInterest": np.linspace(1000.0, 1100.0, 30),  # +10% 24h OI
    })
    mock_cli.get_funding_history.return_value = pd.DataFrame({
        "timestamp": np.arange(50) * 28_800_000,
        "fundingRate": [0.00005] * 49 + [0.0003],  # extreme positive funding -> short trap
    })
    mock_cli._public_get.return_value = {
        "b": [[str(float(df15_dn["close"].iloc[-1]) * 0.9998), "2000"]],
        "a": [[str(float(df15_dn["close"].iloc[-1]) * 1.0002), "2000"]],
    }
    btc_neutral = BtcMarketContext(True, True, "NEUTRAL", "NEUTRAL", 0.1, False, 0.0, {"btc_price": 90000})

    short_cand = evaluate_symbol_12_schools(mock_cli, "SOLUSDT", btc_neutral, cfg)
    assert short_cand is not None, "Expected 1H SHORT to pass even when 4H is BULL!"
    assert short_cand.direction == "SHORT"
    assert short_cand.context_payload["higher_tf"]["4H"]["trend"] == "BULL"
    assert "S12_Deriv_OI_Funding" in short_cand.fired_schools
    assert short_cand.chart_png_b64 is not None and len(short_cand.chart_png_b64) > 1000

    # Now drop last candle RVOL below 1.2 -> S12 must be rejected!
    df15_low_vol = df15_dn.copy()
    df15_low_vol.loc[239, "volume"] = 800.0  # RVOL = 0.8 < 1.2
    mock_cli.get_klines.side_effect = lambda sym, interval="15", **kw: (
        df15_low_vol if str(interval) == "15" else (df1h_bear if str(interval) == "60" else df4h_bull)
    )
    assert evaluate_symbol_12_schools(mock_cli, "SOLUSDT", btc_neutral, cfg) is None, "Low RVOL S12 must be rejected!"
    print("[PASS] 3. 1H pullback SHORT (with 4H=BULL context) & S12 RVOL>=1.2 filter verified")

    # ------------------------------------------------------------------
    # 4. 3-Stage Smart Break-Even + Real ATR(14) Trailing
    # ------------------------------------------------------------------
    spec_eth = InstrumentSpec(
        symbol="ETHUSDT", base_coin="ETH", quote_coin="USDT", status="Trading",
        min_order_qty=0.01, qty_step=0.01, min_notional=5.0, tick_size=0.01,
        min_leverage=1.0, max_leverage=100.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = AITradeJournal(Path(tmpdir) / "test_be.db")
        cli_p = BybitClient(mode="demo", cache_dir=Path(tmpdir) / "cache", public_mainnet=True)
        exe = AITradeExecutor(cli_p, journal, cfg, TelegramNotifier(enabled=False), paper_mode=True)
        exe._specs_cache = {"ETHUSDT": spec_eth}

        # Open LONG @ 2999.90, SL = 1.6% -> sl_dist = 47.9984
        tid = exe.open_trade(cand, dec_clamped)
        assert tid is not None
        t = journal.get_active_trades()[0]
        ep = float(t["entry_price"])
        init_sl = float(t["sl_price"])
        sl_dist = ep * (float(t["sl_pct"]) / 100.0)
        future_close_ms = int(time.time() * 1000) + 600_000

        # Fill limit
        exe._manage_paper_trade(t, {"ETHUSDT": {"last_price": ep - 0.05}}, spec_eth)
        t = journal.get_active_trades()[0]
        assert t["status"] == "OPEN" and t["be_stage"] == 0

        # Intra-bar wick +1.4R, but closed 15m only +1.1R -> SL MUST NOT move!
        exe._manage_paper_trade(
            t,
            {"ETHUSDT": {"last_price": ep + 1.4 * sl_dist}},
            spec_eth,
            closed_bar_override={"bar_ts": 1, "bar_close_ms": future_close_ms, "close": ep + 1.1 * sl_dist, "atr14": 20.0},
        )
        t = journal.get_active_trades()[0]
        assert t["be_stage"] == 0 and abs(t["sl_price"] - init_sl) < 1e-6, "SL moved prematurely on intra-bar wick!"

        # Stage 1: Closed 15m >= +1.3R -> SL moves to -0.5R
        exe._manage_paper_trade(
            t,
            {"ETHUSDT": {"last_price": ep + 1.35 * sl_dist}},
            spec_eth,
            closed_bar_override={"bar_ts": 2, "bar_close_ms": future_close_ms, "close": ep + 1.32 * sl_dist, "atr14": 20.0},
        )
        t = journal.get_active_trades()[0]
        exp_s1 = round(ep - 0.5 * sl_dist, 2)
        assert t["be_stage"] == 1 and abs(t["sl_price"] - exp_s1) <= 0.02, f"Stage 1 SL mismatch: {t['sl_price']} vs {exp_s1}"

        # Stage 2: TP1 (55%) hits at 1.8R -> SL moves to -0.35R buffer behind entry
        exe._manage_paper_trade(t, {"ETHUSDT": {"last_price": float(t["tp1_price"]) + 1.0}}, spec_eth)
        t = journal.get_active_trades()[0]
        exp_s2 = round(ep - 0.35 * sl_dist, 2)
        assert t["status"] == "TP1_HIT" and t["be_stage"] == 2
        assert abs(t["sl_price"] - exp_s2) <= 0.02, f"Stage 2 SL mismatch: {t['sl_price']} vs {exp_s2}"

        # Stage 3: Closed 15m >= +2.3R -> SL moves to +0.3R and enables real 4.0*ATR(14) trailing
        cur_atr = 18.0  # 4.0 * 18.0 = 72.0 trail distance
        c_s3 = ep + 2.35 * sl_dist
        exe._manage_paper_trade(
            t,
            {"ETHUSDT": {"last_price": c_s3}},
            spec_eth,
            closed_bar_override={"bar_ts": 3, "bar_close_ms": future_close_ms, "close": c_s3, "atr14": cur_atr},
        )
        t = journal.get_active_trades()[0]
        exp_s3 = max(round(ep + 0.3 * sl_dist, 2), round(c_s3 - 4.0 * cur_atr, 2))
        assert t["be_stage"] == 3 and t["use_trailing"] == 1
        assert abs(t["sl_price"] - exp_s3) <= 0.02, f"Stage 3 SL mismatch: {t['sl_price']} vs {exp_s3}"

        # Active ATR trailing: price rallies to +4.0R -> SL trails strictly at peak - 4.0 * ATR(14)!
        rally_p = ep + 4.0 * sl_dist
        exe._manage_paper_trade(
            t,
            {"ETHUSDT": {"last_price": rally_p}},
            spec_eth,
            closed_bar_override={"bar_ts": 4, "bar_close_ms": future_close_ms, "close": rally_p, "atr14": cur_atr},
        )
        t = journal.get_active_trades()[0]
        exp_tr = round(rally_p - 4.0 * cur_atr, 2)
        assert abs(t["sl_price"] - exp_tr) <= 0.02, f"ATR trailing SL mismatch: {t['sl_price']} vs {exp_tr}"
        print(f"[PASS] 4. 3-Stage Smart BE & 4.0xATR trailing verified (S1={exp_s1} -> S2={exp_s2} -> S3={exp_s3} -> Trail={exp_tr})")

    # ------------------------------------------------------------------
    # 5. Daily -6.0% Stop WITHOUT Double-Counting uPnL on Bybit Demo
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = AITradeJournal(Path(tmpdir) / "test_dd.db")
        ex_cli = MagicMock()
        ex_cli.has_credentials = True
        ex_cli.mode = "demo"
        # Day starts at $2000 equity
        ex_cli.get_wallet_balance.return_value = {"equity": 2000.0, "available": 2000.0}
        exe_demo = AITradeExecutor(ex_cli, journal, cfg, TelegramNotifier(enabled=False), paper_mode=False)
        ok, _ = exe_demo.check_daily_loss_limit(tickers={})
        assert ok is True

        # Insert an open LONG with -$80 uPnL; Bybit UTA equity ALREADY reflects it ($1920 = -4.0% DD)
        journal.insert_trade({
            "symbol": "ETHUSDT", "direction": "LONG", "status": "OPEN", "mode": "demo",
            "entry_time": "2026-11-24T10:00:00Z", "entry_price": 3000.0, "sl_price": 2955.0,
            "init_sl_price": 2955.0, "tp1_price": 3080.0, "tp2_price": 3150.0, "be_price": 2984.0,
            "be_stage": 0, "use_trailing": 0, "peak_price": 3000.0, "qty": 1.0, "tp1_qty": 0.55,
            "remaining_qty": 1.0, "leverage": 10, "margin_usdt": 300.0, "notional_usdt": 3000.0,
            "sl_pct": 1.5, "tp1_r": 1.8, "tp2_r": 3.5, "primary_school": "S12_Deriv_OI_Funding",
            "schools_json": "[]", "ai_confidence": 80, "ai_reason": "test", "entry_order_id": "ord-1",
        })
        ex_cli.get_wallet_balance.return_value = {"equity": 1920.0, "available": 1620.0}
        ok2, reason2 = exe_demo.check_daily_loss_limit(tickers={"ETHUSDT": {"last_price": 2920.0}})
        assert ok2 is True, f"Double uPnL bug! -4% real DD triggered halt: {reason2}"

        # Now real equity drops to $1875 (-6.25% <= -6.0%) -> MUST halt!
        ex_cli.get_wallet_balance.return_value = {"equity": 1875.0, "available": 1575.0}
        ok3, reason3 = exe_demo.check_daily_loss_limit(tickers={"ETHUSDT": {"last_price": 2875.0}})
        assert ok3 is False and "Daily drawdown" in reason3
        print("[PASS] 5. Daily -6.0% stop verified (no double uPnL counting on Demo UTA equity)")

    # ------------------------------------------------------------------
    # 6. Partial Limit Fill in PENDING_LIMIT Cancels Remainder & Resizes TPs
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = AITradeJournal(Path(tmpdir) / "test_partial.db")
        ex_cli = MagicMock()
        ex_cli.has_credentials = True
        ex_cli.mode = "demo"
        ex_cli.get_positions.return_value = [{"symbol": "ETHUSDT", "size": "0.40", "avgPrice": "2999.50"}]
        ex_cli.place_limit_tp_order.return_value = {"orderId": "tp-ord-1"}
        exe_demo = AITradeExecutor(ex_cli, journal, cfg, TelegramNotifier(enabled=False), paper_mode=False)

        tid = journal.insert_trade({
            "symbol": "ETHUSDT", "direction": "LONG", "status": "PENDING_LIMIT", "mode": "demo",
            "entry_time": "2026-11-24T10:00:00Z", "expires_at_utc": "2099-01-01T00:00:00Z",
            "entry_price": 3000.0, "sl_price": 2955.0, "init_sl_price": 2955.0,
            "tp1_price": 3081.0, "tp2_price": 3157.5, "be_price": 2984.25, "be_stage": 0,
            "use_trailing": 0, "peak_price": 3000.0, "qty": 1.0, "tp1_qty": 0.55,
            "remaining_qty": 1.0, "leverage": 10, "margin_usdt": 300.0, "notional_usdt": 3000.0,
            "sl_pct": 1.5, "tp1_r": 1.8, "tp2_r": 3.5, "primary_school": "S12_Deriv_OI_Funding",
            "schools_json": "[]", "ai_confidence": 82, "ai_reason": "test", "entry_order_id": "lim-123",
        })
        t_pend = journal.get_active_trades()[0]
        exe_demo._manage_exchange_trade(t_pend, {"ETHUSDT": {"last_price": 3000.0}}, spec_eth)

        ex_cli.cancel_order.assert_called_once_with("ETHUSDT", "lim-123")
        t_open = journal.get_active_trades()[0]
        assert t_open["status"] == "OPEN"
        assert abs(t_open["qty"] - 0.40) < 1e-6 and abs(t_open["tp1_qty"] - 0.22) < 1e-6
        # Verify TP1 & TP2 orders placed strictly matching 0.40 total (0.22 + 0.18)
        tp_calls = ex_cli.place_limit_tp_order.call_args_list
        assert len(tp_calls) == 2
        assert abs(tp_calls[0].args[2] - 0.22) < 1e-6 and abs(tp_calls[1].args[2] - 0.18) < 1e-6
        print("[PASS] 6. Partial PENDING_LIMIT fill verified (remainder cancelled + TP1/TP2 resized to 0.22 + 0.18)")

    # ------------------------------------------------------------------
    # 7. Headless In-Memory PNG Chart Renderer + Safe Fallback
    # ------------------------------------------------------------------
    df_ch = _make_synth_klines(80, 100.0, 0.15, 900_000)
    b64 = render_15m_chart_png_b64(
        "ETHUSDT", "LONG", df_ch,
        df_ch["close"].ewm(span=20).mean().to_numpy(),
        df_ch["close"].ewm(span=50).mean().to_numpy(),
        df_ch["close"].rolling(20, min_periods=1).mean().to_numpy(),
        {"nearest_resistance": 114.0, "nearest_support": 108.0},
        {"nearest_resistance": 116.0, "nearest_support": 104.0},
        sl_pct=1.4, tp1_r=1.8, candles_count=72,
    )
    assert b64 is not None
    raw_png = base64.b64decode(b64)
    assert raw_png.startswith(b"\x89PNG\r\n\x1a\n") and len(raw_png) > 5000
    # Safe fallback on broken DataFrame must return None without raising
    assert render_15m_chart_png_b64("BAD", "LONG", pd.DataFrame(), np.array([]), np.array([]), np.array([]), {}, {}, 1.4, 1.8) is None
    print(f"[PASS] 7. In-memory PNG chart renderer verified ({len(raw_png):,} bytes PNG, safe fallback OK)")

    print("\nALL 7 COMPREHENSIVE VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()
