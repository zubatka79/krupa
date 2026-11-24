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
from ai_trader_v2.divergences import (
    analyze_mtf_divergences,
    compute_divergence_priority_delta,
    detect_tf_divergence,
    find_confirmed_pivots,
    format_divergence_banner,
)
from ai_trader_v2.tv_advisors import (
    analyze_tv_advisors,
    compute_session_levels,
    compute_tv_advisors_priority_delta,
    compute_volume_profile,
    detect_liquidity_sweep,
    format_tv_advisors_banner,
)
from ai_trader_v2.executor import AITradeExecutor, TelegramNotifier
from ai_trader_v2.journal import AITradeJournal
from ai_trader_v2.run_ai_trader import load_config
from ai_trader_v2.ta_schools import (
    ALL_12_SCHOOLS,
    BtcMarketContext,
    CandidateSetup,
    _find_swing_levels,
    evaluate_symbol_12_schools,
)
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
    assert cfg["divergence_enabled"] is True and cfg["divergence_min_indicators"] == 2
    assert cfg["tv_sfp_enabled"] is True and cfg["tv_vp_enabled"] is True and cfg["tv_session_enabled"] is True
    assert cfg["vp_bins"] == 50 and float(cfg["vp_value_area_pct"]) == 70.0
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

    # ------------------------------------------------------------------
    # 8. Multi-Indicator Divergences (Regular, Hidden, Zero-Repaint right=2, DIV Banner on PNG)
    # ------------------------------------------------------------------
    def _build_div_df(closes: list[float], vols: list[float], interval_ms: int = 900_000) -> pd.DataFrame:
        n_c = len(closes)
        base_ts = 1_760_000_000_000
        cl_a = np.array(closes, dtype=float)
        op_a = np.roll(cl_a, 1)
        op_a[0] = cl_a[0]
        # Distinct pivot extremes anchored on close so reversal bars never tie with pivot lows/highs
        hi_a = cl_a + 0.25
        lo_a = cl_a - 0.25
        v_a = np.array(vols, dtype=float)
        return pd.DataFrame({
            "timestamp": [base_ts + i * interval_ms for i in range(n_c)],
            "open": op_a, "high": hi_a, "low": lo_a, "close": cl_a,
            "volume": v_a, "turnover": v_a * cl_a,
        })

    # 8a. Synthetic REGULAR_BULL: sharp selloff to pivot low p1 (100 -> 82, heavy vol),
    # strong rebound on huge buying vol (82 -> 98), gentle low-volume drift to slightly lower low p2 (81.0),
    # followed by 2 closed confirmation bars (right=2)!
    reg_bull_closes = (
        [100.0 + 0.1 * i for i in range(25)]      # clean gentle uptrend, no fake flat-plateau pivot high
        + [97.0, 93.0, 88.0, 84.0, 82.0]          # sharp crash to p1=29 (82.0)
        + [86.0, 90.0, 94.0, 97.0, 98.0]          # strong high-vol rally
        + [96.0, 93.5, 91.0, 88.5, 86.0, 83.5, 81.0]  # slower low-vol drift to p2=41 (81.0 < 82.0)
        + [83.5, 85.5]                            # 2 confirmation bars (right=2) -> n=44
    )
    reg_bull_vols = (
        [1000.0] * 25
        + [3000.0, 3500.0, 4000.0, 4500.0, 5000.0]
        + [6000.0, 6500.0, 7000.0, 6500.0, 6000.0]
        + [700.0, 650.0, 600.0, 600.0, 550.0, 500.0, 500.0]
        + [2200.0, 2500.0]
    )
    df_rb_full = _build_div_df(reg_bull_closes, reg_bull_vols)
    p2_idx = 41
    # Zero-repaint check: BEFORE 2 confirmation bars close (at p2 and p2+1), p2 MUST NOT be a confirmed pivot!
    assert p2_idx not in find_confirmed_pivots(df_rb_full["low"].iloc[: p2_idx + 1].to_numpy(), left=4, right=2, is_high=False)
    assert p2_idx not in find_confirmed_pivots(df_rb_full["low"].iloc[: p2_idx + 2].to_numpy(), left=4, right=2, is_high=False)
    assert detect_tf_divergence(df_rb_full.iloc[: p2_idx + 2].reset_index(drop=True), left=4, right=2)["type"] != "REGULAR_BULL"

    # After right=2 bars close -> REGULAR_BULL must trigger with >= 3/4 STRONG and bars_ago == 2!
    rb_res = detect_tf_divergence(df_rb_full, left=4, right=2, max_bars_ago=8, min_indicators=2)
    assert rb_res["type"] == "REGULAR_BULL" and rb_res["count"] >= 3 and rb_res["strength"] == "STRONG"
    assert rb_res["bars_ago"] == 2, f"Expected bars_ago=2, got {rb_res}"

    # 8b. Synthetic HIDDEN_BEAR: downtrend with Lower High (p1=108 -> p2=104), while bounce to p2
    # is sharper on heavier buying volume so indicators print a Higher High at p2!
    hid_bear_closes = (
        [120.0 - 0.5 * i for i in range(25)]            # steady downtrend 120 -> 108
        + [106.0, 105.0, 105.5, 106.5, 108.0]           # modest bounce to p1=29 (108.0)
        + [104.0, 100.0, 96.0, 92.0, 90.0]              # selloff to 90.0
        + [93.0, 96.5, 100.0, 102.5, 104.0]             # sharp bounce to lower high p2=39 (104.0 < 108.0)
        + [101.0, 99.0]                                  # 2 confirmation bars (right=2)
    )
    hid_bear_vols = (
        [1500.0] * 25
        + [1200.0, 1200.0, 900.0, 900.0, 900.0]
        + [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        + [4500.0, 5000.0, 5500.0, 6000.0, 6500.0]
        + [2000.0, 2200.0]
    )
    df_hb = _build_div_df(hid_bear_closes, hid_bear_vols)
    hb_res = detect_tf_divergence(df_hb, left=4, right=2, max_bars_ago=8, min_indicators=2)
    assert hb_res["type"] == "HIDDEN_BEAR" and hb_res["count"] >= 2, f"Expected HIDDEN_BEAR, got {hb_res}"

    # 8c. MTF summary flags + priority delta + DIV banner on PNG chart
    df_flat = _build_div_df([100.0] * 50, [1000.0] * 50)
    mtf_short = analyze_mtf_divergences(df_hb, df_hb, df_rb_full, direction="SHORT", min_indicators=2)
    assert mtf_short["summary"]["confirms_15m"] is True
    assert mtf_short["summary"]["confirms_1h"] is True
    assert mtf_short["summary"]["opposes_4h"] is True
    assert mtf_short["summary"]["h4_exhaustion_warning"] is True
    # Confirming 15m+1H without 4H opposition yields positive bonus; opposing 4H STRONG reduces priority
    mtf_clean = analyze_mtf_divergences(df_rb_full, df_rb_full, df_flat, direction="LONG", min_indicators=2)
    assert compute_divergence_priority_delta(mtf_clean) > 0
    mtf_opposed = analyze_mtf_divergences(df_flat, df_flat, df_rb_full, direction="SHORT", min_indicators=2)
    assert compute_divergence_priority_delta(mtf_opposed) < 0

    banner = format_divergence_banner(mtf_short)
    assert banner.startswith("DIV: 15m H_BEAR(") and "4H BULL(" in banner, f"Unexpected banner: {banner}"
    # Verify chart renders cleanly with DIV banner in header (capture title via matplotlib Axes.set_title spy)
    import matplotlib.axes
    seen_titles: list[str] = []
    orig_set_title = matplotlib.axes.Axes.set_title
    def _spy_title(self, label, *a, **kw):
        seen_titles.append(str(label))
        return orig_set_title(self, label, *a, **kw)
    matplotlib.axes.Axes.set_title = _spy_title
    try:
        b64_div = render_15m_chart_png_b64(
            "ETHUSDT", "SHORT", df_ch,
            df_ch["close"].ewm(span=20).mean().to_numpy(),
            df_ch["close"].ewm(span=50).mean().to_numpy(),
            df_ch["close"].rolling(20, min_periods=1).mean().to_numpy(),
            {}, {}, sl_pct=1.4, tp1_r=1.8, candles_count=72, div_banner=banner,
        )
    finally:
        matplotlib.axes.Axes.set_title = orig_set_title
    assert b64_div is not None and any("DIV:" in t and "H_BEAR" in t for t in seen_titles)
    print(f"[PASS] 8. Divergences verified (REG_BULL {rb_res['count']}/4, HID_BEAR {hb_res['count']}/4, zero-repaint right=2, PNG '{banner}')")

    # ------------------------------------------------------------------
    # 9. TradingView Advisors (BULL/BEAR SFP, 4d VPVR POC/VAH/VAL, PDH/PDL/Asia UTC, PNG DIV+ADV)
    # ------------------------------------------------------------------
    # 9a. Exact UTC calendar-day Session Levels (Day 0: PDH=112.0, PDL=94.0; Day 1 Asia 00-08 UTC: High=106.0, Low=101.0)
    day0_ms = 1_763_856_000_000  # 2025-11-23 00:00:00 UTC exact midnight
    day1_ms = day0_ms + 86_400_000
    rows_s = []
    for k in range(96):  # Day 0 (96 x 15m)
        ts = day0_ms + k * 900_000
        h, l, c = 104.0, 99.0, 102.0
        if k == 20:
            h = 112.0  # PDH
        if k == 50:
            l = 94.0   # PDL
        rows_s.append((ts, c, h, l, c, 1000.0, 1000.0 * c))
    for k in range(44):  # Day 1 00:00 .. 10:45 UTC (first 32 bars = Asia 00-08 UTC)
        ts = day1_ms + k * 900_000
        h, l, c = 104.5, 102.0, 103.5
        if k == 10:
            h = 106.0  # ASIA_HIGH
        if k == 18:
            l = 101.0  # ASIA_LOW
        if k == 43:
            c, h, l = 111.7, 111.8, 111.3  # 0.27% below PDH=112.0 -> session_obstacle_ahead for LONG!
        rows_s.append((ts, c, h, l, c, 1000.0, 1000.0 * c))
    df_sess = pd.DataFrame(rows_s, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    sess_l = compute_session_levels(df_sess, None, close_now=111.7, direction="LONG")
    assert sess_l["pdh"] == 112.0 and sess_l["pdl"] == 94.0
    assert sess_l["asia_high"] == 106.0 and sess_l["asia_low"] == 101.0
    assert sess_l["nearest_session_level"]["name"] == "PDH"
    assert sess_l["session_obstacle_ahead"] is True

    # 9b. 4-day Volume Profile (POC anchored at massive volume cluster around 100.0)
    vp_rows = []
    for i in range(120):
        ts = day0_ms + i * 900_000
        if i < 60:
            # High volume cluster strictly around 100.0 (99.5 .. 100.5)
            vp_rows.append((ts, 100.0, 100.5, 99.5, 100.0, 8000.0, 800000.0))
        elif i < 90:
            vp_rows.append((ts, 96.0, 97.0, 95.0, 96.0, 400.0, 38400.0))
        else:
            vp_rows.append((ts, 104.0, 105.0, 103.0, 104.0, 400.0, 41600.0))
    df_vp = pd.DataFrame(vp_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    vp_above = compute_volume_profile(df_vp, close_now=104.8, direction="LONG", bins=50, value_area_pct=70.0)
    assert 99.4 <= vp_above["poc"] <= 100.6, f"POC off cluster: {vp_above}"
    assert vp_above["val"] <= vp_above["poc"] <= vp_above["vah"]
    assert vp_above["price_zone"] == "ABOVE_VAH" and vp_above["dist_to_poc_pct"] < 0 and vp_above["poc_obstacle_ahead"] is False
    # Price 0.6% below POC heading LONG -> poc_obstacle_ahead MUST be True!
    vp_obs = compute_volume_profile(df_vp, close_now=vp_above["poc"] * 0.994, direction="LONG", bins=50, value_area_pct=70.0)
    assert vp_obs["poc_obstacle_ahead"] is True, f"Expected poc_obstacle_ahead=True, got {vp_obs}"

    # 9c. BULL_SFP (sweeps PDL=94.0 with high volume and closes back at 94.6) & BEAR_SFP (sweeps PDH=112.0)
    df_sfp_bull = df_sess.copy()
    df_sfp_bull.loc[len(df_sfp_bull) - 1, ["open", "high", "low", "close", "volume"]] = [94.3, 94.8, 93.4, 94.6, 2200.0]
    sfp_b_long = detect_liquidity_sweep(df_sfp_bull, df_flat, df_flat, sess_l, direction="LONG", rvol_min=1.2)
    assert sfp_b_long["sweep_type"] == "BULL_SFP" and sfp_b_long["swept_tf"] == "PDL"
    assert sfp_b_long["with_volume"] is True and sfp_b_long["aligns_with_direction"] is True and sfp_b_long["trap_warning"] is False
    # Same BULL_SFP evaluated for a SHORT candidate -> trap_warning MUST be True!
    sfp_b_short = detect_liquidity_sweep(df_sfp_bull, df_flat, df_flat, sess_l, direction="SHORT", rvol_min=1.2)
    assert sfp_b_short["trap_warning"] is True and sfp_b_short["aligns_with_direction"] is False

    df_sfp_bear = df_sess.copy()
    df_sfp_bear.loc[len(df_sfp_bear) - 1, ["open", "high", "low", "close", "volume"]] = [111.8, 112.6, 111.4, 111.5, 2400.0]
    sfp_bear_s = detect_liquidity_sweep(df_sfp_bear, df_flat, df_flat, sess_l, direction="SHORT", rvol_min=1.2)
    assert sfp_bear_s["sweep_type"] == "BEAR_SFP" and sfp_bear_s["swept_tf"] == "PDH" and sfp_bear_s["aligns_with_direction"] is True

    # 9d. Priority adjustment + combined PNG render with BOTH DIV and ADV lines + POC/PDH/PDL overlays
    tv_pos = {"liquidity_sweep": sfp_b_long, "volume_profile": vp_above, "session_levels": {**sess_l, "session_obstacle_ahead": False}}
    tv_neg = {"liquidity_sweep": sfp_b_short, "volume_profile": vp_obs, "session_levels": sess_l}
    assert compute_tv_advisors_priority_delta(tv_pos) > 0
    assert compute_tv_advisors_priority_delta(tv_neg) < 0

    adv_banner = format_tv_advisors_banner(tv_pos)
    assert adv_banner.startswith("ADV: SFP=BULL(PDL) | VP=ABOVE_VAH (POC ") and "| SESS=PDH " in adv_banner, f"Bad ADV banner: {adv_banner}"

    seen_titles2: list[str] = []
    def _spy_title2(self, label, *a, **kw):
        seen_titles2.append(str(label))
        return orig_set_title(self, label, *a, **kw)
    matplotlib.axes.Axes.set_title = _spy_title2
    try:
        b64_both = render_15m_chart_png_b64(
            "ETHUSDT", "LONG", df_ch,
            df_ch["close"].ewm(span=20).mean().to_numpy(),
            df_ch["close"].ewm(span=50).mean().to_numpy(),
            df_ch["close"].rolling(20, min_periods=1).mean().to_numpy(),
            {}, {}, sl_pct=1.4, tp1_r=1.8, candles_count=72,
            div_banner=banner, adv_banner=adv_banner,
            tv_levels={"poc": 106.0, "pdh": 111.0, "pdl": 102.0},
        )
    finally:
        matplotlib.axes.Axes.set_title = orig_set_title
    assert b64_both is not None and any(("DIV:" in t and "ADV:" in t) for t in seen_titles2)
    print(f"[PASS] 9. TV Advisors verified (BULL/BEAR SFP, POC={vp_above['poc']}, PDH/PDL/Asia, PNG DIV+ADV '{adv_banner}')")

    # ------------------------------------------------------------------
    # 10. _find_swing_levels() Excludes Breakout Bar Wick (No Fake 0.1R Wall)
    # ------------------------------------------------------------------
    # Prior history has a real confirmed swing support at 96.0 (4% below breakdown close 100.0).
    # The latest closed breakout bar breaks down to close=100.0 with a lower wick to 99.85 (0.15% below close).
    sr_rows = []
    for i in range(40):
        ts = day0_ms + i * 3_600_000
        if i == 10:
            # Confirmed prior pivot low at 96.0
            sr_rows.append((ts, 98.0, 99.0, 96.0, 98.5, 1000.0, 98500.0))
        elif i in (8, 9, 11, 12):
            sr_rows.append((ts, 101.0, 102.0, 99.5, 101.0, 1000.0, 101000.0))
        else:
            sr_rows.append((ts, 104.0, 105.5, 102.5, 104.0, 1000.0, 104000.0))
    # Append latest closed breakout candle: close=100.0, lower wick=99.85 (would be fake 0.15% / 0.1R wall!)
    sr_rows.append((day0_ms + 40 * 3_600_000, 103.5, 103.6, 99.85, 100.0, 2500.0, 250000.0))
    df_brk = pd.DataFrame(sr_rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])

    sr_res = _find_swing_levels(df_brk, lookback=60, price=100.0)
    assert sr_res["nearest_support"] == 96.0, f"Breakout bar wick leaked into nearest_support: {sr_res}"
    assert sr_res["dist_to_sup_pct"] == 4.0, f"Expected 4.0% clearance to prior support, got {sr_res}"
    assert round(sr_res["dist_to_sup_pct"] / 1.4, 2) > 2.5  # > 1.5R clean clearance!
    print(f"[PASS] 10. _find_swing_levels() breakout-wick exclusion verified (nearest_support=96.0, dist=4.0%, wick 99.85 ignored)")

    print("\nALL 10 COMPREHENSIVE VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()
