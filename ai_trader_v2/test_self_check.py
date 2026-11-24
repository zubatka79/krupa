"""Offline end-to-end verification of ai_trader_v2 config, 12 schools, AI guardrails, SQLite & paper TP1->BE->TP2 lifecycle."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_trader_v2.ai_judge import AITradeJudge
from ai_trader_v2.executor import AITradeExecutor, TelegramNotifier
from ai_trader_v2.journal import AITradeJournal
from ai_trader_v2.run_ai_trader import load_config
from ai_trader_v2.ta_schools import ALL_12_SCHOOLS, CandidateSetup
from bot.bybit_client import BybitClient


def main():
    # 1. Config integrity check
    cfg = load_config()
    assert len(cfg["symbols"]) == 29, f"Expected 29 coins, got {len(cfg['symbols'])}"
    assert len(cfg["schools_enabled"]) == 12, f"Expected 12 schools, got {len(cfg['schools_enabled'])}"
    assert set(cfg["schools_enabled"]) == set(ALL_12_SCHOOLS)
    assert cfg["mode"] == "demo"
    assert cfg["min_schools_non_s12"] == 2
    assert cfg["min_rvol_non_s12"] == 1.2
    assert cfg["leverage"] == 10
    assert cfg["margin_pct_per_trade"] == 10.0
    assert cfg["max_open_positions"] == 6
    assert cfg["max_same_direction"] == 4
    assert cfg["daily_loss_limit_pct"] == 6.0
    print("[PASS] 1. Config validated (29 coins, 12 schools, testnet default, 10%x10x, max 6/4, -6% daily stop)")

    # 2. AI Guardrail Validator checks (wild LLM outputs must be clamped or rejected)
    cand = CandidateSetup(
        symbol="ETHUSDT",
        direction="LONG",
        bar_ts=1764000000000,
        close_price=3000.0,
        atr_14=32.0,
        atr_pct=1.066,  # 1.1 * ATR = 1.173% -> min SL = 1.2%
        rvol=1.6,
        adx_15m=26.0,
        rsi_15m=58.0,
        fired_schools=["S12_Deriv_OI_Funding", "S07_Ichimoku_Kumo"],
        primary_school="S12_Deriv_OI_Funding",
        priority_score=58.0,
        regime_label="STRONG_TREND",
        sl_candidates_pct={"swing_10_sl_pct": 1.35},
        default_sl_pct=1.35,
        recommended_tp1_r=1.8,
        recommended_tp2_r=3.5,
        orderbook={"spread_bps": 0.2, "depth_03_usdt": 5000000.0, "best_bid": 2999.9, "best_ask": 3000.1},
    )

    # Wild SL=4.5% must be clamped to 1.6% max!
    dec_clamped = AITradeJudge.validate_and_clamp(
        {
            "decision": "ENTER",
            "confidence": 85,
            "selected_school": "S12_Deriv_OI_Funding",
            "sl_pct": 4.5,
            "tp1_r": 1.8,
            "tp2_r": 3.5,
            "use_trailing_after_tp1": True,
            "reason_ru": "Тест зажима SL.",
        },
        cand,
        cfg,
        "test",
    )
    assert dec_clamped.decision == "ENTER"
    assert dec_clamped.sl_pct == 1.6, f"Expected SL clamped to 1.6, got {dec_clamped.sl_pct}"

    # Low confidence must be forced to SKIP
    dec_low_conf = AITradeJudge.validate_and_clamp(
        {"decision": "ENTER", "confidence": 60, "sl_pct": 1.4, "tp1_r": 1.8, "tp2_r": 3.2, "reason_ru": "Слабо"},
        cand,
        cfg,
        "test",
    )
    assert dec_low_conf.decision == "SKIP"
    print("[PASS] 2. AI Guardrails validated (SL strictly clamped [1.2%, 1.6%], low confidence rejected)")

    # 3. Full SQLite + Paper Executor Lifecycle (PENDING_LIMIT -> OPEN -> TP1_HIT+BE(+0.15%) -> TP2_HIT)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_p = Path(tmpdir) / "test_ai.db"
        journal = AITradeJournal(db_p)
        client = BybitClient(mode="testnet", cache_dir=Path(tmpdir) / "cache", public_mainnet=True)
        notifier = TelegramNotifier(enabled=False)
        exe = AITradeExecutor(client=client, journal=journal, config=cfg, notifier=notifier, paper_mode=True)
        # Inject mock spec so no network needed
        from bot.bybit_client import InstrumentSpec
        exe._specs_cache = {
            "ETHUSDT": InstrumentSpec(
                symbol="ETHUSDT", base_coin="ETH", quote_coin="USDT", status="Trading",
                min_order_qty=0.01, qty_step=0.01, min_notional=5.0, tick_size=0.01,
                min_leverage=1.0, max_leverage=100.0,
            )
        }
        tid = exe.open_trade(cand, dec_clamped)
        assert tid is not None
        active = journal.get_active_trades()
        assert len(active) == 1 and active[0]["status"] == "PENDING_LIMIT"
        ep = active[0]["entry_price"]
        tp1_p = active[0]["tp1_price"]
        tp2_p = active[0]["tp2_price"]
        be_p = active[0]["be_price"]

        # Simulate limit fill
        exe._manage_paper_trade(active[0], {"ETHUSDT": {"last_price": ep - 0.05}}, exe._specs_cache["ETHUSDT"])
        t = journal.get_active_trades()[0]
        assert t["status"] == "OPEN"

        # Simulate TP1 (55%) hit -> SL must immediately move to BE +0.15%!
        exe._manage_paper_trade(t, {"ETHUSDT": {"last_price": tp1_p + 0.5}}, exe._specs_cache["ETHUSDT"])
        t = journal.get_active_trades()[0]
        assert t["status"] == "TP1_HIT"
        assert abs(t["sl_price"] - be_p) < 1e-6, f"SL not moved to BE: {t['sl_price']} vs {be_p}"

        # Simulate TP2 (45%) hit -> CLOSED with positive Net PnL!
        exe._manage_paper_trade(t, {"ETHUSDT": {"last_price": tp2_p + 0.5}}, exe._specs_cache["ETHUSDT"])
        assert len(journal.get_active_trades()) == 0
        rep = journal.get_summary_report(2000.0)
        assert rep["closed_trades"] == 1 and rep["wins"] == 1 and rep["net_pnl_usdt"] > 0
        print(
            f"[PASS] 3. Paper lifecycle validated: PENDING -> OPEN -> TP1(55%)+BE({be_p}) -> TP2({tp2_p}) | "
            f"Net PnL: +${rep['net_pnl_usdt']:.2f} | Equity: ${rep['paper_equity_usdt']:.2f}"
        )

    print("ALL OFFLINE VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    main()
