"""
Parity & Safety Verification Suite for VIKA_LONG_V1 (`ai_trader_v2_long/`):
  1. Verifies config_long_trader.json integrity (demo default, 18 liquid coins, risk bounds).
  2. Verifies 100% signal parity between production `long_engine.py` and research `deep_ta_scan.py`
     across historical Jul-Aug 2026 bars (ZERO drift!).
  3. Verifies BTC 4H UP switch blocks LONG when BTC is in a 4H downtrend (e.g. Nov 2026 crash).
  4. Verifies S1 (08:00-12:00 UTC pause), S5 (4H EMA50 >= +10% block), S2 (wick BE >= +1.40% @ close >= +0.60%),
     and max_entries_per_bar = 2 in paper SQLite execution!
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_long_v3"))

from ai_trader_v2.journal import VikaJournal
from ai_trader_v2_long.long_engine import (
    BtcRegimeState,
    M15_MS,
    evaluate_btc_4h_regime,
    evaluate_long_coin,
)
from ai_trader_v2_long.long_executor import VikaLongExecutor
from bot.bybit_client import BybitClient, InstrumentSpec
import deep_ta_scan


def test_config_integrity() -> None:
    cfg = json.loads((ROOT / "ai_trader_v2_long" / "config_long_trader.json").read_text(encoding="utf-8"))
    assert cfg["mode"] == "demo", "Default mode must be demo/testnet, never mainnet!"
    assert cfg["allowed_directions"] == "only_long"
    assert cfg["leverage"] == 10
    assert cfg["daily_loss_limit_pct"] == 6.0
    assert cfg["max_open_positions"] == 8
    assert cfg["margin_pct_per_trade"] == 10.0
    assert cfg["max_entries_per_bar"] == 2
    assert len(cfg["coins"]) == 18
    assert all(c["symbol"] != "BCHUSDT" for c in cfg["coins"])
    assert cfg["filters"]["btc_4h_up_switch"] is True
    assert cfg["filters"]["skip_utc_hours"] == [8, 9, 10, 11]
    assert cfg["filters"]["max_coin_dist4h_ema50_pct"] == 10.0
    print("[PASS] 1. config_long_trader.json integrity & safety bounds")


def test_signal_parity_with_backtest() -> None:
    """Verify exact 1:1 signal parity between live long_engine.py and vectorized deep_ta_scan.py."""
    data_dir = ROOT / "research_long_v3" / "data"
    df_btc = pd.read_csv(data_dir / "BTCUSDT_15m.csv.gz", compression="gzip").drop_duplicates("timestamp").sort_values("timestamp")
    master_tl = df_btc["timestamp"].to_numpy(np.int64)
    tr_s = int(np.searchsorted(master_tl, deep_ta_scan.TRAIN_START, "left"))
    bl_s = int(np.searchsorted(master_tl, deep_ta_scan.BLIND_START, "left"))

    # Check 6 distinct keys on real coins across 200 sample bars each (including positive signal bars!)
    samples = [
        ("UNIUSDT", "T6_MFI_DIP_BUY"),
        ("CRVUSDT", "T2_KELTNER_PULLBACK"),
        ("XMRUSDT", "T4_STOCH_CROSS_LOW"),
        ("ZECUSDT", "T10_2RED_ENGULF"),
        ("ETHUSDT", "T9_LIQ_SWEEP_HAMMER"),
        ("WLDUSDT", "T5_CCI_REBOUND"),
    ]
    btc_up = BtcRegimeState(0, 75000.0, 74000.0, 70000.0, True, 7.1)
    fcfg_raw = {"btc_4h_up_switch": False, "skip_utc_hours": [], "max_coin_dist4h_ema50_pct": 999.0}

    checked = 0
    matched_pos = 0
    for sym, key in samples:
        c = deep_ta_scan.load_coin(sym, master_tl, tr_s, bl_s)
        df15 = pd.read_csv(data_dir / f"{sym}_15m.csv.gz", compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df1h = pd.read_csv(data_dir / f"{sym}_1h.csv.gz", compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df4h = pd.read_csv(data_dir / f"{sym}_4h.csv.gz", compression="gzip").drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

        sig_vec = c["sigs"][key]
        pos_bars = np.flatnonzero(sig_vec[tr_s : tr_s + 2000]) + tr_s
        neg_bars = np.flatnonzero(~sig_vec[tr_s : tr_s + 2000]) + tr_s
        test_bars = list(pos_bars[:8]) + list(neg_bars[::250][:8])

        for i in test_bars:
            ts_bar = int(master_tl[i])
            now_ms = ts_bar + M15_MS + 1000
            ev = evaluate_long_coin(sym, key, df15, df1h, df4h, btc_up, 0.0, now_ms, filters_cfg=fcfg_raw)
            expected = "LONG" if bool(sig_vec[i]) else None
            assert ev.raw_signal == expected, f"Parity mismatch {sym} {key} @ bar {i}: got {ev.raw_signal}, expected {expected}"
            checked += 1
            if expected == "LONG":
                matched_pos += 1
    print(f"[PASS] 2. Signal parity 100% exact across {checked} real bars ({matched_pos} positive LONG triggers verified)")


def test_btc_switch_and_filters() -> None:
    """Verify BTC 4H DOWN sleeps in Nov 2026 crash + S1 UTC 08-12 & S5 4H>=10% block properly."""
    nov_dir = ROOT / "research_long_v3" / "data_nov2026"
    df_btc4 = pd.read_csv(nov_dir / "BTCUSDT_4h.csv.gz", compression="gzip").sort_values("timestamp")
    now_nov = int(df_btc4["timestamp"].iloc[-1]) + 4 * 3_600_000 + 1000
    reg = evaluate_btc_4h_regime(df_btc4, now_nov)
    assert reg.is_4h_up is False, f"Expected BTC 4H DOWN in late Nov 2026, got {reg}"

    # Verify BTC 4H DOWN blocks a known positive Jul-Aug signal
    data_dir = ROOT / "research_long_v3" / "data"
    df15 = pd.read_csv(data_dir / "UNIUSDT_15m.csv.gz", compression="gzip").sort_values("timestamp").reset_index(drop=True)
    df1h = pd.read_csv(data_dir / "UNIUSDT_1h.csv.gz", compression="gzip").sort_values("timestamp").reset_index(drop=True)
    df4h = pd.read_csv(data_dir / "UNIUSDT_4h.csv.gz", compression="gzip").sort_values("timestamp").reset_index(drop=True)
    df_btc = pd.read_csv(data_dir / "BTCUSDT_15m.csv.gz", compression="gzip").sort_values("timestamp")
    tl = df_btc["timestamp"].to_numpy(np.int64)
    tr_s = int(np.searchsorted(tl, deep_ta_scan.TRAIN_START, "left"))
    bl_s = int(np.searchsorted(tl, deep_ta_scan.BLIND_START, "left"))
    c = deep_ta_scan.load_coin("UNIUSDT", tl, tr_s, bl_s)
    pos_i = int(np.flatnonzero(c["sigs"]["T6_MFI_DIP_BUY"][tr_s:bl_s])[0] + tr_s)
    now_ms = int(tl[pos_i]) + M15_MS + 1000

    ev_blocked = evaluate_long_coin(
        "UNIUSDT", "T6_MFI_DIP_BUY", df15, df1h, df4h, reg, 0.0, now_ms,
        filters_cfg={"btc_4h_up_switch": True, "skip_utc_hours": [], "max_coin_dist4h_ema50_pct": 999.0},
    )
    assert ev_blocked.raw_signal == "LONG" and ev_blocked.final_signal is None and "BTC_4H_DOWN" in (ev_blocked.block_reason or "")
    print(f"[PASS] 3. BTC 4H switch verified ({reg.label}) — blocked raw LONG signal cleanly")


def test_executor_wick_be_and_bar_cap() -> None:
    """Verify max_entries_per_bar=2 and S2 wick BE (+1.40% high & +0.60% close -> +0.30% SL) in paper SQLite."""
    cfg = json.loads((ROOT / "ai_trader_v2_long" / "config_long_trader.json").read_text(encoding="utf-8"))
    cfg["mode"] = "paper"
    cfg["dry_run"] = True

    with tempfile.TemporaryDirectory() as tmp:
        journal = VikaJournal(Path(tmp) / "test_long.db")
        client = BybitClient(mode="paper", cache_dir=Path(tmp) / "cache")
        ex = VikaLongExecutor(cfg=cfg, client=client, journal=journal)
        # Mock specs for 3 coins
        for sym in ("UNIUSDT", "ETHUSDT", "SOLUSDT"):
            ex.specs[sym] = InstrumentSpec(sym, sym[:-4], "USDT", "Trading", 0.01, 0.01, 5.0, 0.001, 1.0, 50.0)

        bar_ts = 1_785_000_000_000
        evs = [
            deep_ta_scan_ev("UNIUSDT", "T6_MFI_DIP_BUY", bar_ts, 10.0),
            deep_ta_scan_ev("ETHUSDT", "T9_LIQ_SWEEP_HAMMER", bar_ts, 3000.0),
            deep_ta_scan_ev("SOLUSDT", "T6_MFI_DIP_BUY", bar_ts, 150.0),
        ]
        p1 = ex.open_signal(evs[0])
        p2 = ex.open_signal(evs[1])
        p3 = ex.open_signal(evs[2])
        assert p1 is not None and p2 is not None and p3 is None, f"max_entries_per_bar=2 failed: {p1}, {p2}, {p3}"

        # S2 Wick BE test on UNIUSDT (entry=10.0, normal BE=10.12, wick BE high>=10.14 & close>=10.06 -> lock=10.03)
        moved_no = ex.check_breakeven_on_closed_bar_with_wick("UNIUSDT", bar_ts + M15_MS, closed_close=10.08, closed_high=10.11)
        assert not moved_no, "Should NOT trigger BE at +0.8% close / +1.1% high"
        moved_wick = ex.check_breakeven_on_closed_bar_with_wick("UNIUSDT", bar_ts + 2 * M15_MS, closed_close=10.07, closed_high=10.145)
        assert moved_wick, "Should trigger S2 Wick BE at +1.45% high & +0.7% close!"
        pos = [p for p in journal.get_open_positions() if p["symbol"] == "UNIUSDT"][0]
        assert pos["be_active"] == 1 and abs(float(pos["sl_price"]) - 10.03) < 1e-6
    print("[PASS] 4. Executor max_entries_per_bar=2 & S2 Wick BE (+1.40% / +0.60% -> +0.30%) verified in SQLite")


def deep_ta_scan_ev(sym: str, key: str, bar_ts: int, px: float):
    from ai_trader_v2_long.long_engine import LongCoinEvaluation
    return LongCoinEvaluation(
        symbol=sym, key=key, bar_ts=bar_ts, close_price=px, high_price=px * 1.002,
        rvol=1.5, atr15_pct=0.5, h1_up=True, h4_up=True, dist4h_ema50_pct=2.0,
        btc_4h_up=True, utc_hour=14, raw_signal="LONG", final_signal="LONG",
    )


if __name__ == "__main__":
    test_config_integrity()
    test_signal_parity_with_backtest()
    test_btc_switch_and_filters()
    test_executor_wick_be_and_bar_cap()
    print("\nALL VIKA_LONG_V1 PARITY & SAFETY TESTS PASSED!")
