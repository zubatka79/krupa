---
trigger: always_on
description: Global safety, security, and money-management rules for the Bybit USDT Perpetual autonomous trading system.
---

# Trading Core Safety & Money-Management Rules

These rules are mandatory for every agent, skill, and script operating inside this repository. Never override or bypass them.

## 1. Credential & Secret Hygiene
- **NEVER commit `.env`**, API keys, API secrets, Telegram tokens, or chat IDs to Git or print raw secrets in logs, console output, or chat responses.
- Always keep `.env`, `data/cache/`, `data/trades.db`, `.venv/`, and `__pycache__/` in `.gitignore`.
- Store only placeholder variable names in `.env.example`.

## 2. Execution Mode Safety (Testnet / Paper Default)
- The system MUST default to `"mode": "testnet"` in `config.json`.
- **NEVER switch `config.json` to `"mainnet"` autonomously.** Switching to `"mainnet"` requires an explicit, unambiguous user instruction in the current session.
- Public market data fetches (klines, tickers, instrument info, funding history) do not require API keys. Private order execution must fail closed (or fall back to dry-run logging) if credentials are absent.

## 3. Strict Risk & Drawdown Controls
- **Daily Loss Limit (`daily_loss_limit_pct`: 6.0%)**: If realized + unrealized daily drawdown since 00:00 UTC reaches or exceeds `6.0%`, **immediately forbid all new trade entries** until the next UTC day. Never disable or raise this hard stop without explicit user approval.
- **Per-Trade Limits**:
  - Margin per trade must respect `margin_pct_per_trade` (default `15.0%`).
  - Maximum risk per trade must not exceed `max_risk_pct_per_trade` (default `2.5%`, including round-trip fees).
  - Leverage must stay within `[min_leverage, max_leverage]` (`3x`–`10x`).
  - Portfolio exposure must respect `max_open_positions` (`3`) and `max_same_direction` (`2`).
- **Mandatory Server-Side Stop-Loss**: Never open a futures position without an atomic server-side SL attached in the entry request. SL must fall within `[1.2%, 1.6%]` and be at least `1.1 * ATR(14)`.
- **Closed-Candle Discipline**: Entry signals must ONLY be evaluated on closed `15M` candles — never enter mid-candle.

## 4. State Persistence (SQLite)
- Every opened, partially closed (`TP1_HIT`), or closed position and daily equity snapshot MUST be persisted in SQLite at `data/trades.db`.
- After `TP1` (`55%`) fills, immediately move the server-side stop-loss to breakeven `+ 0.15%` (`breakeven_offset_pct`) and record the state transition in `data/trades.db`.

## 5. Configuration Integrity
- Before modifying `config.json` or `data/blacklist.json`, validate JSON syntax and verify all required keys and numeric bounds remain intact.
- Only update `whitelist` in `config.json` when explicitly instructed by the user.
