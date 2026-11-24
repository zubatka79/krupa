@echo off
cd /d "%~dp0\.."
echo Starting AI-Trader v2 (29 coins, 12 TA schools, Variant 2 consensus, 15m + 1H/4H/BTC + AI Judge) on BYBIT DEMO...
".venv\Scripts\python.exe" "ai_trader_v2\run_ai_trader.py" --demo
pause
