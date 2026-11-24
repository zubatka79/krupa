@echo off
cd /d "%~dp0\.."
echo Starting AI-Trader v2 (29 coins, 12 TA schools, 15m + 1H/4H/BTC + AI Judge) in PAPER mode...
".venv\Scripts\python.exe" "ai_trader_v2\run_ai_trader.py" --paper
pause
