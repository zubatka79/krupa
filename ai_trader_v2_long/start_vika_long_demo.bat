@echo off
cd /d "%~dp0\.."
.venv\Scripts\python.exe ai_trader_v2_long\run_long_trader.py --demo
pause
