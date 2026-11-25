@echo off
cd /d "%~dp0\.."
echo Starting VIKA (N2+N7) on Bybit DEMO (10x Cross, 10 coins)...
.venv\Scripts\python.exe -u ai_trader_v2\run_ai_trader.py --demo
pause
