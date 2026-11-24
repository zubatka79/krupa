#!/usr/bin/env bash
# ==============================================================================
# Ubuntu VPS Switcher:
#   1) Cleanly stops & disables legacy 6-bot services (ada/wlfi/ltc/mnt/crv/hbar),
#      krupa-watchdog, and krupa-supervisor (leaving old configs untouched on disk).
#   2) Registers & starts `krupa-ai-trader.service` (AI-Trader v2: 29 coins,
#      12 TA schools, 15m + 1H/4H/BTC, Variant 2 consensus, Claude Judge) on Bybit DEMO.
#
# Usage on Ubuntu VPS:
#   chmod +x ai_trader_v2/deploy_ai_trader_vps.sh
#   sudo ./ai_trader_v2/deploy_ai_trader_vps.sh
# (If running as root, use: sudo ALLOW_ROOT=1 ./ai_trader_v2/deploy_ai_trader_vps.sh)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="${APP_DIR}/.venv/bin/python"

if [ "$(id -u)" -ne 0 ]; then
    echo "[x] Run with sudo: sudo ./ai_trader_v2/deploy_ai_trader_vps.sh" >&2
    exit 1
fi

if [ "$(id -u "${APP_USER}")" -eq 0 ] && [ "${ALLOW_ROOT:-0}" != "1" ]; then
    echo "[x] Refusing to run service as root without ALLOW_ROOT=1." >&2
    exit 1
fi

echo "=== [1/4] Stopping & disabling legacy Krupa bots ==="
for SVC in \
    krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-bot-crv krupa-bot-hbar \
    krupa-watchdog krupa-supervisor krupa-ai-supervisor; do
    if systemctl list-unit-files "${SVC}.service" >/dev/null 2>&1; then
        systemctl stop "${SVC}.service" 2>/dev/null || true
        systemctl disable "${SVC}.service" 2>/dev/null || true
        echo "  [-] Stopped & disabled ${SVC}.service"
    fi
done

echo "=== [2/4] Ensuring Python venv & isolated ai_trader_v2 directories ==="
if [ ! -x "${PYTHON_BIN}" ]; then
    sudo -u "${APP_USER}" python3 -m venv "${APP_DIR}/.venv"
    sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install --upgrade pip
    sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
fi
sudo -u "${APP_USER}" mkdir -p "${APP_DIR}/data/ai_trader_v2/cache" "${APP_DIR}/logs"

echo "=== [3/4] Installing krupa-ai-trader.service (Bybit DEMO, Variant 2) ==="
cat >/etc/systemd/system/krupa-ai-trader.service <<EOF
[Unit]
Description=Krupa AI-Trader v2 (29 Coins, 12 TA Schools, 15m+1H/4H/BTC, Bybit DEMO)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart="${PYTHON_BIN}" "${APP_DIR}/ai_trader_v2/run_ai_trader.py" --demo
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
EOF

echo "=== [4/4] Starting krupa-ai-trader.service ==="
systemctl daemon-reload
systemctl enable --now krupa-ai-trader.service
sleep 2
systemctl status krupa-ai-trader.service --no-pager -l || true
echo ""
echo "[OK] Legacy bots stopped. AI-Trader v2 is now LIVE on Bybit DEMO."
echo "Logs: journalctl -u krupa-ai-trader -f  OR  tail -f ${APP_DIR}/logs/ai_trader_v2.log"
