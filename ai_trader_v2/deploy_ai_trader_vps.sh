#!/usr/bin/env bash
# ==============================================================================
# Ubuntu VPS Switcher -> Vika N2N7 (VIKA_N2_N7, 10 coins, 10x Cross, Bybit DEMO)
#   1) Cleanly stops & disables legacy Krupa services including krupa-ai-trader
#      (29 coins / AI Judge), 6-bot services, watchdog, supervisor, and docker stacks.
#   2) Installs & starts exclusively `vika-n2n7.service` in --demo mode.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="${APP_DIR}/.venv/bin/python"
SERVICE_NAME="vika-n2n7"

if [ "$(id -u)" -ne 0 ]; then
    echo "[x] Run with sudo: sudo ./ai_trader_v2/deploy_ai_trader_vps.sh" >&2
    exit 1
fi

echo "=== [1/4] Stopping & disabling legacy bot services ==="
for SVC in \
    krupa-ai-trader krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-bot-crv krupa-bot-hbar \
    krupa-watchdog krupa-supervisor krupa-ai-supervisor; do
    if systemctl list-unit-files "${SVC}.service" >/dev/null 2>&1; then
        systemctl stop "${SVC}.service" 2>/dev/null || true
        systemctl disable "${SVC}.service" 2>/dev/null || true
        echo "  [-] Stopped & disabled ${SVC}.service"
    fi
done

if command -v docker >/dev/null 2>&1; then
    (cd "${APP_DIR}" && docker compose down 2>/dev/null) || true
fi

echo "=== [2/4] Ensuring Python venv & isolated ai_trader_v2 directories ==="
if [ ! -x "${PYTHON_BIN}" ]; then
    if [ "${APP_USER}" != "root" ]; then
        sudo -u "${APP_USER}" python3 -m venv "${APP_DIR}/.venv"
        sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install --upgrade pip
        sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
    else
        python3 -m venv "${APP_DIR}/.venv"
        "${APP_DIR}/.venv/bin/pip" install --upgrade pip
        "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
    fi
fi
mkdir -p "${APP_DIR}/data/ai_trader_v2/cache" "${APP_DIR}/logs"
if [ "${APP_USER}" != "root" ]; then
    chown -R "${APP_USER}:" "${APP_DIR}/data/ai_trader_v2" "${APP_DIR}/logs"
fi

echo "=== [3/4] Installing ${SERVICE_NAME}.service (Vika N2N7, Bybit DEMO) ==="
cat >/etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=Vika N2N7 (VIKA_N2_N7) Bybit Autonomous Trader (10 coins, 10x Cross, DEMO)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart="${PYTHON_BIN}" -u "${APP_DIR}/ai_trader_v2/run_ai_trader.py" --demo
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full

[Install]
WantedBy=multi-user.target
EOF

echo "=== [4/4] Starting ${SERVICE_NAME}.service ==="
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl restart "${SERVICE_NAME}.service"
sleep 3
systemctl status "${SERVICE_NAME}.service" --no-pager -l || true
echo ""
echo "[OK] Legacy bots stopped. Vika N2N7 is now LIVE on Bybit DEMO."
echo "Logs: journalctl -u ${SERVICE_NAME} -f  OR  tail -f ${APP_DIR}/logs/ai_trader_v2.log"
