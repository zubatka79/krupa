#!/usr/bin/env bash
# ==============================================================================
# Ubuntu VPS Switcher -> Vika Hybrid (in-place on `vika-n2n7.service`, 18L+18S, 15x, 5x12%, DEMO)
#   1) Stops & disables any legacy/duplicate bots so ONLY ONE unified engine runs.
#   2) Replaces `vika-n2n7.service` in-place with Vika Hybrid (weekly macro calendar
#      auto-refresh is built directly into the bot background loop — zero extra timers).
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

echo "=== [1/4] Stopping & disabling duplicate / legacy services ==="
for SVC in \
    vika-hybrid vika-long vika-long-v1 ai-trader-v2 ai-trader-v2-long \
    krupa-ai-trader krupa-ai-trader-long krupa-macro-calendar \
    krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-bot-crv krupa-bot-hbar \
    krupa-watchdog krupa-supervisor krupa-ai-supervisor; do
    if systemctl list-unit-files "${SVC}.service" >/dev/null 2>&1; then
        systemctl stop "${SVC}.service" 2>/dev/null || true
        systemctl disable "${SVC}.service" 2>/dev/null || true
    fi
done
systemctl stop krupa-macro-calendar.timer 2>/dev/null || true
systemctl disable krupa-macro-calendar.timer 2>/dev/null || true

if command -v docker >/dev/null 2>&1; then
    (cd "${APP_DIR}" && docker compose down --remove-orphans 2>/dev/null) || true
fi
pkill -f "ai_trader_v2_long/run_long_trader.py" 2>/dev/null || true

echo "=== [2/4] Ensuring Python venv & directories ==="
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
else
    if [ "${APP_USER}" != "root" ]; then
        sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install -q -r "${APP_DIR}/requirements.txt"
    else
        "${APP_DIR}/.venv/bin/pip" install -q -r "${APP_DIR}/requirements.txt"
    fi
fi

mkdir -p "${APP_DIR}/data/ai_trader_v2/cache" "${APP_DIR}/logs" "${APP_DIR}/config"
if [ "${APP_USER}" != "root" ]; then
    chown -R "${APP_USER}:" "${APP_DIR}/data/ai_trader_v2" "${APP_DIR}/logs" "${APP_DIR}/config"
fi

echo "=== [3/4] Updating ${SERVICE_NAME}.service in-place -> Vika Hybrid (18L+18S, 15x, 5x12%, DEMO) ==="
cat >/etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=Vika Hybrid Bybit Autonomous Trader (All-Thick-18 18L+18S, 15x Isolated, 5x12%, DEMO)
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

echo "=== [4/4] Restarting ${SERVICE_NAME}.service ==="
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl restart "${SERVICE_NAME}.service"
sleep 3
systemctl status "${SERVICE_NAME}.service" --no-pager -l || true
echo ""
echo "[OK] In-place replacement complete. ONLY ${SERVICE_NAME}.service (Vika Hybrid 18L+18S) is running."
