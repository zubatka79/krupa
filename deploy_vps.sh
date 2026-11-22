#!/usr/bin/env bash
# ==============================================================================
# Ubuntu 22.04 / 24.04 VPS Auto-Deployment Script for Krupa 4-Bot Portfolio
# + 24/7 Pure-Python Watchdog + AI Supervisor (Advisor Mode)
#
# Usage on Ubuntu VPS:
#   chmod +x deploy_vps.sh
#   sudo ./deploy_vps.sh
# ==============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="${APP_DIR}/.venv/bin/python"

echo "=== [1/4] Installing system packages (python3-venv, python3-pip, git) ==="
apt-get update -y
apt-get install -y python3 python3-venv python3-pip git

echo "=== [2/4] Setting up Python virtual environment & dependencies ==="
sudo -u "${APP_USER}" python3 -m venv "${APP_DIR}/.venv"
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install --upgrade pip
sudo -u "${APP_USER}" "${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
sudo -u "${APP_USER}" mkdir -p "${APP_DIR}/data/cache" "${APP_DIR}/logs"

if [ ! -f "${APP_DIR}/.env" ]; then
    cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
    chown "${APP_USER}:${APP_USER}" "${APP_DIR}/.env"
    chmod 600 "${APP_DIR}/.env"
    echo "[!] Created ${APP_DIR}/.env from .env.example — fill in your API keys before starting!"
fi

echo "=== [3/4] Registering systemd services (4 Trading Bots + Watchdog + AI Supervisor) ==="
for COIN in ada wlfi ltc mnt; do
    cat >/etc/systemd/system/krupa-bot-${COIN}.service <<EOF
[Unit]
Description=Krupa Bybit Trading Bot (${COIN^^}USDT)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${PYTHON_BIN} ${APP_DIR}/run_bot.py --config config_${COIN}.json
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
done

cat >/etc/systemd/system/krupa-watchdog.service <<EOF
[Unit]
Description=Krupa 24/7 Pure-Python Technical Watchdog
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${PYTHON_BIN} ${APP_DIR}/watchdog.py --interval-sec 150
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/systemd/system/krupa-ai-supervisor.service <<EOF
[Unit]
Description=Krupa AI Supervisor & Telegram Assistant (Advisor Mode)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${PYTHON_BIN} ${APP_DIR}/ai_supervisor.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-watchdog krupa-ai-supervisor

echo "=== [4/4] Deployment complete! ==="
echo "1. Edit ${APP_DIR}/.env and insert your Bybit, Telegram, and AI keys."
echo "2. Start all services:"
echo "   sudo systemctl start krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-watchdog krupa-ai-supervisor"
echo "3. Check status:"
echo "   sudo systemctl status krupa-bot-* krupa-watchdog krupa-ai-supervisor"
