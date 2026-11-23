#!/usr/bin/env bash
# ==============================================================================
# Ubuntu 22.04 / 24.04 VPS Auto-Deployment Script for Krupa 4-Bot Portfolio
# + 24/7 Pure-Python Watchdog + AI Supervisor (Advisor Mode)
#
# Usage on Ubuntu VPS (as a normal user that will own and run the bots):
#   chmod +x deploy_vps.sh
#   sudo ./deploy_vps.sh
# Running the services as root requires ALLOW_ROOT=1.
# ==============================================================================
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="${APP_DIR}/.venv/bin/python"

if [ "$(id -u)" -ne 0 ]; then
    echo "[x] This script installs packages and systemd units, run it with sudo: sudo ./deploy_vps.sh" >&2
    exit 1
fi

if [ "$(id -u "${APP_USER}")" -eq 0 ] && [ "${ALLOW_ROOT:-0}" != "1" ]; then
    cat >&2 <<EOF
[x] The services would run as root (APP_USER=${APP_USER}).
    Log in as a normal user that owns ${APP_DIR} and run: sudo ./deploy_vps.sh
    To create one: sudo adduser krupa && sudo chown -R krupa: "${APP_DIR}"
    If you really want root, rerun with: sudo ALLOW_ROOT=1 ./deploy_vps.sh
EOF
    exit 1
fi
APP_GROUP="$(id -gn "${APP_USER}")"

# systemd unit values: % is a specifier, $ is variable expansion, quotes and
# backslashes must be escaped inside a quoted ExecStart argument.
systemd_escape_path() {
    local v="$1"
    v="${v//%/%%}"
    printf '%s' "$v"
}
systemd_quote_arg() {
    local v="$1"
    v="${v//\\/\\\\}"
    v="${v//\"/\\\"}"
    v="${v//%/%%}"
    v="${v//\$/\$\$}"
    printf '"%s"' "$v"
}

UNIT_WORKDIR="$(systemd_escape_path "${APP_DIR}")"
UNIT_PYTHON="$(systemd_quote_arg "${PYTHON_BIN}")"

unit_hardening() {
    cat <<EOF
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
EOF
}

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
    echo "[!] Created ${APP_DIR}/.env from .env.example - fill in your API keys before starting!"
fi
chown "${APP_USER}:${APP_GROUP}" "${APP_DIR}/.env"
chmod 600 "${APP_DIR}/.env"

echo "=== [3/4] Registering systemd services (6 Trading Bots + Watchdog + AI Supervisor) ==="
for COIN in ada wlfi ltc mnt crv hbar; do
    cat >/etc/systemd/system/krupa-bot-${COIN}.service <<EOF
[Unit]
Description=Krupa Bybit Trading Bot (${COIN^^}USDT)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${UNIT_WORKDIR}
ExecStart=${UNIT_PYTHON} $(systemd_quote_arg "${APP_DIR}/run_bot.py") --config config_${COIN}.json
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
$(unit_hardening)

[Install]
WantedBy=multi-user.target
EOF
done

cat >/etc/systemd/system/krupa-watchdog.service <<EOF
[Unit]
Description=Krupa 24/7 Pure-Python Technical Watchdog
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${UNIT_WORKDIR}
ExecStart=${UNIT_PYTHON} $(systemd_quote_arg "${APP_DIR}/watchdog.py") --interval-sec 150
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
$(unit_hardening)

[Install]
WantedBy=multi-user.target
EOF

cat >/etc/systemd/system/krupa-ai-supervisor.service <<EOF
[Unit]
Description=Krupa AI Supervisor & Telegram Assistant (Advisor Mode)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=10

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${UNIT_WORKDIR}
ExecStart=${UNIT_PYTHON} $(systemd_quote_arg "${APP_DIR}/ai_supervisor.py")
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
$(unit_hardening)

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-bot-crv krupa-bot-hbar krupa-watchdog krupa-ai-supervisor

echo "=== [4/4] Deployment complete! ==="
echo "1. Edit ${APP_DIR}/.env and insert your Bybit, Telegram, and AI keys."
echo "   TELEGRAM_CHAT_ID (and optionally TELEGRAM_ALLOWED_USER_IDS) must be set explicitly:"
echo "   the supervisor no longer registers the first user who writes to the bot and ignores all commands until they are set."
echo "2. Start all services:"
echo "   sudo systemctl start krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-bot-crv krupa-bot-hbar krupa-watchdog krupa-ai-supervisor"
echo "3. Check status:"
echo "   sudo systemctl status krupa-bot-* krupa-watchdog krupa-ai-supervisor"
