#!/usr/bin/env bash
# Deploy / Restart VIKA_N2_N7 («ВИКА (N2+N7)») systemd service on Ubuntu VPS
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/krupa}"
SERVICE_NAME="vika-n2n7"

cd "$APP_DIR"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

cat <<EOF | sudo tee "/etc/systemd/system/${SERVICE_NAME}.service" >/dev/null
[Unit]
Description=VIKA_N2_N7 Bybit Autonomous Trader (10 coins, 10x Cross)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/.venv/bin/python -u ${APP_DIR}/ai_trader_v2/run_ai_trader.py --demo
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager
