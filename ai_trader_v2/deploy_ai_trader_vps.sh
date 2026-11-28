#!/usr/bin/env bash
# ==============================================================================
# Ubuntu VPS Switcher -> Vika Hybrid (Single Unified Engine: 18L + 18S, 15x, 5x12%, DEMO)
#   1) Cleanly stops & disables ALL legacy bots (old vika-n2n7, any standalone
#      long bot, krupa-ai-trader, 6-bot services, watchdog, supervisor, docker stacks)
#      so ONLY ONE unified Vika Hybrid engine runs on the Bybit account.
#   2) Installs & starts exclusively `vika-hybrid.service` + weekly macro calendar timer.
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
PYTHON_BIN="${APP_DIR}/.venv/bin/python"
SERVICE_NAME="vika-hybrid"

if [ "$(id -u)" -ne 0 ]; then
    echo "[x] Run with sudo: sudo ./ai_trader_v2/deploy_ai_trader_vps.sh" >&2
    exit 1
fi

echo "=== [1/4] Stopping & disabling ALL legacy / separate bot services ==="
for SVC in \
    vika-n2n7 vika-long vika-long-v1 ai-trader-v2 ai-trader-v2-long \
    krupa-ai-trader krupa-ai-trader-long \
    krupa-bot-ada krupa-bot-wlfi krupa-bot-ltc krupa-bot-mnt krupa-bot-crv krupa-bot-hbar \
    krupa-watchdog krupa-supervisor krupa-ai-supervisor; do
    if systemctl list-unit-files "${SVC}.service" >/dev/null 2>&1; then
        systemctl stop "${SVC}.service" 2>/dev/null || true
        systemctl disable "${SVC}.service" 2>/dev/null || true
        echo "  [-] Stopped & disabled ${SVC}.service"
    fi
done

# Stop any running docker compose containers (including orphaned old ai-trader-v2 / long containers)
if command -v docker >/dev/null 2>&1; then
    (cd "${APP_DIR}" && docker compose down --remove-orphans 2>/dev/null) || true
    for CID in $(docker ps -q --filter "name=ai-trader" --filter "name=vika" --filter "name=krupa" 2>/dev/null || true); do
        docker stop "${CID}" 2>/dev/null || true
        docker rm "${CID}" 2>/dev/null || true
    done
fi

# Safety guard: terminate any stray standalone long or old vika python processes
pkill -f "ai_trader_v2_long/run_long_trader.py" 2>/dev/null || true
pkill -f "ai_trader_v2/run_ai_trader.py" 2>/dev/null || true

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

echo "=== [3/4] Installing SINGLE unified ${SERVICE_NAME}.service (Vika Hybrid: 18L+18S, 15x, 5x12%, DEMO) ==="
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

# Weekly systemd timer to refresh config/macro_events_utc.json via Claude API (Mondays 03:00 UTC)
cat >/etc/systemd/system/krupa-macro-calendar.service <<EOF
[Unit]
Description=Update US Macro Calendar (FOMC/CPI/NFP) via Claude API for Vika Hybrid
After=network-online.target

[Service]
Type=oneshot
User=${APP_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
ExecStart="${PYTHON_BIN}" "${APP_DIR}/update_macro_calendar.py"
EOF

cat >/etc/systemd/system/krupa-macro-calendar.timer <<EOF
[Unit]
Description=Weekly US Macro Calendar refresh (Mondays 03:00 UTC)

[Timer]
OnCalendar=Mon *-*-* 03:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
EOF

echo "=== [4/4] Starting SINGLE ${SERVICE_NAME}.service & weekly macro calendar timer ==="
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"
systemctl enable --now krupa-macro-calendar.timer || true
systemctl restart "${SERVICE_NAME}.service"
sleep 3
systemctl status "${SERVICE_NAME}.service" --no-pager -l || true
echo ""
echo "[OK] All legacy/separate bots stopped. ONLY Vika Hybrid (18L+18S) is now running on Bybit DEMO."
echo "Logs: journalctl -u ${SERVICE_NAME} -f  OR  tail -f ${APP_DIR}/logs/ai_trader_v2.log"
