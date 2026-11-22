"""24/7 Pure-Python Technical Watchdog for the 4 Bybit Trading Bots (ADA, WLFI, LTC, MNT).

Strictly pure Python (ZERO LLM / AI dependencies). Runs every 2-3 minutes on Windows 11 or Ubuntu VPS:
1. Verifies all 4 bot processes (config_ada/wlfi/ltc/mnt.json) are alive and logs are fresh (<= 20 min).
2. Verifies via Bybit v5 API (respecting BYBIT_MODE from .env) that every open position has:
   - a server-side Stop-Loss (pos.stopLoss > 0), AND
   - a server-side Take-Profit (either pos.takeProfit > 0 OR an active reduceOnly TP order in open orders,
     accounting for TradeExecutor._on_maker_filled and WLFI dynamic EMA20 TP, with a 45s fill grace window).
3. Verifies no pending entry limit orders on Bybit are stuck past their configured TTL (+5 min grace).
4. Sends instant Telegram alerts with anti-spam deduplication (30m cooldown per issue + recovery notice).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

from bot.supervisor_tools import (
    BOT_CONFIGS,
    ROOT_DIR,
    RedactingFilter,
    create_bybit_client,
    query_sqlite_ro,
    redact_secrets,
    resolve_bybit_mode,
    send_telegram_to_all,
    telegram_target_chats,
    telegram_token,
    utc_now,
    utc_now_iso,
)

logger = logging.getLogger("watchdog")

STATE_PATH = ROOT_DIR / "data" / "watchdog_state.json"
LOG_PATH = ROOT_DIR / "logs" / "watchdog.log"

# Alerts depending on exchange data; they are neither raised nor resolved when that data is unavailable
POSITION_ALERT_PREFIXES = ("no_sl_", "no_tp_")
ORDER_ALERT_PREFIXES = ("no_tp_", "ttl_stuck_", "ttl_age_")
# An alert is resolved only after it is absent for this many consecutive checks
RESOLVE_AFTER_CHECKS = 2
# Long cooldown for config-level alerts that would otherwise repeat every 30 minutes
KEY_COOLDOWN_SEC = {"exchange_checks_disabled": 6 * 3600.0}


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ]
    for h in handlers:
        h.addFilter(RedactingFilter())
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


def send_telegram(text: str) -> bool:
    """Send to every chat from TELEGRAM_CHAT_ID (or admin ids). True if at least one got it."""
    load_dotenv(ROOT_DIR / ".env")
    if not telegram_token() or not telegram_target_chats():
        logger.warning("Telegram credentials not configured in .env; alert logged locally only: %s", text)
        return False
    return send_telegram_to_all(text, timeout=12)


def get_running_command_lines() -> List[str]:
    """Cross-platform process command-line inspection (Linux /proc + Windows WMI/CIM)."""
    cmdlines: List[str] = []
    proc_dir = Path("/proc")
    if proc_dir.exists():
        for entry in proc_dir.iterdir():
            if entry.name.isdigit():
                try:
                    raw = (entry / "cmdline").read_bytes()
                    if raw:
                        cmdlines.append(raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore"))
                except Exception:
                    pass
        if cmdlines:
            return cmdlines

    if os.name == "nt":
        try:
            out = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine",
                ],
                timeout=10,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return [line.strip() for line in out.splitlines() if line.strip()]
        except Exception as exc:
            logger.debug("Windows process check fallback note: %s", exc)
    return cmdlines


def load_watchdog_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {"active_alerts": {}, "first_seen_no_tp": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"active_alerts": {}, "first_seen_no_tp": {}}


def save_watchdog_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(f"{STATE_PATH.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.replace(tmp, STATE_PATH)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def check_once(max_log_age_min: float = 20.0, alert_cooldown_min: float = 30.0) -> List[str]:
    """Run all 3 watchdog checks and dispatch deduplicated Telegram alerts."""
    now = utc_now()
    now_ts = now.timestamp()
    state = load_watchdog_state()
    first_seen_no_tp: Dict[str, float] = state.get("first_seen_no_tp", {})
    current_issues: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 1. Process liveness & 15M log freshness across all 4 bots
    # ------------------------------------------------------------------
    # In Docker every bot runs in its own container, so /proc here only shows the watchdog.
    # Liveness then relies on log freshness and the container restart policy.
    skip_proc = os.getenv("WATCHDOG_SKIP_PROCESS_CHECK", "").strip().lower() in ("1", "true", "yes")
    cmdlines = [] if skip_proc else get_running_command_lines()
    for b in BOT_CONFIGS:
        cfg_name = b["config_file"]
        sym = b["symbol"]
        if cmdlines:
            alive = any(cfg_name.lower() in c.lower() for c in cmdlines)
            if not alive:
                current_issues[f"proc_dead_{sym}"] = f"🚨 [WATCHDOG] Процесс бота {sym} ({cfg_name}) НЕ ЗАПУЩЕН!"

        log_p = ROOT_DIR / b["log_path"]
        try:
            log_mtime: Optional[float] = log_p.stat().st_mtime
        except OSError:
            log_mtime = None
        if log_mtime is None:
            current_issues[f"log_missing_{sym}"] = f"⚠️ [WATCHDOG] Лог-файл {b['log_path']} не найден."
        else:
            age_min = (now_ts - log_mtime) / 60.0
            if age_min > max_log_age_min:
                current_issues[f"log_stale_{sym}"] = (
                    f"⚠️ [WATCHDOG] Лог {b['log_path']} ({sym}) не обновлялся {age_min:.1f} мин "
                    f"(порог {max_log_age_min:.0f} мин — возможное зависание или сон)."
                )

    # ------------------------------------------------------------------
    # 2. Bybit API check: Open Positions SL/TP & Stuck Pending Limit Orders
    # ------------------------------------------------------------------
    mode = resolve_bybit_mode()
    client = create_bybit_client(mode=mode, timeout=12)
    positions: List[Dict[str, Any]] = []
    open_orders: List[Dict[str, Any]] = []
    positions_ok = orders_ok = False
    if not client.has_credentials:
        current_issues["exchange_checks_disabled"] = (
            f"⚠️ [WATCHDOG] Нет ключей Bybit API ({mode.upper()}) в .env: проверки SL/TP и зависших "
            f"лимиток ОТКЛЮЧЕНЫ, следится только за процессами и логами."
        )
    else:
        api_errors: List[str] = []
        try:
            positions = client.get_positions()
            positions_ok = True
        except Exception as exc:
            api_errors.append(f"позиции: {redact_secrets(exc)}")
        try:
            open_orders = client.get_open_orders()
            orders_ok = True
        except Exception as exc:
            api_errors.append(f"ордера: {redact_secrets(exc)}")
        if api_errors:
            current_issues["bybit_api_err"] = (
                f"⚠️ [WATCHDOG] Ошибка связи с Bybit API ({mode.upper()}), часть проверок пропущена: "
                + "; ".join(api_errors)
            )

    if positions_ok or orders_ok:
        orders_by_sym: Dict[str, List[Dict[str, Any]]] = {}
        for od in open_orders:
            s = str(od.get("symbol", "")).upper()
            orders_by_sym.setdefault(s, []).append(od)

        # 2A. Verify every open position has server-side SL and TP (pos.takeProfit OR reduceOnly TP order)
        seen_open_syms: Set[str] = set()
        for pos in positions:
            sym = str(pos.get("symbol", "")).upper()
            size = float(pos.get("size") or 0.0)
            if size <= 0:
                continue
            seen_open_syms.add(sym)
            side = str(pos.get("side", ""))  # "Buy" or "Sell"
            sl_val = float(pos.get("stopLoss") or 0.0)
            tp_val = float(pos.get("takeProfit") or 0.0)

            if sl_val <= 0.0:
                current_issues[f"no_sl_{sym}"] = (
                    f"🚨 [WATCHDOG CRITICAL] Открытая позиция {sym} ({side}, size={size}) БЕЗ серверного Stop-Loss!"
                )

            if not orders_ok:
                continue
            sym_orders = orders_by_sym.get(sym, [])
            expected_tp_side = "Sell" if side == "Buy" else "Buy"
            has_reduce_tp_order = any(
                _is_reduce_only(o)
                and str(o.get("side", expected_tp_side)) == expected_tp_side
                and float(o.get("price") or 0.0) > 0
                for o in sym_orders
            )

            if tp_val <= 0.0 and not has_reduce_tp_order:
                # Allow 45s grace window right after Maker entry fill before TradeExecutor places PostOnly TP
                first_ts = first_seen_no_tp.get(sym, now_ts)
                first_seen_no_tp[sym] = first_ts
                if (now_ts - first_ts) >= 45.0:
                    current_issues[f"no_tp_{sym}"] = (
                        f"⚠️ [WATCHDOG] Открытая позиция {sym} ({side}, size={size}) без серверного Take-Profit "
                        f"(нет ни pos.takeProfit, ни reduceOnly лимитного TP-ордера дольше 45с)."
                    )
            else:
                first_seen_no_tp.pop(sym, None)

        if positions_ok:
            for sym in list(first_seen_no_tp.keys()):
                if sym not in seen_open_syms:
                    first_seen_no_tp.pop(sym, None)

        # 2B. Verify no pending Maker entry limit orders are stuck past TTL (+5 min grace)
        db_map = {b["symbol"]: b for b in BOT_CONFIGS}
        for sym, sym_orders in (orders_by_sym.items() if orders_ok else []):
            b_cfg = db_map.get(sym)
            ttl_bars = 4
            if b_cfg:
                try:
                    cjson = json.loads((ROOT_DIR / b_cfg["config_file"]).read_text(encoding="utf-8"))
                    ttl_bars = int(cjson.get("maker_ttl_bars", 4))
                except Exception:
                    pass
            max_age_sec = (ttl_bars * 15 + 5) * 60.0

            # Also check expires_at_utc in SQLite if present
            pending_rows = (
                query_sqlite_ro(
                    b_cfg["db_path"],
                    "SELECT entry_order_id, expires_at_utc FROM trades WHERE status = 'PENDING'",
                )
                if b_cfg
                else []
            )
            exp_by_oid = {str(r.get("entry_order_id", "")): r.get("expires_at_utc") for r in pending_rows}

            for od in sym_orders:
                if _is_reduce_only(od):
                    continue
                oid = str(od.get("orderId", ""))
                exp_iso = exp_by_oid.get(oid)
                if exp_iso:
                    try:
                        exp_dt = datetime.fromisoformat(str(exp_iso).replace("Z", "+00:00"))
                        overdue_min = (now - exp_dt).total_seconds() / 60.0
                        if overdue_min > 5.0:
                            current_issues[f"ttl_stuck_{sym}_{oid[:8]}"] = (
                                f"⚠️ [WATCHDOG] Зависший лимитный ордер {sym} (orderId={oid[:8]}...) "
                                f"просрочил TTL на {overdue_min:.1f} мин!"
                            )
                            continue
                    except ValueError:
                        pass

                created_ms = int(od.get("createdTime") or 0)
                if created_ms > 0:
                    age_sec = now_ts - (created_ms / 1000.0)
                    if age_sec > max_age_sec:
                        current_issues[f"ttl_age_{sym}_{oid[:8]}"] = (
                            f"⚠️ [WATCHDOG] Лимитный ордер входа {sym} (orderId={oid[:8]}...) висит "
                            f"{age_sec/60.0:.1f} мин (превышен TTL {ttl_bars} свечей + 5 мин)."
                        )

    # ------------------------------------------------------------------
    # 3. Anti-spam alert dispatch & recovery notifications
    # ------------------------------------------------------------------
    unchecked_prefixes: List[str] = []
    if not positions_ok:
        unchecked_prefixes.extend(POSITION_ALERT_PREFIXES)
    if not orders_ok:
        unchecked_prefixes.extend(ORDER_ALERT_PREFIXES)
    dispatch_alerts(
        state, current_issues, now_ts,
        cooldown_sec=alert_cooldown_min * 60.0,
        unchecked_prefixes=tuple(unchecked_prefixes),
    )

    state["first_seen_no_tp"] = first_seen_no_tp
    state["last_check_utc"] = utc_now_iso(now)
    save_watchdog_state(state)
    return list(current_issues.values())


def _is_reduce_only(order: Dict[str, Any]) -> bool:
    val = order.get("reduceOnly")
    return val is True or str(val).lower() == "true"


def dispatch_alerts(
    state: Dict[str, Any],
    current_issues: Dict[str, str],
    now_ts: float,
    cooldown_sec: float,
    unchecked_prefixes: Tuple[str, ...] = (),
) -> None:
    """Send new/repeated alerts and recovery notices, updating state in place.

    active_alerts[key] holds the time of the last delivered alert. Keys under unchecked_prefixes
    could not be verified this round and keep their state as is.
    """
    active_alerts: Dict[str, float] = state.setdefault("active_alerts", {})
    missing_checks: Dict[str, int] = state.setdefault("missing_checks", {})

    for key, msg in current_issues.items():
        missing_checks.pop(key, None)
        last_sent = float(active_alerts.get(key, 0.0))
        if (now_ts - last_sent) >= KEY_COOLDOWN_SEC.get(key, cooldown_sec):
            logger.warning(msg)
            if send_telegram(msg):
                active_alerts[key] = now_ts

    for key in list(active_alerts):
        if key in current_issues:
            continue
        if key.startswith(unchecked_prefixes):
            continue
        missing_checks[key] = int(missing_checks.get(key, 0)) + 1
        if missing_checks[key] < RESOLVE_AFTER_CHECKS:
            continue
        active_alerts.pop(key, None)
        missing_checks.pop(key, None)
        rec_msg = f"✅ [WATCHDOG OK] Проблема устранена: {key}"
        logger.info(rec_msg)
        send_telegram(rec_msg)

    for key in list(missing_checks):
        if key not in active_alerts:
            missing_checks.pop(key, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="24/7 Pure-Python Technical Watchdog for 4 Bybit Bots")
    parser.add_argument("--interval-sec", type=int, default=150, help="Check interval in seconds (default 150s = 2.5m)")
    parser.add_argument("--max-log-age-min", type=float, default=20.0, help="Max allowed log inactivity in minutes")
    parser.add_argument("--once", action="store_true", help="Run a single check and exit")
    args = parser.parse_args()

    setup_logging()
    mode = resolve_bybit_mode()
    logger.info("Watchdog started | Bybit Mode: %s | Interval: %ds | MaxLogAge: %.1fm", mode.upper(), args.interval_sec, args.max_log_age_min)

    consecutive_failures = 0
    last_failure_alert = 0.0
    while True:
        try:
            issues = check_once(max_log_age_min=args.max_log_age_min)
            consecutive_failures = 0
            if args.once:
                logger.info("Watchdog single check complete: %d active issue(s).", len(issues))
                break
        except Exception as exc:
            consecutive_failures += 1
            logger.exception("Watchdog loop unexpected error: %s", exc)
            if args.once:
                break
            if consecutive_failures >= 3 and time.time() - last_failure_alert >= 30 * 60:
                msg = (
                    f"🚨 [WATCHDOG] Проверка падает {consecutive_failures} раз подряд, мониторинг ботов не работает: "
                    f"{type(exc).__name__}: {redact_secrets(exc)[:300]}"
                )
                if send_telegram(msg):
                    last_failure_alert = time.time()
        time.sleep(max(30, args.interval_sec))


if __name__ == "__main__":
    main()
