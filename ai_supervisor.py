"""AI Supervisor & Telegram Assistant (Advisor Mode) for the 4 Bybit Trading Bots.

CRITICAL SAFETY RULE (ADVISOR MODE):
- The AI NEVER generates 15M entry signals and NEVER modifies config_*.json or pauses coins autonomously.
- Autonomous scheduled tasks (News Scout, Market Strategist, Trade Auditor, Missed Fill, Liquidity Monitor,
  R&D Coin Scout) ONLY analyze and send recommendations/buttons to Telegram.
- Pausing or unpausing a coin in data/blacklist.json happens STRICTLY on explicit user command or button click
  from an authorized admin (TELEGRAM_ALLOWED_USER_IDS) or the configured TELEGRAM_CHAT_ID chats.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from bot.supervisor_tools import (
    BOT_CONFIGS,
    BOT_SYMBOLS,
    MAX_PAUSE_HOURS,
    RedactingFilter,
    RND_LOCK_PATH,
    RND_RESULT_PATH,
    ROOT_DIR,
    BlacklistError,
    audit_missed_fills,
    audit_trades_and_hidden_costs,
    check_liquidity_and_caps,
    compute_market_regime,
    create_bybit_client,
    fetch_coin_news_digest,
    get_active_pauses,
    parse_id_list,
    parse_pause_hours,
    pid_alive,
    read_rnd_lock,
    redact_secrets,
    resolve_bybit_mode,
    set_coin_pause,
    utc_now,
    utc_now_iso,
    write_rnd_lock,
)

logger = logging.getLogger("ai_supervisor")
LOG_PATH = ROOT_DIR / "logs" / "ai_supervisor.log"
USAGE_PATH = ROOT_DIR / "data" / "ai_usage.json"
ALERT_THRESHOLDS_USD = (30.0, 40.0, 45.0, 50.0)


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


# ----------------------------------------------------------------------
# Universal AI Client (Google Gemini Native REST + OpenAI-Compatible Token/Proxy)
# ----------------------------------------------------------------------
class UniversalAIClient:
    """Supports both Google Gemini API (GEMINI_API_KEY) and OpenAI/OpenRouter/Proxy tokens (AI_API_KEY)."""

    SYSTEM_PROMPT = (
        "Ты — ИИ-супервайзер и риск-менеджер портфеля из 4 алгоритмических ботов на Bybit USDT Perpetual "
        "(Вариант Б / пакет NEW, 10x Isolated, 15% маржи на сделку): "
        "1) ADAUSDT (Trend Fibo 0.50 + фильтр EMA200 < 5%), "
        "2) WLFIUSDT (Counter-Trend Keltner 3.2xATR + RSI 40/60), "
        "3) LTCUSDT (Trend Bollinger 1.4xSD + BTC Veto 0.7%, TTL 2), "
        "4) MNTUSDT (Donchian-10 Breakout + H1 EMA50 slope > 0 + BTC Veto 0.5% + ранний БУ). "
        "Эталон портфеля NEW: Win Rate 84.0% (OOS 80.7%), Profit Factor 1.92, Max DD 11.0%. "
        "ПРАВИЛО БЕЗОПАСНОСТИ: ты работаешь строго в режиме СОВЕТНИКА (Advisor Mode). "
        "Отвечай кратко, чётко, по делу на русском языке, с цифрами из переданных данных. "
        "Если видишь угрозу (безоткатный обвал BTC или опасная новость по монете), порекомендуй пользователю "
        "нажать кнопку паузы в Telegram."
    )

    def __init__(self, on_budget_alert: Optional[Any] = None) -> None:
        self.on_budget_alert = on_budget_alert
        self.reload_env()

    def reload_env(self) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5").strip() or "claude-opus-5-5"
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        self.ai_key = os.getenv("AI_API_KEY", "").strip()
        self.ai_base_url = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        self.ai_model = os.getenv("AI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        try:
            self.monthly_limit_usd = max(1.0, float(os.getenv("AI_MONTHLY_LIMIT_USD", "50.0").strip()))
        except ValueError:
            self.monthly_limit_usd = 50.0

    @property
    def is_configured(self) -> bool:
        has_ant = bool(self.anthropic_key and "your_" not in self.anthropic_key.lower())
        has_g = bool(self.gemini_key and "your_" not in self.gemini_key.lower())
        has_a = bool(self.ai_key and "your_" not in self.ai_key.lower())
        return has_ant or has_g or has_a

    def _load_usage(self) -> Dict[str, Any]:
        month_key = utc_now().strftime("%Y-%m")
        default = {
            "month": month_key,
            "spent_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
            "alerted_thresholds": [],
        }
        if not USAGE_PATH.exists():
            return default
        try:
            data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
            if data.get("month") != month_key:
                return default
            return data
        except Exception:
            return default

    def _save_usage(self, usage: Dict[str, Any]) -> None:
        USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = USAGE_PATH.with_name(f"{USAGE_PATH.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(usage, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.replace(tmp, USAGE_PATH)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def get_budget_status(self) -> Dict[str, Any]:
        self.reload_env()
        u = self._load_usage()
        spent = float(u.get("spent_usd", 0.0))
        return {
            "month": u.get("month"),
            "spent_usd": round(spent, 4),
            "limit_usd": self.monthly_limit_usd,
            "remaining_usd": round(max(0.0, self.monthly_limit_usd - spent), 4),
            "requests": int(u.get("requests", 0)),
            "input_tokens": int(u.get("input_tokens", 0)),
            "output_tokens": int(u.get("output_tokens", 0)),
            "halted": spent >= self.monthly_limit_usd,
        }

    def _record_usage(self, in_tok: int, out_tok: int, in_rate_per_m: float, out_rate_per_m: float) -> None:
        usage = self._load_usage()
        cost = (in_tok / 1_000_000.0) * in_rate_per_m + (out_tok / 1_000_000.0) * out_rate_per_m
        usage["input_tokens"] = int(usage.get("input_tokens", 0)) + in_tok
        usage["output_tokens"] = int(usage.get("output_tokens", 0)) + out_tok
        usage["requests"] = int(usage.get("requests", 0)) + 1
        usage["spent_usd"] = round(float(usage.get("spent_usd", 0.0)) + cost, 6)
        spent = float(usage["spent_usd"])
        alerted = list(usage.get("alerted_thresholds", []))

        for th in ALERT_THRESHOLDS_USD:
            if th <= self.monthly_limit_usd and spent >= th and th not in alerted:
                alerted.append(th)
                if th >= self.monthly_limit_usd:
                    msg = (
                        f"🛑 [AI BUDGET LIMIT REACHED] Расход Claude API за {usage['month']} достиг "
                        f"${spent:.2f} (лимит ${self.monthly_limit_usd:.2f}). Запросы к ИИ отключены до 1-го числа; "
                        f"4 торговых бота, Watchdog и кнопки Telegram продолжают работать."
                    )
                else:
                    msg = (
                        f"⚠️ [AI BUDGET ALERT] Расход ИИ за {usage['month']} превысил ${th:.0f} "
                        f"(текущий расход: ${spent:.2f} из ${self.monthly_limit_usd:.2f})."
                    )
                logger.warning(msg)
                if self.on_budget_alert is not None:
                    try:
                        self.on_budget_alert(msg)
                    except Exception:
                        pass

        usage["alerted_thresholds"] = alerted
        self._save_usage(usage)

    def ask(self, prompt: str, context_data: Optional[Dict[str, Any]] = None, max_tokens: int = 700) -> Optional[str]:
        self.reload_env()
        if not self.is_configured:
            return None

        usage = self._load_usage()
        if float(usage.get("spent_usd", 0.0)) >= self.monthly_limit_usd:
            logger.warning("AI call skipped: monthly budget cap $%.2f reached.", self.monthly_limit_usd)
            return None

        full_user_text = prompt
        if context_data is not None:
            compact_json = json.dumps(context_data, ensure_ascii=False, default=str)
            full_user_text = f"{prompt}\n\nДАННЫЕ (JSON):\n{compact_json}"

        # 1. Prefer Anthropic Claude Native API if ANTHROPIC_API_KEY is set
        if self.anthropic_key and "your_" not in self.anthropic_key.lower():
            try:
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": self.anthropic_model,
                    "max_tokens": max_tokens,
                    "system": self.SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": full_user_text}],
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=(8, 15))
                if resp.status_code == 200:
                    data = resp.json()
                    u = data.get("usage", {})
                    self._record_usage(
                        int(u.get("input_tokens", 0)),
                        int(u.get("output_tokens", 0)),
                        in_rate_per_m=4.0,
                        out_rate_per_m=20.0,
                    )
                    blocks = data.get("content", [])
                    texts = [str(b.get("text", "")) for b in blocks if b.get("type") == "text"]
                    if texts:
                        return "\n".join(texts).strip()
                logger.warning("Anthropic API non-200 (%s): %s", resp.status_code, redact_secrets(resp.text[:250]))
            except Exception as exc:
                logger.warning("Anthropic API request failed: %s", redact_secrets(exc))

        # 2. Google Gemini Native API if GEMINI_API_KEY is set
        if self.gemini_key and "your_" not in self.gemini_key.lower():
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
                headers = {"x-goog-api-key": self.gemini_key, "Content-Type": "application/json"}
                payload = {
                    "system_instruction": {"parts": [{"text": self.SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": full_user_text}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=(8, 15))
                if resp.status_code == 200:
                    data = resp.json()
                    meta = data.get("usageMetadata", {})
                    self._record_usage(
                        int(meta.get("promptTokenCount", 0)),
                        int(meta.get("candidatesTokenCount", 0)),
                        in_rate_per_m=0.3,
                        out_rate_per_m=2.5,
                    )
                    cands = data.get("candidates", [])
                    if cands:
                        parts = cands[0].get("content", {}).get("parts", [])
                        if parts:
                            return str(parts[0].get("text", "")).strip()
                logger.warning("Gemini API non-200 (%s): %s", resp.status_code, redact_secrets(resp.text[:250]))
            except Exception as exc:
                logger.warning("Gemini API request failed: %s", redact_secrets(exc))

        # 3. Fallback / Alternative: OpenAI / OpenRouter / Custom Token endpoint
        if self.ai_key and "your_" not in self.ai_key.lower():
            try:
                url = f"{self.ai_base_url}/chat/completions"
                headers = {"Authorization": f"Bearer {self.ai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": self.ai_model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": full_user_text},
                    ],
                    "temperature": 0.2,
                    "max_tokens": max_tokens,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=(8, 15))
                if resp.status_code == 200:
                    data = resp.json()
                    u = data.get("usage", {})
                    self._record_usage(
                        int(u.get("prompt_tokens", 0)),
                        int(u.get("completion_tokens", 0)),
                        in_rate_per_m=4.0,
                        out_rate_per_m=20.0,
                    )
                    choices = data.get("choices", [])
                    if choices:
                        return str(choices[0].get("message", {}).get("content", "")).strip()
                logger.warning("AI_BASE_URL non-200 (%s): %s", resp.status_code, redact_secrets(resp.text[:250]))
            except Exception as exc:
                logger.warning("AI_BASE_URL request failed: %s", redact_secrets(exc))

        return None


# ----------------------------------------------------------------------
# Subprocess Launcher for Heavy R&D Coin Scout (with Lock Guard)
# ----------------------------------------------------------------------
_rnd_proc: Optional[subprocess.Popen] = None


def _rnd_running(pid: Any) -> bool:
    if _rnd_proc is not None and pid == _rnd_proc.pid:
        # poll() also reaps the finished child, otherwise it stays a zombie and looks alive
        return _rnd_proc.poll() is None
    return pid_alive(pid)


def trigger_rnd_scout_subprocess() -> Tuple[bool, str]:
    """Launch R&D Coin Scout in a separate background Python process with lock protection."""
    global _rnd_proc
    lock_info = read_rnd_lock()
    if lock_info is not None:
        pid = lock_info.get("pid")
        try:
            started = datetime.fromisoformat(str(lock_info.get("started_at")).replace("Z", "+00:00"))
            age_min = (utc_now() - started).total_seconds() / 60.0
        except (TypeError, ValueError):
            try:
                age_min = (time.time() - RND_LOCK_PATH.stat().st_mtime) / 60.0
            except OSError:
                age_min = 0.0
        if _rnd_running(pid):
            return False, f"⏳ R&D Coin Scout уже выполняется в фоне (PID {pid}, запущен {age_min:.1f} мин назад)."
        if not lock_info and age_min < 1.0:
            # lock is being written right now
            return False, "⏳ R&D Coin Scout уже запускается, подожди минуту."
        try:
            RND_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    if not write_rnd_lock(os.getpid(), exclusive=True):
        return False, "⏳ R&D Coin Scout уже запускается, подожди минуту."

    cmd = [sys.executable, "-m", "bot.supervisor_tools", "--rnd-scout", "--notify-telegram"]
    try:
        _rnd_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        try:
            RND_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"⚠️ Не удалось запустить фоновый R&D Coin Scout: {redact_secrets(exc)}"
    try:
        write_rnd_lock(_rnd_proc.pid)
    except OSError as exc:
        logger.warning("Could not update R&D lock with child pid: %s", exc)
    return True, "🔬 Запустил R&D Coin Scout (SOL, XRP, SUI, AVAX, DOGE на 180д, 3 фолда + OOS) в отдельном фоновом процессе. Как закончится расчёт (~30-60 сек) - пришлю готовый отчёт сюда."


# ----------------------------------------------------------------------
# Telegram Supervisor Bot (Polling + Inline Advisor Buttons + Schedules)
# ----------------------------------------------------------------------
class TelegramSupervisor:
    # Updates queued while the supervisor was down are dropped if older than this
    STALE_UPDATE_SEC = 60

    def __init__(self) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.allowed_user_ids = set(parse_id_list(os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")))
        self.chat_ids = parse_id_list(os.getenv("TELEGRAM_CHAT_ID", ""))
        self.target_chats = self.chat_ids or sorted(self.allowed_user_ids)
        self.ai = UniversalAIClient(on_budget_alert=self.send_message)
        self.offset = 0
        self.started_at = time.time()
        self.backlog_drained = False
        self._access_warned = False

    @property
    def telegram_ready(self) -> bool:
        return bool(self.token and "your_telegram" not in self.token.lower())

    @property
    def access_configured(self) -> bool:
        return bool(self.chat_ids or self.allowed_user_ids)

    def is_authorized(self, chat: Dict[str, Any], sender: Dict[str, Any]) -> bool:
        chat_id = str(chat.get("id", ""))
        user_id = str(sender.get("id", ""))
        if self.allowed_user_ids:
            if user_id not in self.allowed_user_ids:
                return False
            return chat.get("type") == "private" or chat_id in self.chat_ids
        return bool(chat_id) and chat_id in self.chat_ids

    def _api(self, method: str, payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
        if not self.telegram_ready:
            return {"ok": False}
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        resp = requests.post(url, json=payload, timeout=(8, timeout))
        if resp.status_code == 200:
            return resp.json()
        return {"ok": False, "status": resp.status_code, "text": redact_secrets(resp.text[:300])}

    def send_message(
        self,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        targets = [chat_id] if chat_id else self.target_chats
        if not self.telegram_ready or not targets:
            logger.info("[TG DRY OUTPUT]\n%s", text)
            return
        for target in targets:
            payload: Dict[str, Any] = {
                "chat_id": target,
                "text": text[:4000],
                "disable_web_page_preview": True,
            }
            if reply_markup is not None:
                payload["reply_markup"] = reply_markup
            try:
                res = self._api("sendMessage", payload, timeout=12)
                if not res.get("ok"):
                    logger.warning("Telegram send_message to %s failed: HTTP %s %s", target, res.get("status"), res.get("text", ""))
            except Exception as exc:
                logger.warning("Telegram send_message to %s failed: %s", target, redact_secrets(exc))

    @staticmethod
    def main_menu_keyboard() -> Dict[str, Any]:
        return {
            "inline_keyboard": [
                [
                    {"text": "📊 Позиции и баланс", "callback_data": "cmd:positions"},
                    {"text": "📈 Отчёт за сутки", "callback_data": "cmd:audit_1d"},
                ],
                [
                    {"text": "📅 Отчёт за 7 дней", "callback_data": "cmd:audit_7d"},
                    {"text": "🧭 Режим рынка (BTC/ETH)", "callback_data": "cmd:regime"},
                ],
                [
                    {"text": "📰 Новости по 4 монетам", "callback_data": "cmd:news"},
                    {"text": "🎯 Упущенные лимитки", "callback_data": "cmd:missed"},
                ],
                [
                    {"text": "💧 Стакан и потолки", "callback_data": "cmd:liquidity"},
                    {"text": "🔬 Поиск 5-й монеты", "callback_data": "cmd:rnd"},
                ],
                [
                    {"text": "⏸ Пауза WLFI 12ч", "callback_data": "pause:WLFIUSDT:12"},
                    {"text": "▶️ Снять паузы", "callback_data": "pause:ALL:0"},
                ],
            ]
        }

    @staticmethod
    def advisor_pause_keyboard(symbol: str = "WLFIUSDT") -> Dict[str, Any]:
        short_sym = symbol.replace("USDT", "")
        return {
            "inline_keyboard": [
                [
                    {"text": f"⏸ Пауза {short_sym} на 12ч", "callback_data": f"pause:{symbol}:12"},
                    {"text": f"⏸ Пауза {short_sym} на 24ч", "callback_data": f"pause:{symbol}:24"},
                ],
                [
                    {"text": f"▶️ Снять паузу с {short_sym}", "callback_data": f"pause:{symbol}:0"},
                ],
            ]
        }

    # ------------------------------------------------------------------
    # Report Builders (Deterministic + Optional AI Commentary)
    # ------------------------------------------------------------------
    def build_positions_text(self) -> str:
        data = audit_trades_and_hidden_costs(days=1)
        w = data["wallet"]
        active = data["active_trades"]
        pauses = data["pauses"]["active_entries"]
        lines = [
            f"📊 Статус портфеля ({data['mode'].upper()})",
            f"💰 Баланс (Equity): ${w.get('equity', 0.0):,.2f} | Доступно: ${w.get('available', 0.0):,.2f}",
        ]
        if data["pauses"].get("error"):
            lines.append(f"⚠️ {data['pauses']['error']}")
        if pauses:
            p_str = ", ".join(f"{e['symbol']} (до {e.get('expires_at', '—')})" for e in pauses)
            lines.append(f"⏸ Активные паузы: {p_str}")
        else:
            lines.append("▶️ Все 4 бота активны (пауз нет)")

        if not active:
            lines.append("\nОткрытых позиций и висящих лимиток сейчас нет.")
        else:
            lines.append(f"\nАктивные позиции / лимитные ордера ({len(active)}):")
            for t in active:
                d_badge = "🟢 LONG" if str(t.get("direction", "")).upper() == "LONG" else "🔴 SHORT"
                st_label = "⏳ Лимитка" if t["status"] == "PENDING" else f"🚀 В сделке ({t['status']})"
                lines.append(
                    f"• {t['symbol']} {d_badge} | 10x Isolated | {st_label}\n"
                    f"  Вход: {t['entry_price']} | TP: {t['tp1_price']} | SL: {t['sl_price']} | Маржа: ${float(t.get('margin_usdt') or 0):.2f}"
                )
        return "\n".join(lines)

    def build_audit_report(self, days: int = 1, title: str = "Аудит сделок и скрытых комиссий") -> str:
        data = audit_trades_and_hidden_costs(days=days)
        w = data["wallet"]
        ps = data["portfolio_summary"]
        hc = data["exchange_hidden_costs"]
        bench = ps["benchmark"]

        lines = [
            f"📈 {title} (за {days} дн. | {data['mode'].upper()})",
            f"💰 Equity: ${w.get('equity', 0.0):,.2f} (Доступно: ${w.get('available', 0.0):,.2f})",
            f"• Сделок: {ps['trades']} | Win Rate: {ps['win_rate_pct']}% (эталон NEW: {bench['win_rate_pct']}%)",
            f"• Profit Factor: {ps['profit_factor']} (эталон NEW: {bench['profit_factor']})",
            f"• Чистый PnL (БД): {ps['net_pnl_db_usdt']:+.2f} USDT",
            f"• Скрытые издержки Bybit: Фандинг {hc['funding_net_usdt']:+.3f} USDT | "
            f"Комиссии биржи {hc['exchange_exec_fees_usdt']:.3f} USDT (Maker: {hc['maker_fills']}, Taker: {hc['taker_fills']})",
            f"• Итого с учётом фандинга: {ps['net_after_funding_usdt']:+.2f} USDT",
            "\nПо монетам:",
        ]
        for sym, st in data["per_symbol"].items():
            lines.append(
                f"• {sym}: N={st['trades']} | WR {st['win_rate_pct']}% (эт. {st['benchmark_wr_pct']}%) | "
                f"SL={st['sl_count']} | Flip={st['h1_flip_exits']} | Net {st['net_pnl_usdt']:+.2f}$"
            )

        ai_note = self.ai.ask(
            "Дай короткий (2-3 предложения) вывод аудитора по результатам сделок и скрытым комиссиям в сравнении с эталоном пакета NEW.",
            context_data=data,
        )
        if ai_note:
            lines.append(f"\n🤖 Вывод ИИ-аудитора:\n{ai_note}")
        return "\n".join(lines)

    def build_market_regime_report(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        reg = compute_market_regime()
        btc = reg["assets"].get("BTCUSDT", {})
        eth = reg["assets"].get("ETHUSDT", {})
        if not reg.get("data_ok", True):
            lines = [
                f"🧭 Market Strategist - Режим рынка: {reg['regime']}",
                "⚠️ Нет данных: свечи BTC 15m не получены или их меньше 200, режим рынка не определён.",
                f"• BTCUSDT: ${btc.get('last_price', 0):,.1f} | 24h {btc.get('price_24h_pcnt', 0):+.2f}% | "
                f"Funding {btc.get('funding_rate_pct', 0):+.4f}%",
                f"• Совет по WLFI: {reg['wlfi_advisor_note']}",
            ]
            kb = self.advisor_pause_keyboard("WLFIUSDT") if reg["regime"] == "PANIC_FLUSH" else None
            return "\n".join(lines), kb
        lines = [
            f"🧭 Market Strategist — Режим рынка: {reg['regime']}",
            f"• BTCUSDT: ${btc.get('last_price', 0):,.1f} | 1H {btc.get('ret_1h_pct', 0):+.2f}% | "
            f"4H {btc.get('ret_4h_pct', 0):+.2f}% ({btc.get('trend_4h')}, ADX={btc.get('adx_4h')}) | "
            f"1D {btc.get('trend_1d')} | Funding {btc.get('funding_rate_pct', 0):+.4f}%",
            f"• ETHUSDT: ${eth.get('last_price', 0):,.1f} | 4H {eth.get('ret_4h_pct', 0):+.2f}% "
            f"({eth.get('trend_4h')}, ADX={eth.get('adx_4h')}) | Funding {eth.get('funding_rate_pct', 0):+.4f}%",
            f"• Совет по WLFI: {reg['wlfi_advisor_note']}",
        ]
        ai_note = self.ai.ask(
            "Оцени режим рынка по BTC/ETH 4H/1D и фандингу в 2-3 предложениях. Нужно ли предложить пользователю поставить контртрендовый WLFI на паузу?",
            context_data=reg,
        )
        if ai_note:
            lines.append(f"\n🤖 Комментарий ИИ-стратега:\n{ai_note}")

        kb = self.advisor_pause_keyboard("WLFIUSDT") if reg["regime"] == "PANIC_FLUSH" else None
        return "\n".join(lines), kb

    def build_news_report(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        digest = fetch_coin_news_digest()
        items = digest.get("coin_and_risk_headlines", [])
        lines = ["📰 News Scout — Крипто-новости по портфелю (ADA, WLFI, LTC, MNT):"]
        if not items:
            lines.append("• Экстренных новостей, делистингов, взломов или крупных разлоков по нашим 4 монетам не обнаружено.")
        else:
            for it in items[:6]:
                lines.append(f"• [{it['tags']}] {it['title']} ({it['source']})")

        ai_note = self.ai.ask(
            "Проверь заголовки новостей. Есть ли прямая угроза для ADAUSDT, WLFIUSDT, LTCUSDT или MNTUSDT (делистинг, взлом, разлок, суд)? Ответь в 2 предложениях.",
            context_data=digest,
        )
        if ai_note:
            lines.append(f"\n🤖 Оценка ИИ-разведчика:\n{ai_note}")

        kb = None
        for sym in ("WLFIUSDT", "ADAUSDT", "LTCUSDT", "MNTUSDT"):
            if any(sym in it.get("tags", "") and "RISK" in it.get("tags", "") for it in items):
                kb = self.advisor_pause_keyboard(sym)
                break
        return "\n".join(lines), kb

    def build_missed_fills_report(self) -> str:
        mf = audit_missed_fills(days=14)
        lines = ["🎯 Missed Fill Auditor — Упущенные лимитки (за 14 дн.):"]
        for sym, st in mf["per_symbol"].items():
            lines.append(
                f"• {sym}: Отменено по TTL={st['cancelled_ttl']} | Не дошло <=0.12% до входа: {st['near_miss_010_pct']} "
                f"(из них ушли к TP: {st['would_hit_tp']}, к SL: {st['would_hit_sl']})"
            )
        ai_note = self.ai.ask(
            "Проанализируй статистику отменённых по TTL лимитных ордеров. Стоит ли менять отступ входа или текущие настройки оптимальны? (2 предложения)",
            context_data=mf,
        )
        if ai_note:
            lines.append(f"\n🤖 Вывод ИИ:\n{ai_note}")
        return "\n".join(lines)

    def build_liquidity_report(self) -> str:
        aud = audit_trades_and_hidden_costs(days=1)
        eq = float(aud["wallet"].get("equity") or 2000.0)
        liq = check_liquidity_and_caps(equity=eq)
        lines = [f"💧 Liquidity & Cap Monitor (Депозит ${eq:,.2f}):"]
        for sym, st in liq["per_symbol"].items():
            lines.append(
                f"• {sym}: Потолок ${st['current_cap_usdt']:,.0f} (текущий ордер ${st['current_order_notional_usdt']:,.0f}) | "
                f"Глубина ±0.3%: ${st['depth_within_03pct_usdt']:,.0f} | Спред: {st['spread_bps']} bps | {st['recommendation']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Command & Natural-Language Handler (Russian)
    # ------------------------------------------------------------------
    def handle_user_text(self, text: str, chat_id: str) -> None:
        low = text.strip().lower()

        if low in ("/start", "/menu", "меню", "помощь", "/help"):
            self.send_message(
                "🤖 ИИ-супервайзер портфеля (ADA, WLFI, LTC, MNT) в режиме СОВЕТНИКА.\n"
                "Выбери действие кнопкой ниже или напиши команду обычным текстом:",
                reply_markup=self.main_menu_keyboard(),
                chat_id=chat_id,
            )
            return

        # 1. Pause / unpause commands. Unpause is checked first: "Включи ADA, сними паузу" must not pause ADA.
        intent = parse_pause_intent(low)
        if intent is not None:
            self.handle_pause_command(intent, text, low, chat_id)
            return

        # 3. Positions & Balance
        if any(w in low for w in ("позици", "баланс", "статус", "ордер")):
            self.send_message(self.build_positions_text(), reply_markup=self.main_menu_keyboard(), chat_id=chat_id)
            return

        # 4. Daily / Weekly Trade Audit
        if "недел" in low or "7 д" in low:
            self.send_message(self.build_audit_report(days=7, title="Отчёт за неделю"), chat_id=chat_id)
            return
        if any(w in low for w in ("отчет", "отчёт", "сутк", "день", "сделк", "комисс", "фандинг")):
            self.send_message(self.build_audit_report(days=1, title="Отчёт за сутки"), chat_id=chat_id)
            return

        # 5. Missed Fills
        if "упущен" in low or "лимитк" in low or "ttl" in low:
            self.send_message(self.build_missed_fills_report(), chat_id=chat_id)
            return

        # 6. Liquidity & Caps
        if "стакан" in low or "ликвидн" in low or "потол" in low:
            self.send_message(self.build_liquidity_report(), chat_id=chat_id)
            return

        # 7. Market Regime
        if "режим" in low or "рынок" in low or "биткоин" in low or "btc" in low:
            msg, kb = self.build_market_regime_report()
            self.send_message(msg, reply_markup=kb, chat_id=chat_id)
            return

        # 8. News Scout
        if "новост" in low:
            msg, kb = self.build_news_report()
            self.send_message(msg, reply_markup=kb, chat_id=chat_id)
            return

        # 9. AI Budget Status
        if any(w in low for w in ("бюджет", "расход", "лимит", "токен")):
            b = self.ai.get_budget_status()
            st = "🛑 ЛИМИТ ИСЧЕРПАН (ИИ отключён)" if b["halted"] else "✅ Активен"
            self.send_message(
                f"💳 Бюджет ИИ ({b['month']}): {st}\n"
                f"• Потрачено: ${b['spent_usd']:.2f} из ${b['limit_usd']:.2f} (осталось ${b['remaining_usd']:.2f})\n"
                f"• Запросов: {b['requests']} | Токенов: {b['input_tokens']:,} in / {b['output_tokens']:,} out\n"
                f"• Оповещения настроены на: $30, $40, $45 и стоп на $50.",
                chat_id=chat_id,
            )
            return

        # 10. R&D 5th Coin Scout (heavy, so only an explicit command or the menu button)
        if is_rnd_command(low):
            _, msg = trigger_rnd_scout_subprocess()
            self.send_message(msg, chat_id=chat_id)
            return

        # 10. Free-form AI question with live portfolio + market context
        ctx = {
            "portfolio": audit_trades_and_hidden_costs(days=3),
            "market_regime": compute_market_regime(),
        }
        ans = self.ai.ask(f"Вопрос пользователя в Telegram: {text}", context_data=ctx)
        if ans:
            self.send_message(ans, reply_markup=self.main_menu_keyboard(), chat_id=chat_id)
        else:
            self.send_message(
                self.build_positions_text() + "\n\n(ИИ-ключ ещё не указан в .env — доступны все кнопки и команды меню)",
                reply_markup=self.main_menu_keyboard(),
                chat_id=chat_id,
            )

    def handle_pause_command(self, intent: str, text: str, low: str, chat_id: str) -> None:
        symbols, unknown, wants_all = extract_pause_symbols(low)
        coins_hint = "Бот торгует только " + ", ".join(s.replace("USDT", "") for s in BOT_SYMBOLS) + "."
        if unknown:
            self.send_message(f"⚠️ Не знаю монету {', '.join(unknown)}. {coins_hint} Ничего не изменил.", chat_id=chat_id)
            return

        if intent == "unpause":
            try:
                if symbols:
                    for sym in symbols:
                        set_coin_pause(sym, hours=0)
                    self.send_message(f"▶️ Пауза с {', '.join(symbols)} снята. Бот снова в работе.", chat_id=chat_id)
                else:
                    for sym in BOT_SYMBOLS:
                        set_coin_pause(sym, hours=0)
                    self.send_message("▶️ Все паузы со всех 4 монет сняты.", chat_id=chat_id)
            except BlacklistError as exc:
                self.send_message(f"⚠️ Паузы не изменены: {exc}", chat_id=chat_id)
            return

        if not symbols or wants_all:
            self.send_message(f"⚠️ Укажи одну монету для паузы, например: «поставь WLFI на паузу на 12ч». {coins_hint}", chat_id=chat_id)
            return
        hours = 12.0
        m = HOURS_RE.search(low)
        if m:
            per_unit = 1.0 if m.group(2) in ("ч", "час", "h") else 24.0
            try:
                raw_hours: Any = float(m.group(1).replace(",", ".")) * per_unit
            except ValueError:
                raw_hours = None
            hours = parse_pause_hours(raw_hours)
            if hours is None:
                self.send_message(f"⚠️ Срок паузы должен быть больше 0 и не больше {MAX_PAUSE_HOURS:.0f}ч (7 дней).", chat_id=chat_id)
                return
        try:
            for sym in symbols:
                res = set_coin_pause(sym, hours=hours, reason=f"User command: {text[:60]}")
                self.send_message(
                    f"⏸ {res['symbol']} поставлен на паузу на {hours:g}ч (до {res['expires_at']}).\n"
                    f"По истечении таймера бот автоматически вернётся в работу.",
                    reply_markup=self.advisor_pause_keyboard(sym),
                    chat_id=chat_id,
                )
        except BlacklistError as exc:
            self.send_message(f"⚠️ Пауза не поставлена: {exc}", chat_id=chat_id)

    def handle_pause_callback(self, data: str, chat_id: str) -> None:
        parts = data.split(":")
        if len(parts) != 3:
            return
        sym, hrs_str = parts[1], parts[2]
        if sym != "ALL" and sym not in BOT_SYMBOLS:
            logger.warning("Rejected pause callback with unknown symbol: %r", sym[:20])
            self.send_message(f"⚠️ Неизвестная монета в кнопке: {sym[:20]}", chat_id=chat_id)
            return
        try:
            if sym == "ALL":
                for s in BOT_SYMBOLS:
                    set_coin_pause(s, hours=0)
                self.send_message("▶️ Все паузы со всех 4 монет сняты.", chat_id=chat_id)
            elif hrs_str.strip() == "0":
                set_coin_pause(sym, hours=0)
                self.send_message(f"▶️ Пауза с {sym} снята. Бот снова в работе.", chat_id=chat_id)
            else:
                hrs = parse_pause_hours(hrs_str)
                if hrs is None:
                    logger.warning("Rejected pause callback with bad hours: %r", hrs_str[:20])
                    self.send_message(f"⚠️ Некорректный срок паузы. Допустимо от 0 до {MAX_PAUSE_HOURS:.0f}ч.", chat_id=chat_id)
                    return
                res = set_coin_pause(sym, hours=hrs, reason=f"Telegram button pause {hrs:g}h")
                self.send_message(
                    f"⏸ {sym} поставлен на паузу на {hrs:g}ч (до {res['expires_at']}).",
                    reply_markup=self.advisor_pause_keyboard(sym),
                    chat_id=chat_id,
                )
        except BlacklistError as exc:
            self.send_message(f"⚠️ Паузы не изменены: {exc}", chat_id=chat_id)

    def handle_callback(self, cb: Dict[str, Any]) -> None:
        cb_id = cb.get("id")
        data = str(cb.get("data", ""))
        msg = cb.get("message") or {}
        chat = msg.get("chat") or {}
        chat_id = str(chat.get("id", ""))
        if not self.is_authorized(chat, cb.get("from") or {}):
            logger.warning(
                "Rejected callback from user %s in chat %s",
                (cb.get("from") or {}).get("id"),
                chat_id or "?",
            )
            return

        try:
            self._api("answerCallbackQuery", {"callback_query_id": cb_id})
        except Exception:
            pass

        if data.startswith("pause:"):
            self.handle_pause_callback(data, chat_id)
            return

        if data == "cmd:positions":
            self.send_message(self.build_positions_text(), chat_id=chat_id)
        elif data == "cmd:audit_1d":
            self.send_message(self.build_audit_report(days=1, title="Отчёт за сутки"), chat_id=chat_id)
        elif data == "cmd:audit_7d":
            self.send_message(self.build_audit_report(days=7, title="Отчёт за 7 дней"), chat_id=chat_id)
        elif data == "cmd:regime":
            txt, kb = self.build_market_regime_report()
            self.send_message(txt, reply_markup=kb, chat_id=chat_id)
        elif data == "cmd:news":
            txt, kb = self.build_news_report()
            self.send_message(txt, reply_markup=kb, chat_id=chat_id)
        elif data == "cmd:missed":
            self.send_message(self.build_missed_fills_report(), chat_id=chat_id)
        elif data == "cmd:liquidity":
            self.send_message(self.build_liquidity_report(), chat_id=chat_id)
        elif data == "cmd:rnd":
            _, txt = trigger_rnd_scout_subprocess()
            self.send_message(txt, chat_id=chat_id)

    def poll_telegram_once(self) -> bool:
        """One getUpdates round. Returns False when the caller should back off before the next one."""
        if not self.telegram_ready:
            return False
        if not self.access_configured:
            if not self._access_warned:
                logger.warning(
                    "TELEGRAM_CHAT_ID and TELEGRAM_ALLOWED_USER_IDS are empty: Telegram commands are disabled. "
                    "Set them in .env and restart the supervisor."
                )
                self._access_warned = True
            return False

        backlog = not self.backlog_drained
        try:
            res = self._api("getUpdates", {"offset": self.offset, "timeout": 0 if backlog else 5}, timeout=12)
        except Exception as exc:
            logger.warning("Telegram getUpdates failed: %s", redact_secrets(exc))
            return False
        if not res.get("ok"):
            logger.warning("Telegram getUpdates not ok: HTTP %s %s", res.get("status"), res.get("text", ""))
            return False

        updates = res.get("result", [])
        if backlog and not updates:
            self.backlog_drained = True
        for upd in updates:
            self.offset = max(self.offset, int(upd["update_id"]) + 1)
            try:
                self.handle_update(upd, backlog=backlog)
            except Exception as exc:
                logger.exception("Telegram update %s failed: %s", upd.get("update_id"), redact_secrets(exc))
        return True

    def handle_update(self, upd: Dict[str, Any], backlog: bool = False) -> None:
        if "callback_query" in upd:
            if backlog:
                # Button presses carry no own timestamp, so the ones queued during downtime are dropped
                logger.info("Skipped queued callback %s from before startup", upd.get("update_id"))
                return
            self.handle_callback(upd["callback_query"])
            return
        m = upd.get("message")
        if not m:
            return
        if backlog and int(m.get("date") or 0) < self.started_at - self.STALE_UPDATE_SEC:
            logger.info("Skipped stale queued message %s", upd.get("update_id"))
            return
        chat = m.get("chat") or {}
        cid = str(chat.get("id", ""))
        txt = str(m.get("text", "")).strip()
        if not txt:
            return
        if not self.is_authorized(chat, m.get("from") or {}):
            logger.warning("Rejected message from user %s in chat %s", (m.get("from") or {}).get("id"), cid)
            return
        self.handle_user_text(txt, chat_id=cid)


PAUSE_WORD = "пауз"
UNPAUSE_VERBS = ("сними", "снять", "снимай", "убери", "убрать", "отмени", "отменить", "сбрось", "сбросить", "возобнов")
PAUSE_VERBS = ("постав", "стоп", "останов", "приостан", "выключ", "включи паузу", "включить паузу", "на паузу")
HOURS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ч|час|h|дн|день|дня|сут|d\b)")
_LATIN_WORD_RE = re.compile(r"[a-z][a-z0-9]*")
_NOT_A_COIN = {"usdt", "all", "h", "tp", "sl", "ttl"}
_ALL_WORDS = ("все", "всё", "всех", "all")


def parse_pause_intent(low: str) -> Optional[str]:
    """'unpause', 'pause' or None. Unpause wins when both kinds of words are present."""
    if PAUSE_WORD not in low:
        return None
    if any(v in low for v in UNPAUSE_VERBS):
        return "unpause"
    if any(v in low for v in PAUSE_VERBS):
        return "pause"
    return None


def extract_pause_symbols(low: str) -> Tuple[List[str], List[str], bool]:
    """Bot symbols named in the text, unknown latin tickers, and whether 'все/all' was said."""
    known = {s.replace("USDT", "").lower(): s for s in BOT_SYMBOLS}
    found: List[str] = []
    unknown: List[str] = []
    for word in _LATIN_WORD_RE.findall(low):
        base = word[:-4] if word.endswith("usdt") and len(word) > 4 else word
        if base in _NOT_A_COIN:
            continue
        if base in known:
            if known[base] not in found:
                found.append(known[base])
        elif len(base) >= 2 and base.upper() not in unknown:
            unknown.append(base.upper())
    words = re.findall(r"\w+", low)
    wants_all = any(w in _ALL_WORDS for w in words)
    return found, unknown, wants_all


def is_rnd_command(low: str) -> bool:
    low = low.strip()
    if low in ("/rnd", "r&d", "/scout"):
        return True
    return any(k in low for k in ("5-й монет", "5-ой монет", "пятой монет", "пятую монет", "r&d scout", "запусти r&d"))


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Supervisor & Telegram Advisor for 4 Bybit Bots")
    parser.add_argument("--once", action="store_true", help="Run a single diagnostic summary and exit")
    args = parser.parse_args()

    setup_logging()
    sup = TelegramSupervisor()
    mode = resolve_bybit_mode()
    logger.info(
        "AI Supervisor started (ADVISOR MODE) | Bybit Mode: %s | Telegram Ready: %s | AI Configured: %s",
        mode.upper(),
        sup.telegram_ready,
        sup.ai.is_configured,
    )

    if args.once:
        print(sup.build_positions_text())
        print("-" * 60)
        txt, _ = sup.build_market_regime_report()
        print(txt)
        return

    if sup.telegram_ready and not sup.access_configured:
        logger.warning("Telegram commands disabled: set TELEGRAM_CHAT_ID and/or TELEGRAM_ALLOWED_USER_IDS in .env")

    last_4h_bucket = -1
    last_daily_key = ""
    last_weekly_key = ""
    poll_failures = 0

    while True:
        try:
            if sup.poll_telegram_once():
                poll_failures = 0
            else:
                poll_failures += 1
                time.sleep(min(60, 5 * 2 ** min(poll_failures - 1, 4)) if sup.telegram_ready and sup.access_configured else 5)
            now = utc_now()

            # 1. Every 4 hours: Market Strategist + News Scout (alert if PANIC_FLUSH or risk news)
            bucket_4h = int(now.timestamp()) // (4 * 3600)
            if bucket_4h != last_4h_bucket:
                if last_4h_bucket != -1:
                    reg = compute_market_regime()
                    if reg["regime"] == "PANIC_FLUSH":
                        txt, kb = sup.build_market_regime_report()
                        sup.send_message(f"⚠️ [MARKET STRATEGIST ALERT]\n{txt}", reply_markup=kb)
                    elif reg["regime"] == "UNKNOWN":
                        sup.send_message(
                            "⚠️ [MARKET STRATEGIST] Нет данных по свечам BTC, режим рынка не определён. "
                            "Проверка на пролив BTC за 4ч в этом цикле не выполнена."
                        )
                    news_txt, news_kb = sup.build_news_report()
                    if news_kb is not None:
                        sup.send_message(f"⚠️ [NEWS SCOUT ALERT]\n{news_txt}", reply_markup=news_kb)
                last_4h_bucket = bucket_4h

            # 2. Morning (07:00 UTC) and Evening (19:00 UTC) Automated Briefings
            day_hour_key = f"{now.strftime('%Y-%m-%d')}_{now.hour}"
            if now.hour in (7, 19) and now.minute < 5 and day_hour_key != last_daily_key:
                last_daily_key = day_hour_key
                label = "Утренняя сводка" if now.hour == 7 else "Вечерняя сводка"
                sup.send_message(sup.build_audit_report(days=1, title=label), reply_markup=sup.main_menu_keyboard())

            # 3. Weekly Sunday (10:00 UTC): Missed Fills + Liquidity & Caps + Bi-weekly R&D Coin Scout
            week_key = f"{now.strftime('%Y-%W')}_{now.weekday()}_{now.hour}"
            if now.weekday() == 6 and now.hour == 10 and now.minute < 5 and week_key != last_weekly_key:
                last_weekly_key = week_key
                sup.send_message(sup.build_missed_fills_report())
                sup.send_message(sup.build_liquidity_report())
                if int(now.strftime("%W")) % 2 == 0:
                    trigger_rnd_scout_subprocess()

        except Exception as exc:
            logger.exception("AI Supervisor loop error: %s", redact_secrets(exc))
            time.sleep(10)


if __name__ == "__main__":
    main()
