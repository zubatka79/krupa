"""AI Supervisor & Telegram Assistant (Advisor Mode) for the 4 Bybit Trading Bots.

CRITICAL SAFETY RULE (ADVISOR MODE):
- The AI NEVER generates 15M entry signals and NEVER modifies config_*.json or pauses coins autonomously.
- Autonomous scheduled tasks (News Scout, Market Strategist, Trade Auditor, Missed Fill, Liquidity Monitor,
  R&D Coin Scout) ONLY analyze and send recommendations/buttons to Telegram.
- Pausing or unpausing a coin in data/blacklist.json happens STRICTLY on explicit user command or button click
  from the authorized TELEGRAM_CHAT_ID.
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
    RND_LOCK_PATH,
    RND_RESULT_PATH,
    ROOT_DIR,
    audit_missed_fills,
    audit_trades_and_hidden_costs,
    check_liquidity_and_caps,
    compute_market_regime,
    create_bybit_client,
    fetch_coin_news_digest,
    get_active_pauses,
    resolve_bybit_mode,
    set_coin_pause,
    utc_now,
    utc_now_iso,
)

logger = logging.getLogger("ai_supervisor")
LOG_PATH = ROOT_DIR / "logs" / "ai_supervisor.log"


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ]
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

    def __init__(self) -> None:
        self.reload_env()

    def reload_env(self) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        self.ai_key = os.getenv("AI_API_KEY", "").strip()
        self.ai_base_url = os.getenv("AI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        self.ai_model = os.getenv("AI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    @property
    def is_configured(self) -> bool:
        has_g = bool(self.gemini_key and "your_" not in self.gemini_key.lower())
        has_a = bool(self.ai_key and "your_" not in self.ai_key.lower())
        return has_g or has_a

    def ask(self, prompt: str, context_data: Optional[Dict[str, Any]] = None, max_tokens: int = 700) -> Optional[str]:
        self.reload_env()
        if not self.is_configured:
            return None

        full_user_text = prompt
        if context_data is not None:
            compact_json = json.dumps(context_data, ensure_ascii=False, default=str)
            full_user_text = f"{prompt}\n\nДАННЫЕ (JSON):\n{compact_json}"

        # 1. Prefer Google Gemini Native API if GEMINI_API_KEY is set
        if self.gemini_key and "your_" not in self.gemini_key.lower():
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
                payload = {
                    "system_instruction": {"parts": [{"text": self.SYSTEM_PROMPT}]},
                    "contents": [{"role": "user", "parts": [{"text": full_user_text}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
                }
                resp = requests.post(url, json=payload, timeout=(8, 15))
                if resp.status_code == 200:
                    data = resp.json()
                    cands = data.get("candidates", [])
                    if cands:
                        parts = cands[0].get("content", {}).get("parts", [])
                        if parts:
                            return str(parts[0].get("text", "")).strip()
                logger.warning("Gemini API non-200 (%s): %s", resp.status_code, resp.text[:250])
            except Exception as exc:
                logger.warning("Gemini API request failed: %s", exc)

        # 2. Fallback / Alternative: OpenAI / OpenRouter / Custom Token endpoint
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
                    choices = data.get("choices", [])
                    if choices:
                        return str(choices[0].get("message", {}).get("content", "")).strip()
                logger.warning("AI_BASE_URL non-200 (%s): %s", resp.status_code, resp.text[:250])
            except Exception as exc:
                logger.warning("AI_BASE_URL request failed: %s", exc)

        return None


# ----------------------------------------------------------------------
# Subprocess Launcher for Heavy R&D Coin Scout (with Lock Guard)
# ----------------------------------------------------------------------
def trigger_rnd_scout_subprocess() -> Tuple[bool, str]:
    """Launch R&D Coin Scout in a separate background Python process with lock protection."""
    if RND_LOCK_PATH.exists():
        try:
            lock_info = json.loads(RND_LOCK_PATH.read_text(encoding="utf-8"))
            age_min = (
                utc_now() - datetime.fromisoformat(str(lock_info.get("started_at")).replace("Z", "+00:00"))
            ).total_seconds() / 60.0
            if age_min < 25.0:
                return False, f"⏳ R&D Coin Scout уже выполняется в фоне (PID {lock_info.get('pid')}, запущен {age_min:.1f} мин назад)."
        except Exception:
            pass
        try:
            RND_LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    cmd = [sys.executable, "-m", "bot.supervisor_tools", "--rnd-scout", "--notify-telegram"]
    try:
        subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True, "🔬 Запустил R&D Coin Scout (SOL, XRP, SUI, AVAX, DOGE на 180д, 3 фолда + OOS) в отдельном фоновом процессе. Как закончится расчёт (~30–60 сек) — пришлю готовый отчёт сюда."
    except Exception as exc:
        return False, f"⚠️ Не удалось запустить фоновый R&D Coin Scout: {exc}"


# ----------------------------------------------------------------------
# Telegram Supervisor Bot (Polling + Inline Advisor Buttons + Schedules)
# ----------------------------------------------------------------------
class TelegramSupervisor:
    def __init__(self) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.ai = UniversalAIClient()
        self.offset = 0

    @property
    def telegram_ready(self) -> bool:
        return bool(self.token and "your_telegram" not in self.token.lower())

    def _api(self, method: str, payload: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
        if not self.telegram_ready:
            return {"ok": False}
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        resp = requests.post(url, json=payload, timeout=(8, timeout))
        return resp.json() if resp.status_code == 200 else {"ok": False, "status": resp.status_code, "text": resp.text}

    def send_message(
        self,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        target = chat_id or self.chat_id
        if not self.telegram_ready or not target or "your_telegram" in target.lower():
            logger.info("[TG DRY OUTPUT]\n%s", text)
            return
        payload: Dict[str, Any] = {
            "chat_id": target,
            "text": text[:4000],
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        try:
            self._api("sendMessage", payload, timeout=12)
        except Exception as exc:
            logger.warning("Telegram send_message failed: %s", exc)

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

        # 1. Explicit Pause command: "поставь WLFI на паузу на 12 часов"
        if "пауз" in low and ("постав" in low or "включ" in low or "стоп" in low):
            target_sym = None
            for s in ("WLFI", "ADA", "LTC", "MNT"):
                if s.lower() in low:
                    target_sym = f"{s}USDT"
                    break
            if target_sym:
                m = re.search(r"(\d+)\s*(ч|час|h)", low)
                hours = float(m.group(1)) if m else 12.0
                res = set_coin_pause(target_sym, hours=hours, reason=f"User command: {text[:60]}")
                self.send_message(
                    f"⏸ {res['symbol']} поставлен на паузу на {hours:.0f}ч (до {res['expires_at']}).\n"
                    f"По истечении таймера бот автоматически вернётся в работу.",
                    reply_markup=self.advisor_pause_keyboard(target_sym),
                    chat_id=chat_id,
                )
                return

        # 2. Explicit Unpause command: "сними паузу с WLFI" / "сними паузы"
        if ("сними" in low or "убери" in low or "отмени" in low or "включи" in low) and "пауз" in low:
            target_sym = None
            for s in ("WLFI", "ADA", "LTC", "MNT"):
                if s.lower() in low:
                    target_sym = f"{s}USDT"
                    break
            if target_sym:
                set_coin_pause(target_sym, hours=0)
                self.send_message(f"▶️ Пауза с {target_sym} снята. Бот снова в работе.", chat_id=chat_id)
            else:
                for s in ("WLFIUSDT", "ADAUSDT", "LTCUSDT", "MNTUSDT"):
                    set_coin_pause(s, hours=0)
                self.send_message("▶️ Все паузы со всех 4 монет сняты.", chat_id=chat_id)
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

        # 9. R&D 5th Coin Scout
        if "5-й" in low or "5 монет" in low or "пятой" in low or "поиск" in low or "r&d" in low:
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

    def handle_callback(self, cb: Dict[str, Any]) -> None:
        cb_id = cb.get("id")
        data = str(cb.get("data", ""))
        msg = cb.get("message", {})
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if self.chat_id and "your_telegram" not in self.chat_id.lower() and chat_id != self.chat_id:
            return

        try:
            self._api("answerCallbackQuery", {"callback_query_id": cb_id})
        except Exception:
            pass

        if data.startswith("pause:"):
            parts = data.split(":")
            if len(parts) == 3:
                sym, hrs_str = parts[1], parts[2]
                hrs = float(hrs_str)
                if sym == "ALL":
                    for s in ("WLFIUSDT", "ADAUSDT", "LTCUSDT", "MNTUSDT"):
                        set_coin_pause(s, hours=0)
                    self.send_message("▶️ Все паузы со всех 4 монет сняты.", chat_id=chat_id)
                elif hrs <= 0:
                    set_coin_pause(sym, hours=0)
                    self.send_message(f"▶️ Пауза с {sym} снята. Бот снова в работе.", chat_id=chat_id)
                else:
                    res = set_coin_pause(sym, hours=hrs, reason=f"Telegram button pause {hrs:.0f}h")
                    self.send_message(
                        f"⏸ {sym} поставлен на паузу на {hrs:.0f}ч (до {res['expires_at']}).",
                        reply_markup=self.advisor_pause_keyboard(sym),
                        chat_id=chat_id,
                    )
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

    def poll_telegram_once(self) -> None:
        if not self.telegram_ready:
            return
        try:
            res = self._api("getUpdates", {"offset": self.offset, "timeout": 5}, timeout=12)
            for upd in res.get("result", []):
                self.offset = max(self.offset, int(upd["update_id"]) + 1)
                if "callback_query" in upd:
                    self.handle_callback(upd["callback_query"])
                elif "message" in upd:
                    m = upd["message"]
                    cid = str(m.get("chat", {}).get("id", ""))
                    txt = str(m.get("text", "")).strip()
                    if not txt:
                        continue
                    if not self.chat_id or "your_telegram" in self.chat_id.lower():
                        logger.info("Auto-registered first Telegram user message (saved TELEGRAM_CHAT_ID to .env)")
                        self.chat_id = cid
                        try:
                            env_p = ROOT_DIR / ".env"
                            if env_p.exists():
                                etxt = env_p.read_text(encoding="utf-8")
                                if "TELEGRAM_CHAT_ID=" in etxt:
                                    etxt = re.sub(r"^TELEGRAM_CHAT_ID=.*$", f"TELEGRAM_CHAT_ID={cid}", etxt, flags=re.M)
                                else:
                                    etxt += f"\nTELEGRAM_CHAT_ID={cid}\n"
                                env_p.write_text(etxt, encoding="utf-8")
                        except Exception:
                            pass
                    if cid == self.chat_id:
                        self.handle_user_text(txt, chat_id=cid)
        except Exception as exc:
            logger.debug("Telegram poll note: %s", exc)


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

    last_4h_bucket = -1
    last_daily_key = ""
    last_weekly_key = ""

    while True:
        try:
            sup.poll_telegram_once()
            now = utc_now()

            # 1. Every 4 hours: Market Strategist + News Scout (alert if PANIC_FLUSH or risk news)
            bucket_4h = int(now.timestamp()) // (4 * 3600)
            if bucket_4h != last_4h_bucket:
                if last_4h_bucket != -1:
                    reg = compute_market_regime()
                    if reg["regime"] == "PANIC_FLUSH":
                        txt, kb = sup.build_market_regime_report()
                        sup.send_message(f"⚠️ [MARKET STRATEGIST ALERT]\n{txt}", reply_markup=kb)
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
            logger.exception("AI Supervisor loop error: %s", exc)
            time.sleep(10)


if __name__ == "__main__":
    main()
