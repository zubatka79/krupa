"""
Complete Telegram Control Center for AI-Trader v2 (`ai_trader_v2/telegram_ui.py`).

Replaces the legacy 6-bot Advisor menu with a dedicated interactive interface for:
  - 29 liquid Bybit coins · 12 TA schools · 15m + 1H/4H/BTC · Variant 2 Consensus · Claude AI Judge
  - Live Bybit Demo / Testnet / Paper portfolio & open positions monitor
  - Recent AI verdicts (ENTER & SKIP with Russian reasoning)
  - 24h / 7d PnL reports & 12-school performance leaderboard
  - On-demand live scan of all 29 coins + live BTC 1H/4H & US Macro Shield status
  - Global pause/resume + per-coin pause/unpause + toggle SKIP notifications
  - Free-form Russian Q&A powered by Claude with live AI-Trader v2 context
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

from ai_trader_v2.ai_judge import AITradeJudge, redact_secrets
from ai_trader_v2.executor import AITradeExecutor
from ai_trader_v2.journal import AITradeJournal
from ai_trader_v2.ta_schools import ALL_12_SCHOOLS, build_btc_context
from bot.bybit_client import BybitClient
from bot.news_guard import NewsGuard

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
STATE_PATH = Path(os.getenv("AI_TRADER_DATA_DIR", str(ROOT_DIR / "data" / "ai_trader_v2"))) / "bot_state.json"
HOURS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ч|час|h|дн|день|дня|сут|d\b)", re.I)


def load_bot_state() -> Dict[str, Any]:
    """Load persisted pause / UI toggle state (global pause, per-coin pauses, send_skips)."""
    default: Dict[str, Any] = {
        "global_paused": False,
        "paused_coins": {},  # symbol -> ISO UTC expiry
        "telegram_send_skips": False,
    }
    if not STATE_PATH.exists():
        return default
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return default
        out = dict(default)
        out.update(data)
        if not isinstance(out.get("paused_coins"), dict):
            out["paused_coins"] = {}
        # Auto-prune expired coin pauses
        now = datetime.now(timezone.utc)
        cleaned: Dict[str, str] = {}
        for sym, exp_iso in out["paused_coins"].items():
            try:
                exp_dt = datetime.fromisoformat(str(exp_iso).replace("Z", "+00:00"))
                if exp_dt > now:
                    cleaned[sym.upper()] = str(exp_iso)
            except ValueError:
                pass
        out["paused_coins"] = cleaned
        return out
    except Exception:
        return default


def save_bot_state(state: Dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_name(f"{STATE_PATH.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def is_symbol_paused(symbol: str, state: Optional[Dict[str, Any]] = None) -> bool:
    st = state if state is not None else load_bot_state()
    if bool(st.get("global_paused", False)):
        return True
    return symbol.upper() in (st.get("paused_coins") or {})


class AITraderTelegramUI:
    """Interactive Telegram bot controller tailored exclusively for AI-Trader v2."""

    STALE_UPDATE_SEC = 120

    def __init__(
        self,
        client: BybitClient,
        executor: AITradeExecutor,
        journal: AITradeJournal,
        judge: AITradeJudge,
        news_guard: NewsGuard,
        config: Dict[str, Any],
        scan_callback: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        load_dotenv(override=True)
        self.client = client
        self.executor = executor
        self.journal = journal
        self.judge = judge
        self.news_guard = news_guard
        self.config = config
        self.scan_callback = scan_callback

        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        raw_users = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
        self.allowed_users: Set[str] = {u.strip() for u in raw_users.split(",") if u.strip()}

        self.offset = 0
        self.started_at = int(time.time())
        self.backlog_drained = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._scan_lock = threading.Lock()
        self._session = requests.Session()

    @property
    def is_ready(self) -> bool:
        return bool(
            self.bot_token
            and "your_" not in self.bot_token.lower()
            and (self.chat_id or self.allowed_users)
        )

    def _api(self, method: str, payload: Dict[str, Any], timeout: int = 12) -> Dict[str, Any]:
        if not self.bot_token:
            return {"ok": False}
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        resp = self._session.post(url, json=payload, timeout=(5, timeout))
        if resp.status_code != 200:
            return {"ok": False, "status": resp.status_code, "text": redact_secrets(resp.text[:200])}
        return resp.json()

    def is_authorized(self, chat: Dict[str, Any], user: Dict[str, Any]) -> bool:
        cid = str(chat.get("id", ""))
        uid = str(user.get("id", ""))
        if self.chat_id and cid == self.chat_id:
            return True
        if self.allowed_users and uid in self.allowed_users:
            return True
        return False

    def send(
        self,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        target = chat_id or self.chat_id
        if not self.bot_token or not target:
            return
        safe_txt = redact_secrets(text)
        # Split long messages safely under Telegram's 4096-char limit
        chunks = [safe_txt[i : i + 3900] for i in range(0, len(safe_txt), 3900)] or [""]
        for idx, chunk in enumerate(chunks):
            payload: Dict[str, Any] = {
                "chat_id": target,
                "text": chunk,
                "disable_web_page_preview": True,
            }
            if reply_markup is not None and idx == len(chunks) - 1:
                payload["reply_markup"] = reply_markup
            try:
                self._api("sendMessage", payload, timeout=10)
            except Exception as exc:
                logger.warning("Telegram send error: %s", redact_secrets(exc))

    def register_slash_commands(self) -> None:
        """Replace Telegram's native '/' command menu with AI-Trader v2 commands."""
        if not self.is_ready:
            return
        commands = [
            {"command": "menu", "description": "Главное меню AI-Trader v2"},
            {"command": "positions", "description": "Баланс, слоты и открытые сделки"},
            {"command": "ai", "description": "Последние вердикты Claude AI (ENTER / SKIP)"},
            {"command": "report", "description": "Отчёт за 24ч (PnL, WR, PF)"},
            {"command": "week", "description": "Отчёт за 7 дней"},
            {"command": "scan", "description": "Запустить скан 29 монет по 12 школам сейчас"},
            {"command": "market", "description": "Режим BTC 1H/4H, тренды и Макро-щит"},
            {"command": "schools", "description": "Статистика по 12 школам теханализа"},
            {"command": "budget", "description": "Бюджет Claude API и параметры движка"},
            {"command": "pause", "description": "Поставить новые входы на паузу"},
            {"command": "resume", "description": "Возобновить торговлю"},
        ]
        try:
            self._api("setMyCommands", {"commands": commands}, timeout=10)
        except Exception as exc:
            logger.debug("setMyCommands failed: %s", redact_secrets(exc))

    def main_menu_keyboard(self) -> Dict[str, Any]:
        st = load_bot_state()
        paused = bool(st.get("global_paused", False))
        pause_btn = (
            {"text": "▶️ Возобновить входы", "callback_data": "v2:resume_all"}
            if paused
            else {"text": "⏸ Пауза новых входов", "callback_data": "v2:pause_all"}
        )
        skips_on = bool(st.get("telegram_send_skips", False))
        skips_btn = {
            "text": "🔔 Скипы AI: ВКЛ" if skips_on else "🔕 Скипы AI: ВЫКЛ",
            "callback_data": "v2:toggle_skips",
        }
        return {
            "inline_keyboard": [
                [
                    {"text": "📊 Позиции и Баланс", "callback_data": "v2:positions"},
                    {"text": "🤖 Вердикты AI", "callback_data": "v2:ai_log"},
                ],
                [
                    {"text": "📈 Отчёт 24ч", "callback_data": "v2:report_1d"},
                    {"text": "📅 Отчёт 7д", "callback_data": "v2:report_7d"},
                ],
                [
                    {"text": "🔍 Скан 29 монет", "callback_data": "v2:scan_now"},
                    {"text": "🧭 Рынок + Макро", "callback_data": "v2:market"},
                ],
                [
                    {"text": "🏫 12 Школ ТА", "callback_data": "v2:schools"},
                    {"text": "💳 Бюджет AI", "callback_data": "v2:budget"},
                ],
                [pause_btn, skips_btn],
            ]
        }

    # ------------------------------------------------------------------
    # Report Builders (100% tailored to AI-Trader v2)
    # ------------------------------------------------------------------
    def build_welcome_text(self) -> str:
        st = load_bot_state()
        mode_lbl = "PAPER" if self.executor.paper_mode else f"BYBIT {self.client.mode.upper()}"
        status_lbl = "⏸ НА ПАУЗЕ (новые входы отключены)" if st["global_paused"] else "🟢 АКТИВЕН 24/7"
        paused_coins = st.get("paused_coins") or {}
        extra = f"\n• Монеты на паузе: {', '.join(paused_coins.keys())}" if paused_coins else ""
        return (
            f"🤖 AI-TRADER v2 — Панель управления\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"• Статус: {status_lbl}\n"
            f"• Биржа / Режим: {mode_lbl}\n"
            f"• AI-Судья: {self.judge.active_provider_label}\n"
            f"• Вселенная: {len(self.config['symbols'])} ликвидных монет · 12 школ ТА\n"
            f"• ТФ: 15m (фильтры 1H + 4H + BTC) · Консенсус: S12 или 2+ школы (RVOL >= {self.config.get('min_rvol_non_s12', 1.2)})\n"
            f"• Риск: {self.config['margin_pct_per_trade']:.0f}% × {self.config['leverage']}x | "
            f"Слоты: {self.config['max_open_positions']} (в 1 сторону: {self.config['max_same_direction']})\n"
            f"• SL: {self.config['sl_min_pct']}–{self.config['sl_max_pct']}% | TP1 55% → БУ +{self.config['breakeven_offset_pct']}% | "
            f"Дневной стоп: -{self.config['daily_loss_limit_pct']}%"
            f"{extra}\n\n"
            f"Выбери раздел кнопкой ниже или напиши команду текстом (например: «пауза ETH 12ч» или любой вопрос к Claude):"
        )

    def build_positions_text(self) -> str:
        st = load_bot_state()
        eq = self.executor.get_current_equity()
        avail = eq
        ex_positions: List[Dict[str, Any]] = []
        ex_orders: List[Dict[str, Any]] = []
        if not self.executor.paper_mode:
            try:
                bal = self.client.get_wallet_balance("USDT")
                eq = float(bal.get("equity", eq))
                avail = float(bal.get("available", eq))
                ex_positions = self.client.get_positions()
                ex_orders = self.client.get_open_orders()
            except Exception as exc:
                logger.warning("Failed fetching live exchange state: %s", exc)

        active = self.journal.get_active_trades()
        tickers = self.executor._safe_get_tickers()
        unr = self.executor.compute_unrealized_pnl(active, tickers)
        daily = self.journal.ensure_daily_row(eq + unr)
        start_eq = float(daily.get("start_equity") or eq or 1.0)
        day_chg_pct = ((eq + unr - start_eq) / start_eq) * 100.0 if start_eq > 0 else 0.0
        longs = sum(1 for t in active if t["direction"] == "LONG")
        shorts = sum(1 for t in active if t["direction"] == "SHORT")

        mode_lbl = "PAPER" if self.executor.paper_mode else f"BYBIT {self.client.mode.upper()}"
        halt_txt = " 🛑 ДНЕВНОЙ СТОП" if int(daily.get("halted", 0)) == 1 else ""
        pause_txt = " ⏸ ПАУЗА" if st["global_paused"] else ""

        lines = [
            f"📊 ПОЗИЦИИ И БАЛАНС ({mode_lbl}){pause_txt}{halt_txt}",
            f"• Баланс (Equity): ${eq:,.2f} (свободно: ${avail:,.2f})",
            f"• За UTC-сутки: {day_chg_pct:+.2f}% (лимит -{self.config['daily_loss_limit_pct']:.1f}%) | uPnL: ${unr:+.2f}",
            f"• Слоты AI-Trader: {len(active)}/{self.config['max_open_positions']} "
            f"(LONG {longs}/{self.config['max_same_direction']} | SHORT {shorts}/{self.config['max_same_direction']})",
        ]

        if not active:
            lines.append("• Активных сделок AI-Trader v2 сейчас нет.")
        else:
            lines.append("\n📌 Активные сделки:")
            for t in active:
                sym = t["symbol"]
                cp = float(tickers.get(sym, {}).get("last_price", 0.0))
                ep = float(t["entry_price"])
                qty = float(t["remaining_qty"])
                t_unr = ((cp - ep) * qty if t["direction"] == "LONG" else (ep - cp) * qty) if (cp > 0 and t["status"] != "PENDING_LIMIT") else 0.0
                ico = "🟢" if t["direction"] == "LONG" else "🔴"
                lines.append(
                    f"{ico} #{t['id']} {sym} {t['direction']} [{t['status']}] | AI {t['ai_confidence']}/100\n"
                    f"   Вход: {ep} | Тек: {cp or '-'} | uPnL: ${t_unr:+.2f}\n"
                    f"   SL: {t['sl_price']} | TP1: {t['tp1_price']} | TP2: {t['tp2_price']} | Школа: {t['primary_school']}"
                )

        if not self.executor.paper_mode:
            # Show any stray exchange positions/orders not tracked by AI-Trader v2
            tracked_syms = {t["symbol"] for t in active}
            extra_pos = [p for p in ex_positions if p.get("symbol") not in tracked_syms]
            extra_ord = [o for o in ex_orders if o.get("symbol") not in tracked_syms]
            if extra_pos or extra_ord:
                lines.append(f"\n⚠️ На бирже есть сторонние (не AI-Trader): позиций={len(extra_pos)}, ордеров={len(extra_ord)}")
                for o in extra_ord[:5]:
                    lines.append(f"   • Ордер {o.get('symbol')} {o.get('side')} {o.get('orderType')} @ {o.get('price')} (qty {o.get('qty')})")

        return "\n".join(lines)

    def build_ai_decisions_text(self, limit: int = 8) -> str:
        rows = self.journal.get_recent_ai_decisions(limit=limit)
        if not rows:
            return "🤖 Вердиктов AI-судьи пока нет — бот ждёт первого консенсус-сигнала (S12 или 2+ школы из 12) на закрытой 15m свече."
        lines = [f"🤖 ПОСЛЕДНИЕ {len(rows)} ВЕРДИКТОВ AI-СУДЬИ:"]
        for r in rows:
            ico = "✅" if r["decision"] == "ENTER" else "⏭️"
            dt_short = str(r["decided_at"])[5:16].replace("T", " ")
            try:
                schools = json.loads(r["schools_json"])
            except Exception:
                schools = [r["primary_school"]]
            lines.append(
                f"\n{ico} [{dt_short} UTC] {r['symbol']} {r['direction']} → {r['decision']} ({r['confidence']}/100)\n"
                f"   Школы ({len(schools)}): {', '.join(schools)}\n"
                f"   Цена {r['price']} | SL {r['sl_pct']}% | TP1 {r['tp1_r']}R / TP2 {r['tp2_r']}R\n"
                f"   💬 {r['reason_ru']}"
            )
        return "\n".join(lines)

    def build_period_report_text(self, days: int = 1) -> str:
        rep = self.journal.get_period_report(days=days)
        title = "ОТЧЁТ ЗА 24 ЧАСА" if days == 1 else f"ОТЧЁТ ЗА {days} ДНЕЙ"
        lines = [
            f"📈 {title} (AI-Trader v2)",
            f"• Закрыто сделок: {rep['closed_trades']} (Отменено лимиток по TTL: {rep['expired_limits']})",
            f"• Победы / Убытки: {rep['wins']} / {rep['losses']} | Win Rate: {rep['win_rate_pct']:.1f}%",
            f"• Разбивка выходов: TP2={rep['tp2_hits']} | TP1+БУ/Трейл={rep['be_trail_hits']} | SL={rep['sl_hits']}"
            + (f" | Биржа={rep['exchange_closed']}" if rep["exchange_closed"] else ""),
            f"• Чистый PnL: ${rep['net_pnl_usdt']:+.2f} | Profit Factor: {rep['profit_factor']:.2f} | Комиссии: ${rep['fees_usdt']:.2f}",
            f"• Работа AI-судьи: ENTER={rep['ai_enter']} | SKIP={rep['ai_skip']}",
        ]
        if rep["recent_closed"]:
            lines.append("\nПоследние закрытые:")
            for r in rep["recent_closed"]:
                ico = "🏆" if float(r["net_pnl_usdt"]) > 0 else "❌"
                lines.append(
                    f"  {ico} {r['symbol']} {r['direction']} | ${float(r['net_pnl_usdt']):+.2f} "
                    f"({float(r['net_pnl_pct']):+.1f}%) | {r['exit_reason']} | {r['primary_school']}"
                )
        return "\n".join(lines)

    def build_schools_text(self) -> str:
        stats = self.journal.get_school_stats()
        lines = ["🏫 СТАТИСТИКА 12 ШКОЛ ТЕХАНАЛИЗА:"]
        for sc in ALL_12_SCHOOLS:
            s = stats.get(sc)
            if not s:
                lines.append(f"• {sc}: 0 сигналов")
            else:
                wr = (s["wins"] / s["closed"] * 100.0) if s["closed"] > 0 else 0.0
                lines.append(
                    f"• {sc}: сиги {s['signals']} (ENTER {s['enter']}/SKIP {s['skip']}) | "
                    f"закр {s['closed']} (WR {wr:.0f}%) | PnL ${s['pnl']:+.2f}"
                )
        return "\n".join(lines)

    def build_market_text(self) -> str:
        btc = build_btc_context(self.client, shock_limit_pct=float(self.config.get("btc_shock_pause_pct", 0.9)))
        shield = self.news_guard.evaluate_macro_shield(allow_network_refresh=True)
        lines = ["🧭 РЫНОК (BTC 1H/4H + МАКРО-ЩИТ):"]
        if btc is None:
            lines.append("• ⚠️ Не удалось загрузить свечи BTCUSDT.")
        else:
            s = btc.summary_dict
            lines.append(
                f"• BTCUSDT: ${s['btc_price']:,.1f} | 15m: {s['btc_15m_move_pct']:+.2f}% | 6ч: {s['btc_6h_ret_pct']:+.2f}%"
            )
            lines.append(
                f"• Тренд BTC: 1H={btc.h1_trend} | 4H={btc.h4_trend} | Шок: {'ДА 🛑' if btc.shock_active else 'нет'}"
            )
            lines.append(
                f"• Разрешено для альтов: LONG={'✅' if btc.allow_alt_long else '🚫'} | SHORT={'✅' if btc.allow_alt_short else '🚫'}"
            )

        ok_macro = bool(shield.get("entry_allowed", True))
        lines.append(
            f"\n🛡️ US Macro Shield (CPI/FOMC/NFP): {'✅ Чисто' if ok_macro else '🛑 БЛОКИРОВКА: ' + str(shield.get('reason'))}"
        )
        upcoming = shield.get("upcoming_core_events") or []
        if upcoming:
            lines.append("Ближайшие важные события США (UTC):")
            for ev in upcoming[:4]:
                lines.append(f"  • {ev.get('time_utc', '?')} — {ev.get('title', '?')}")
        else:
            lines.append("• Ближайших критичных US макро-релизов в окне нет.")
        return "\n".join(lines)

    def build_budget_text(self) -> str:
        b = self.judge.get_budget_status()
        st = load_bot_state()
        return (
            f"💳 БЮДЖЕТ AI И ПАРАМЕТРЫ ДВИЖКА\n"
            f"• Провайдер: {self.judge.active_provider_label}\n"
            f"• Расход за {b['month']}: ${b['spent_usd']:.4f} из ${b['limit_usd']:.2f} "
            f"(осталось ${b['remaining_usd']:.2f})\n"
            f"• Вызовов AI: {b['requests']}\n"
            f"• Режим отбора: Вариант 2 (S12 или {self.config.get('min_schools_non_s12', 2)}+ школы & RVOL >= {self.config.get('min_rvol_non_s12', 1.2)})\n"
            f"• Мин. уверенность AI: {self.config.get('min_ai_confidence', 72)}/100\n"
            f"• Скипы AI в Telegram: {'ВКЛ' if st.get('telegram_send_skips') else 'ВЫКЛ'}"
        )

    # ------------------------------------------------------------------
    # Handlers (Callbacks + Commands + Free-form Russian)
    # ------------------------------------------------------------------
    def _run_manual_scan(self, chat_id: str) -> None:
        if not self.scan_callback:
            self.send("⚠️ Ручной скан не подключён.", chat_id=chat_id)
            return
        if not self._scan_lock.acquire(blocking=False):
            self.send("⏳ Скан 29 монет уже выполняется, подожди пару секунд...", chat_id=chat_id)
            return
        try:
            self.send("🔍 Сканирую все 29 монет по 12 школам ТА (15m + 1H/4H + BTC)...", chat_id=chat_id)
            res = self.scan_callback()
            lines = [
                f"✅ СКАН ЗАВЕРШЁН",
                f"• BTC: {res.get('btc_trend', '?')}",
                f"• Проверено монет: {res.get('scanned', 0)} | Кандидатов (Вариант 2): {res.get('candidates', 0)}",
                f"• Запросов к AI: {res.get('ai_called', 0)} | ENTER: {res.get('entered', 0)} | SKIP: {res.get('skipped', 0)}",
            ]
            if res.get("macro_blocked"):
                lines.append(f"• 🛑 Макро-щит: {res.get('macro_reason')}")
            for d in res.get("details", [])[:5]:
                ico = "✅" if d["decision"] == "ENTER" else "⏭️"
                lines.append(
                    f"\n{ico} {d['symbol']} {d['direction']} ({d['confidence']}/100)\n"
                    f"   Школы: {', '.join(d['schools'])}\n"
                    f"   💬 {d['reason_ru']}"
                )
            self.send("\n".join(lines), reply_markup=self.main_menu_keyboard(), chat_id=chat_id)
        except Exception as exc:
            self.send(f"⚠️ Ошибка ручного скана: {redact_secrets(exc)}", chat_id=chat_id)
        finally:
            self._scan_lock.release()

    def handle_callback(self, cb: Dict[str, Any]) -> None:
        cb_id = cb.get("id")
        data = str(cb.get("data", ""))
        msg = cb.get("message") or {}
        chat = msg.get("chat") or {}
        cid = str(chat.get("id", ""))
        if not self.is_authorized(chat, cb.get("from") or {}):
            return
        try:
            self._api("answerCallbackQuery", {"callback_query_id": cb_id}, timeout=5)
        except Exception:
            pass

        if data == "v2:positions":
            self.send(self.build_positions_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:ai_log":
            self.send(self.build_ai_decisions_text(8), reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:report_1d":
            self.send(self.build_period_report_text(1), reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:report_7d":
            self.send(self.build_period_report_text(7), reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:market":
            self.send(self.build_market_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:schools":
            self.send(self.build_schools_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:budget":
            self.send(self.build_budget_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:scan_now":
            threading.Thread(target=self._run_manual_scan, args=(cid,), daemon=True).start()
        elif data == "v2:pause_all":
            st = load_bot_state()
            st["global_paused"] = True
            save_bot_state(st)
            self.send(
                "⏸ Новые входы AI-Trader v2 поставлены на ПАУЗУ.\nОткрытые позиции продолжают сопровождаться (SL / TP1 → БУ / TP2).",
                reply_markup=self.main_menu_keyboard(),
                chat_id=cid,
            )
        elif data == "v2:resume_all":
            st = load_bot_state()
            st["global_paused"] = False
            st["paused_coins"] = {}
            save_bot_state(st)
            self.send("▶️ Торговля AI-Trader v2 ВОЗОБНОВЛЕНА (все 29 монет активны).", reply_markup=self.main_menu_keyboard(), chat_id=cid)
        elif data == "v2:toggle_skips":
            st = load_bot_state()
            st["telegram_send_skips"] = not bool(st.get("telegram_send_skips", False))
            save_bot_state(st)
            lbl = "ВКЛЮЧЕНА 🔔" if st["telegram_send_skips"] else "ВЫКЛЮЧЕНА 🔕"
            self.send(f"Отправка отклонённых AI сделок (SKIP) в чат: {lbl}", reply_markup=self.main_menu_keyboard(), chat_id=cid)

    def _try_handle_coin_pause(self, low: str, cid: str) -> bool:
        if "пауз" not in low and not low.startswith("/pause") and not low.startswith("/resume"):
            return False
        known = {s.replace("USDT", "").lower(): s for s in self.config["symbols"]}
        found = []
        for w in re.findall(r"[a-z0-9]+", low):
            base = w[:-4] if w.endswith("usdt") and len(w) > 4 else w
            if base in known and known[base] not in found:
                found.append(known[base])

        unpause = any(k in low for k in ("сним", "убер", "возобнов", "старт", "/resume", "отмен"))
        st = load_bot_state()
        if unpause:
            if found:
                for sym in found:
                    st["paused_coins"].pop(sym, None)
                save_bot_state(st)
                self.send(f"▶️ Пауза снята с: {', '.join(found)}.", reply_markup=self.main_menu_keyboard(), chat_id=cid)
            else:
                st["global_paused"] = False
                st["paused_coins"] = {}
                save_bot_state(st)
                self.send("▶️ Глобальная пауза и паузы по всем 29 монетам сняты.", reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return True

        if found:
            hrs = 12.0
            m = HOURS_RE.search(low)
            if m:
                val = float(m.group(1).replace(",", "."))
                hrs = val * (24.0 if m.group(2).lower().startswith(("д", "с", "d")) else 1.0)
                hrs = max(0.5, min(168.0, hrs))
            exp = (datetime.now(timezone.utc) + timedelta(hours=hrs)).strftime("%Y-%m-%dT%H:%M:%SZ")
            for sym in found:
                st["paused_coins"][sym] = exp
            save_bot_state(st)
            self.send(f"⏸ {', '.join(found)} на паузе на {hrs:g}ч (до {exp}).", reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return True

        st["global_paused"] = True
        save_bot_state(st)
        self.send("⏸ Все новые входы AI-Trader v2 поставлены на паузу.", reply_markup=self.main_menu_keyboard(), chat_id=cid)
        return True

    def handle_text(self, text: str, cid: str) -> None:
        low = text.strip().lower()
        if low in ("/start", "/menu", "/help", "меню", "помощь"):
            self.send(self.build_welcome_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if low in ("/positions", "позиции", "баланс", "статус") or any(w in low for w in ("позици", "баланс")):
            self.send(self.build_positions_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if low in ("/ai", "вердикты", "ai") or "вердикт" in low:
            self.send(self.build_ai_decisions_text(8), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if low in ("/week",) or "недел" in low or "7 д" in low:
            self.send(self.build_period_report_text(7), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if low in ("/report",) or any(w in low for w in ("отчет", "отчёт", "сутк")):
            self.send(self.build_period_report_text(1), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if low in ("/scan",) or "скан" in low:
            threading.Thread(target=self._run_manual_scan, args=(cid,), daemon=True).start()
            return
        if low in ("/market",) or any(w in low for w in ("рынок", "макро", "биткоин", "btc")):
            self.send(self.build_market_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if low in ("/schools",) or "школ" in low:
            self.send(self.build_schools_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if low in ("/budget",) or any(w in low for w in ("бюджет", "расход", "токен")):
            self.send(self.build_budget_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)
            return
        if "скип" in low:
            st = load_bot_state()
            st["telegram_send_skips"] = not ("выкл" in low or "off" in low)
            save_bot_state(st)
            self.send(
                f"Уведомления о SKIP AI: {'ВКЛ 🔔' if st['telegram_send_skips'] else 'ВЫКЛ 🔕'}",
                reply_markup=self.main_menu_keyboard(),
                chat_id=cid,
            )
            return
        if self._try_handle_coin_pause(low, cid):
            return

        # Free-form question answered by Claude with live AI-Trader v2 portfolio context
        ctx = {
            "portfolio": self.executor.get_portfolio_state(),
            "report_24h": self.journal.get_period_report(1),
            "recent_ai_decisions": self.journal.get_recent_ai_decisions(5),
        }
        prompt = (
            f"Пользователь AI-Trader v2 написал в Telegram: {text}\n"
            f"Ответь кратко (3-5 предложений) на русском, обычным текстом (НЕ JSON), опираясь на данные портфеля:\n"
            f"{json.dumps(ctx, ensure_ascii=False)}"
        )
        chat_sys = (
            "Ты — встроенный ИИ-ассистент автономного криптобота AI-Trader v2 "
            "(29 монет Bybit, 12 школ ТА, 15m + 1H/4H/BTC, плечо 10x, маржа 10%, макс. 6 позиций / 4 в одну сторону). "
            "Отвечай кратко, по делу и на русском обычным текстом."
        )
        ans, _ = self.judge._call_llm(prompt, max_tokens=600, system_prompt=chat_sys)
        if ans:
            self.send(ans, reply_markup=self.main_menu_keyboard(), chat_id=cid)
        else:
            self.send(self.build_welcome_text(), reply_markup=self.main_menu_keyboard(), chat_id=cid)

    # ------------------------------------------------------------------
    # Background Polling Loop
    # ------------------------------------------------------------------
    def _poll_once(self) -> bool:
        if not self.is_ready:
            return False
        backlog = not self.backlog_drained
        try:
            res = self._api("getUpdates", {"offset": self.offset, "timeout": 0 if backlog else 10}, timeout=15)
        except Exception as exc:
            logger.warning("Telegram poll error: %s", redact_secrets(exc))
            return False
        if not res.get("ok"):
            return False
        updates = res.get("result", [])
        if backlog and not updates:
            self.backlog_drained = True
        for upd in updates:
            self.offset = max(self.offset, int(upd["update_id"]) + 1)
            try:
                if "callback_query" in upd:
                    if not backlog:
                        self.handle_callback(upd["callback_query"])
                elif "message" in upd:
                    m = upd["message"]
                    if backlog and int(m.get("date") or 0) < self.started_at - self.STALE_UPDATE_SEC:
                        continue
                    chat = m.get("chat") or {}
                    user = m.get("from") or {}
                    txt = str(m.get("text", "")).strip()
                    if txt and self.is_authorized(chat, user):
                        self.handle_text(txt, str(chat.get("id", "")))
            except Exception as exc:
                logger.warning("Telegram update handler error: %s", redact_secrets(exc))
        return True

    def start_background(self) -> None:
        if not self.is_ready:
            logger.info("Telegram UI not configured — background polling skipped.")
            return
        self.register_slash_commands()

        def _loop() -> None:
            fails = 0
            while not self._stop_event.is_set():
                if self._poll_once():
                    fails = 0
                else:
                    fails += 1
                    self._stop_event.wait(min(30, 3 * fails))

        self._thread = threading.Thread(target=_loop, name="ai-trader-tg-ui", daemon=True)
        self._thread.start()
        logger.info("AI-Trader v2 Telegram Control Center started (interactive menu + slash commands active).")

    def stop(self) -> None:
        self._stop_event.set()
