"""
Telegram Bot Interface strictly designed for VIKA_N2_N7 (10 coins, 4H STRONG div, N2, N7).
Zero legacy menus, zero old coins, zero deprecated filters.

ReplyKeyboard Buttons:
  1. 📊 Статус и Баланс
  2. 📈 Открытые позиции
  3. 🪙 Портфель (10 монет)
  4. 📜 История сделок
  5. ⏸ Пауза / ▶️ Старт
  6. 🛑 Закрыть всё (Panic Close)  [with inline confirmation]
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.vika_engine import CoinEvaluation, KEY_SHORT_NAMES

logger = logging.getLogger("vika.telegram")

BTN_STATUS = "📊 Статус и Баланс"
BTN_POSITIONS = "📈 Открытые позиции"
BTN_PORTFOLIO = "🪙 Портфель (10 монет)"
BTN_HISTORY = "📜 История сделок"
BTN_TOGGLE_PAUSE = "⏸ Пауза / ▶️ Старт"
BTN_PANIC = "🛑 Закрыть всё (Panic Close)"

MAIN_KEYBOARD = {
    "keyboard": [
        [{"text": BTN_STATUS}, {"text": BTN_POSITIONS}],
        [{"text": BTN_PORTFOLIO}, {"text": BTN_HISTORY}],
        [{"text": BTN_TOGGLE_PAUSE}, {"text": BTN_PANIC}],
    ],
    "resize_keyboard": True,
    "is_persistent": True,
}


class VikaTelegramUI:
    def __init__(
        self,
        cfg: Dict[str, Any],
        executor: VikaExecutor,
        journal: VikaJournal,
        get_latest_evals: Callable[[], Dict[str, CoinEvaluation]],
    ) -> None:
        self.cfg = cfg
        self.executor = executor
        self.journal = journal
        self.get_latest_evals = get_latest_evals

        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.enabled = bool(self.token and self.chat_id and "your_telegram" not in self.token.lower())
        self.api_base = f"https://api.telegram.org/bot{self.token}" if self.enabled else ""

        self._session = requests.Session()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset = 0

    def send_message(
        self,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: str = "HTML",
    ) -> bool:
        if not self.enabled:
            logger.info("[TG DISABLED] %s", text.replace("\n", " | "))
            return False
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        payload["reply_markup"] = json.dumps(reply_markup if reply_markup is not None else MAIN_KEYBOARD)
        try:
            r = self._session.post(f"{self.api_base}/sendMessage", json=payload, timeout=15)
            data = r.json()
            if not data.get("ok"):
                logger.warning("Telegram sendMessage failed: %s", data)
                return False
            return True
        except Exception as exc:
            logger.warning("Telegram sendMessage exception: %s", exc)
            return False

    def _answer_callback(self, callback_id: str, text: str = "") -> None:
        if not self.enabled:
            return
        try:
            self._session.post(
                f"{self.api_base}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": text},
                timeout=10,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # View Formatters
    # ------------------------------------------------------------------
    def format_status(self) -> str:
        ds = self.executor.check_daily_stop()
        stats = self.journal.get_total_realized_stats()
        active = self.journal.get_active_positions()
        paused = self.journal.is_paused()
        mode_str = self.executor.mode.upper()
        run_state = "⏸ Пауза" if paused else "▶️ Активен"
        daily_state = "🚨 СТОП -6% (до 00:00 UTC)" if ds["halted"] else "✅ В норме"

        lines = [
            f"📊 <b>СТАТУС И БАЛАНС | ВИКА (N2+N7)</b>",
            f"Режим биржи: <code>Bybit {mode_str} | 10x Cross</code>",
            f"Статус бота: <b>{run_state}</b>",
            f"",
            f"💰 Текущий депозит: <b>${ds['current_equity']:,.2f}</b> (свободно <code>${self.executor.last_available:,.2f}</code>)",
            f"📅 PnL за день (UTC): <b>{ds['daily_pnl_usd']:+,.2f} USDT ({ds['daily_pnl_pct']:+.2f}%)</b>",
            f"🏆 PnL всего (закрытых): <b>{stats['pnl_usd']:+,.2f} USDT</b> (сделок: {stats['trades']}, WR: {stats['wr']:.1f}%, SL: {stats['sl']})",
            f"",
            f"🧩 Слоты: <b>{len(active)}/{self.executor.max_open}</b> (10% × 10x, макс. 80% маржи)",
            f"🛡 Дневной лимит -6%: <b>{daily_state}</b>",
            f"🔎 Фильтры: <code>4H STRONG Div</code> | <code>N2 Анти-кульминация</code> | <code>N7 Funding &lt;1.5bps</code>",
        ]
        if self.executor.stale_warnings:
            lines.append("")
            lines.append("⚠️ <b>Обнаружены внешние позиции/ордера:</b>")
            for w in self.executor.stale_warnings[:5]:
                lines.append(f"  • <code>{w}</code>")
        return "\n".join(lines)

    def format_open_positions(self) -> str:
        active = self.journal.get_active_positions()
        if not active:
            return "📈 <b>ОТКРЫТЫЕ ПОЗИЦИИ (0/8)</b>\nСейчас активных сделок нет."

        ex_map = self.executor.last_exchange_positions
        lines = [f"📈 <b>ОТКРЫТЫЕ ПОЗИЦИИ ({len(active)}/{self.executor.max_open})</b>"]
        for p in active:
            sym = p["symbol"]
            badge = "🟢 LONG" if p["side"] == "LONG" else "🔴 SHORT"
            k_short = KEY_SHORT_NAMES.get(p["key_name"], p["key_name"])
            be_txt = "✅ BE активен" if p["be_active"] else f"⏳ BE при {p['be_trigger_price']}"
            st_txt = "⏳ Лимитный ордер" if p["status"] == "PENDING" else "⚡ В позиции"

            upnl_txt = "—"
            if sym in ex_map:
                ep_ex = ex_map[sym]
                upnl = float(ep_ex.get("unrealisedPnl") or 0.0)
                margin = max(float(p["margin_usd"]), 1.0)
                upnl_pct = upnl / margin * 100.0
                upnl_txt = f"<b>{upnl:+,.2f}$ ({upnl_pct:+.2f}%)</b>"

            lines.append(
                f"\n<b>{sym}</b> {badge} | {st_txt}\n"
                f"  Ключ: <code>{k_short}</code>\n"
                f"  Вход: <code>{p['entry_price']}</code> | PnL: {upnl_txt}\n"
                f"  TP: <code>{p['tp_price']}</code> | SL: <code>{p['sl_price']}</code> | {be_txt}"
            )
        return "\n".join(lines)

    def format_portfolio(self) -> str:
        evals = self.get_latest_evals()
        lines = ["🪙 <b>ПОРТФЕЛЬ ВИКА (N2+N7) — 10 МОНЕТ</b>"]
        for idx, c in enumerate(self.cfg["coins"], 1):
            sym = c["symbol"]
            k_short = c["key"].split("_")[0]
            tp_sl = f"{c['tp_pct']}/{c['sl_pct']}%"
            ev = evals.get(sym)
            if ev is None:
                lines.append(f"{idx}. <b>{sym}</b> | <code>{k_short}</code> ({tp_sl}) | загрузка...")
            else:
                d4 = "🟢" if (not ev.div4h_bear_strong and not ev.div4h_bull_strong) else ("🔴L" if ev.div4h_bear_strong else "🔴S")
                n2 = "🟢" if not ev.n2_blocked else "🔴"
                n7 = "🟢" if (not ev.n7_blocked_long and not ev.n7_blocked_short) else ("🔴L" if ev.n7_blocked_long else "🔴S")
                lines.append(
                    f"{idx}. <b>{sym}</b> | <code>{k_short}</code> ({tp_sl}) | "
                    f"4H:{d4} N2:{n2} FR:{n7} (<code>{ev.funding_bps:+.2f}bps</code>)"
                )
        lines.append("\n<i>Легенда: 🟢 свободно | 🔴 блок (L=Long, S=Short)</i>")
        return "\n".join(lines)

    def format_history(self) -> str:
        closed = self.journal.get_recent_closed(limit=10)
        if not closed:
            return "📜 <b>ИСТОРИЯ СДЕЛОК</b>\nЗакрытых сделок пока нет."
        lines = ["📜 <b>ПОСЛЕДНИЕ 10 ЗАКРЫТЫХ СДЕЛОК</b>"]
        for r in closed:
            pnl = float(r.get("pnl_usd") or 0.0)
            pct = float(r.get("pnl_pct") or 0.0)
            icon = "✅" if pnl >= 0 else "❌"
            dt = datetime.fromtimestamp(int(r["closed_at"]) / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
            side_s = "L" if r["side"] == "LONG" else "S"
            lines.append(
                f"{icon} <code>{dt}</code> <b>{r['symbol']}</b> ({side_s}) [{r['exit_reason']}] "
                f"<b>{pnl:+,.2f}$ ({pct:+.1f}%)</b>"
            )
        return "\n".join(lines)

    def format_startup_welcome(self, lev_status: Dict[str, str]) -> str:
        ds = self.executor.check_daily_stop()
        ok_cnt = sum(1 for v in lev_status.values() if v.startswith("OK") or v.startswith("DRY_RUN"))
        header = (
            f"🤖 <b>ЗАПУСК: ВИКА (N2+N7)</b>\n"
            f"Биржа: <code>Bybit {self.executor.mode.upper()} | 10x Cross ({ok_cnt}/10 монет)</code>\n"
            f"Баланс: <b>${ds['current_equity']:,.2f}</b> | Риск: <code>10% × 10x | слоты 8/8 | дневной стоп -6%</code>\n"
            f"Фильтры: <code>4H STRONG Div + N2 Анти-кульминация + N7 Funding</code>\n"
        )
        if self.executor.stale_warnings:
            header += "\n⚠️ <b>ВНИМАНИЕ — найдены старые позиции/ордера (не закрыты):</b>\n"
            for w in self.executor.stale_warnings:
                header += f"  • <code>{w}</code>\n"
        return header + "\n" + self.format_portfolio()

    # ------------------------------------------------------------------
    # Polling Loop
    # ------------------------------------------------------------------
    def start_polling(self) -> None:
        if not self.enabled:
            return
        # Flush old queued updates on startup so we don't replay stale panic clicks
        try:
            r = self._session.get(f"{self.api_base}/getUpdates", params={"timeout": 1, "offset": -1}, timeout=10).json()
            if r.get("ok") and r.get("result"):
                self._offset = int(r["result"][-1]["update_id"]) + 1
        except Exception:
            pass
        self._thread = threading.Thread(target=self._poll_loop, name="vika-tg-poll", daemon=True)
        self._thread.start()
        logger.info("Telegram polling started")

    def stop(self) -> None:
        self._stop_evt.set()

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                r = self._session.get(
                    f"{self.api_base}/getUpdates",
                    params={"timeout": 20, "offset": self._offset},
                    timeout=30,
                )
                data = r.json()
                if not data.get("ok"):
                    time.sleep(3)
                    continue
                for upd in data.get("result", []):
                    self._offset = int(upd["update_id"]) + 1
                    self._handle_update(upd)
            except Exception as exc:
                logger.debug("Telegram poll retry: %s", exc)
                time.sleep(3)

    def _handle_update(self, upd: Dict[str, Any]) -> None:
        if "callback_query" in upd:
            cq = upd["callback_query"]
            cid = str(cq.get("message", {}).get("chat", {}).get("id", ""))
            if cid != self.chat_id:
                return
            cdata = cq.get("data", "")
            self._answer_callback(cq["id"])
            if cdata == "panic_confirm":
                res = self.executor.panic_close_all()
                self.send_message(res)
            elif cdata == "panic_cancel":
                self.send_message("❌ Panic Close отменён. Бот продолжает работу в штатном режиме.")
            return

        msg = upd.get("message")
        if not msg:
            return
        cid = str(msg.get("chat", {}).get("id", ""))
        if cid != self.chat_id:
            return
        text = (msg.get("text") or "").strip()

        if text in ("/start", "/status", BTN_STATUS):
            self.send_message(self.format_status())
        elif text in ("/positions", BTN_POSITIONS):
            self.send_message(self.format_open_positions())
        elif text in ("/portfolio", BTN_PORTFOLIO):
            self.send_message(self.format_portfolio())
        elif text in ("/history", BTN_HISTORY):
            self.send_message(self.format_history())
        elif text in ("/pause", BTN_TOGGLE_PAUSE):
            new_state = not self.journal.is_paused()
            self.journal.set_paused(new_state)
            if new_state:
                self.send_message(
                    "⏸ <b>БОТ ПОСТАВЛЕН НА ПАУЗУ</b>\n"
                    "Новые сделки открываться не будут. Защитные SL/TP на открытых позициях сохранены."
                )
            else:
                self.send_message("▶️ <b>БОТ ВОЗОБНОВИЛ РАБОТУ</b>\nПоиск новых входов «ВИКА (N2+N7)» активен.")
        elif text in ("/panic", BTN_PANIC):
            confirm_kb = {
                "inline_keyboard": [[
                    {"text": "✅ Да, закрыть всё по рынку", "callback_data": "panic_confirm"},
                    {"text": "❌ Отмена", "callback_data": "panic_cancel"},
                ]]
            }
            self.send_message(
                "🛑 <b>ПОДТВЕРЖДЕНИЕ PANIC CLOSE</b>\n"
                "Закрыть ВСЕ открытые позиции по рынку, отменить активные ордера и поставить бота на паузу?",
                reply_markup=confirm_kb,
            )
