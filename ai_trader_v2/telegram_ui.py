"""
Telegram Bot Interface strictly designed for Vika N2N7 (dynamic N-coin portfolio, Variant B).
Zero legacy menus, zero hardcoded coin counts, zero deprecated filters.

ReplyKeyboard Buttons:
  1. 📊 Статус и Баланс
  2. 📈 Открытые позиции
  3. 🪙 Портфель (18 монет)  [dynamic from config]
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
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.vika_engine import CoinEvaluation, KEY_SHORT_NAMES

logger = logging.getLogger("vika.telegram")

BTN_STATUS = "📊 Статус и Баланс"
BTN_POSITIONS = "📈 Открытые позиции"
BTN_PORTFOLIO_PREFIX = "🪙 Портфель"
BTN_HISTORY = "📜 История сделок"
BTN_TOGGLE_PAUSE = "⏸ Пауза / ▶️ Старт"
BTN_PANIC = "🛑 Закрыть всё (Panic Close)"

DIR_LABELS: Dict[str, str] = {
    "only_short": "🔴 ONLY SHORT",
    "only_long": "🟢 ONLY LONG",
    "long_short": "🟢🔴 LONG + SHORT",
    "paused": "⏸ PAUSED",
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
        self.n_coins = len(cfg.get("coins", []))
        self.btn_portfolio = f"{BTN_PORTFOLIO_PREFIX} ({self.n_coins} монет)"
        self.main_keyboard: Dict[str, Any] = {
            "keyboard": [
                [{"text": BTN_STATUS}, {"text": BTN_POSITIONS}],
                [{"text": self.btn_portfolio}, {"text": BTN_HISTORY}],
                [{"text": BTN_TOGGLE_PAUSE}, {"text": BTN_PANIC}],
            ],
            "resize_keyboard": True,
            "is_persistent": True,
        }
        self.bot_commands: List[Dict[str, str]] = [
            {"command": "status", "description": f"Vika N2N7 ({self.n_coins} монет) — статус, баланс и лимиты"},
            {"command": "positions", "description": "Открытые позиции, TP/SL и статус BE"},
            {"command": "portfolio", "description": f"Портфель {self.n_coins} монет и фильтры (4H/N2/N7/T4/K7)"},
            {"command": "history", "description": "Последние 10 закрытых сделок"},
            {"command": "pause", "description": "Пауза / Возобновление торговли"},
            {"command": "panic", "description": "Panic Close — закрыть всё по рынку"},
        ]

        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        raw_users = os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
        self.allowed_users = {u.strip() for u in raw_users.split(",") if u.strip()}
        self.enabled = bool(
            self.token
            and "your_telegram" not in self.token.lower()
            and (self.chat_id or self.allowed_users)
        )
        self.api_base = f"https://api.telegram.org/bot{self.token}" if self.enabled else ""

        self._send_session = requests.Session()
        self._poll_session = requests.Session()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset = 0

    def _dir_label(self) -> str:
        raw = getattr(self.executor, "allowed_directions", str(self.cfg.get("allowed_directions", "long_short"))).lower()
        return DIR_LABELS.get(raw, "🟢🔴 LONG + SHORT")

    def _keys_summary(self) -> str:
        cnt = Counter(c["key"].split("_")[0] for c in self.cfg.get("coins", []))
        order = ["K1", "K7", "K6", "K8", "K3"]
        parts = [f"{k}:{cnt[k]}" for k in order if cnt.get(k)]
        for k, v in sorted(cnt.items()):
            if k not in order:
                parts.append(f"{k}:{v}")
        return " ".join(parts)

    def is_authorized(self, chat: Dict[str, Any], user: Dict[str, Any]) -> bool:
        cid = str(chat.get("id", ""))
        uid = str(user.get("id", ""))
        if self.chat_id and cid == self.chat_id:
            return True
        if self.allowed_users and (uid in self.allowed_users or cid in self.allowed_users):
            return True
        return False

    def register_commands(self) -> bool:
        """Overwrite Telegram bot slash-command menu via setMyCommands strictly for Vika N2N7."""
        if not self.enabled:
            return False
        try:
            r = self._send_session.post(
                f"{self.api_base}/setMyCommands",
                json={"commands": self.bot_commands},
                timeout=15,
            )
            data = r.json()
            ok = bool(data.get("ok"))
            if not ok:
                logger.warning("Telegram setMyCommands failed: %s", data)
            else:
                logger.info("Telegram setMyCommands updated for Vika N2N7 (%d coins): %d commands", self.n_coins, len(self.bot_commands))
            return ok
        except Exception as exc:
            logger.warning("Telegram setMyCommands exception: %s", exc)
            return False

    def send_message(
        self,
        text: str,
        reply_markup: Optional[Dict[str, Any]] = None,
        parse_mode: str = "HTML",
        chat_id: Optional[str] = None,
    ) -> bool:
        if not self.enabled:
            logger.info("[TG DISABLED] %s", text.replace("\n", " | "))
            return False
        target_chat = str(chat_id or self.chat_id or next(iter(self.allowed_users), "")).strip()
        if not target_chat:
            return False
        payload: Dict[str, Any] = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
            "reply_markup": reply_markup if reply_markup is not None else self.main_keyboard,
        }
        try:
            r = self._send_session.post(f"{self.api_base}/sendMessage", json=payload, timeout=15)
            data = r.json()
            if not data.get("ok"):
                logger.warning("Telegram sendMessage failed (chat=%s): %s", target_chat, data)
                return False
            return True
        except Exception as exc:
            logger.warning("Telegram sendMessage exception (chat=%s): %s", target_chat, exc)
            return False

    def _answer_callback(self, callback_id: str, text: str = "") -> None:
        if not self.enabled:
            return
        try:
            self._send_session.post(
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
        try:
            ds = self.executor.check_daily_stop()
        except Exception as exc:
            logger.warning("format_status balance refresh fallback: %s", exc)
            eq = self.executor.last_equity
            ds = self.journal.ensure_daily_state(eq)
            start_eq = ds["start_equity"]
            pnl = eq - start_eq
            ds["current_equity"] = eq
            ds["daily_pnl_usd"] = round(pnl, 2)
            ds["daily_pnl_pct"] = round((pnl / start_eq * 100.0) if start_eq > 0 else 0.0, 2)
        stats = self.journal.get_total_realized_stats()
        active = self.journal.get_active_positions()
        paused = self.journal.is_paused()
        mode_str = self.executor.mode.upper()
        run_state = "⏸ Пауза" if paused else "▶️ Активен"
        daily_state = "🚨 СТОП -6% (до 00:00 UTC)" if ds["halted"] else "✅ В норме"

        exec_mode_str = (
            "Market IOC + Limit Maker TP (0.02%) + Market SL/BE"
            if getattr(self.executor, "entry_order_type", "Market") == "Market"
            else f"Limit вход (TTL {self.executor.ttl_sec // 60}m) + Market SL/BE"
        )
        dir_str = self._dir_label()
        k7_tag = " | <code>K7 color</code>" if self.cfg.get("filters", {}).get("k7_require_candle_dir", True) else ""
        lines = [
            f"📊 <b>СТАТУС И БАЛАНС | Vika N2N7 ({self.n_coins} монет)</b>",
            f"Режим биржи: <code>Bybit {mode_str} | 10x Cross</code>",
            f"Направление: <b>{dir_str}</b> | Статус: <b>{run_state}</b>",
            f"",
            f"💰 Текущий депозит: <b>${ds['current_equity']:,.2f}</b> (свободно <code>${self.executor.last_available:,.2f}</code>)",
            f"📅 PnL за день (UTC): <b>{ds['daily_pnl_usd']:+,.2f} USDT ({ds['daily_pnl_pct']:+.2f}%)</b>",
            f"🏆 PnL всего (закрытых): <b>{stats['pnl_usd']:+,.2f} USDT</b> (сделок: {stats['trades']}, WR: {stats['wr']:.1f}%, SL: {stats['sl']})",
            f"",
            f"🧩 Слоты: <b>{len(active)}/{self.executor.max_open}</b> (10% × 10x, макс. 80% маржи | R3: макс. {self.executor.max_same_key}/ключ)",
            f"🪙 Монеты ({self.n_coins}): <code>{self._keys_summary()}</code>",
            f"⚡ Ордера: <code>{exec_mode_str}</code>",
            f"🛡 Дневной лимит -6%: <b>{daily_state}</b>",
            f"🔎 Фильтры: <code>4H STRONG Div</code> | <code>N2</code> | <code>N7 FR&lt;1.5bps</code> | <code>T4 4H Slope&gt;0.15%</code>{k7_tag}",
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
            return "📈 <b>ОТКРЫТЫЕ ПОЗИЦИИ (0/8) | Vika N2N7</b>\nСейчас активных сделок нет."

        ex_map = self.executor.last_exchange_positions
        lines = [f"📈 <b>ОТКРЫТЫЕ ПОЗИЦИИ ({len(active)}/{self.executor.max_open}) | Vika N2N7</b>"]
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
        n_coins = self.n_coins
        allowed = getattr(self.executor, "allowed_directions", "long_short")
        lines = [f"🪙 <b>ПОРТФЕЛЬ Vika N2N7 — {n_coins} МОНЕТ ({self._dir_label()})</b>"]
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
                if not ev.t4_blocked_short:
                    t4 = "🟢S" if allowed in ("long_short", "only_short") else "⚪S"
                elif not ev.t4_blocked_long:
                    t4 = "🟢L" if allowed in ("long_short", "only_long") else "⚪L"
                else:
                    t4 = "🔴"
                lines.append(
                    f"{idx}. <b>{sym}</b> | <code>{k_short}</code> ({tp_sl}) | "
                    f"4H:{d4} N2:{n2} FR:{n7} T4:{t4}(<code>{ev.slope4h_pct:+.2f}%</code>)"
                )
        lines.append("\n<i>Легенда: 🟢 свободно | ⚪ выкл. направлением | 🔴 блок | T4: наклон 4H EMA50 &gt;0.15%</i>")
        return "\n".join(lines)

    def format_history(self) -> str:
        closed = self.journal.get_recent_closed(limit=10)
        if not closed:
            return "📜 <b>ИСТОРИЯ СДЕЛОК | Vika N2N7</b>\nЗакрытых сделок пока нет."
        lines = ["📜 <b>ПОСЛЕДНИЕ 10 ЗАКРЫТЫХ СДЕЛОК | Vika N2N7</b>"]
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
        n_coins = len(self.cfg["coins"])
        ok_cnt = sum(1 for v in lev_status.values() if v.startswith("OK") or v.startswith("DRY_RUN"))
        exec_mode_str = (
            "Market IOC + Limit Maker TP (0.02%) + Market SL/BE"
            if getattr(self.executor, "entry_order_type", "Market") == "Market"
            else f"Limit вход (TTL {self.executor.ttl_sec // 60}m)"
        )
        dir_str = self._dir_label()
        k7_tag = " + K7 color" if self.cfg.get("filters", {}).get("k7_require_candle_dir", True) else ""
        header = (
            f"🤖 <b>ЗАПУСК: Vika N2N7 ({n_coins} CLEAN + R3 + T4)</b>\n"
            f"Биржа: <code>Bybit {self.executor.mode.upper()} | 10x Cross ({ok_cnt}/{n_coins} монет)</code>\n"
            f"Направление: <b>{dir_str}</b> | Ключи: <code>{self._keys_summary()}</code>\n"
            f"Баланс: <b>${ds['current_equity']:,.2f}</b> | Риск: <code>10% × 10x | слоты 8/8 (R3: макс.{self.executor.max_same_key}/ключ) | стоп -6%</code>\n"
            f"Ордера: <code>{exec_mode_str}</code>\n"
            f"Фильтры: <code>4H Div + N2 + N7 + T4(4H slope&gt;0.15%){k7_tag}</code>\n"
        )
        if getattr(self.executor, "startup_cancelled_orders", None):
            header += "\n🧹 <b>Отменены зависшие лимитные ордера при старте:</b>\n"
            for item in self.executor.startup_cancelled_orders:
                header += f"  • <code>{item}</code>\n"
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
            r = self._poll_session.get(f"{self.api_base}/getUpdates", params={"timeout": 1, "offset": -1}, timeout=10).json()
            if r.get("ok") and r.get("result"):
                self._offset = int(r["result"][-1]["update_id"]) + 1
        except Exception as exc:
            logger.warning("Telegram initial offset flush note: %s", exc)
        self._thread = threading.Thread(target=self._poll_loop, name="vika-tg-poll", daemon=True)
        self._thread.start()
        logger.info("Telegram polling started (chat_id=%s, allowed_users=%d)", bool(self.chat_id), len(self.allowed_users))

    def stop(self) -> None:
        self._stop_evt.set()

    def _poll_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                r = self._poll_session.get(
                    f"{self.api_base}/getUpdates",
                    params={"timeout": 20, "offset": self._offset},
                    timeout=30,
                )
                data = r.json()
                if not data.get("ok"):
                    logger.warning("Telegram getUpdates non-ok: %s", data.get("description") or data)
                    time.sleep(3)
                    continue
                for upd in data.get("result", []):
                    self._offset = int(upd["update_id"]) + 1
                    try:
                        self._handle_update(upd)
                    except Exception as exc:
                        logger.warning("Telegram _handle_update error: %s", exc)
            except Exception as exc:
                logger.warning("Telegram poll retry: %s", exc)
                time.sleep(3)

    def _handle_update(self, upd: Dict[str, Any]) -> None:
        if "callback_query" in upd:
            cq = upd["callback_query"]
            chat = cq.get("message", {}).get("chat", {}) or {}
            user = cq.get("from", {}) or {}
            if not self.is_authorized(chat, user):
                return
            cid = str(chat.get("id", ""))
            cdata = cq.get("data", "")
            self._answer_callback(cq["id"])
            if cdata == "panic_confirm":
                res = self.executor.panic_close_all()
                self.send_message(res, chat_id=cid)
            elif cdata == "panic_cancel":
                self.send_message("❌ Panic Close отменён. Vika N2N7 продолжает работу в штатном режиме.", chat_id=cid)
            else:
                # Legacy inline button clicked -> send fresh Vika N2N7 status & updated ReplyKeyboard
                self.send_message(
                    f"🔄 Интерфейс обновлён до <b>Vika N2N7</b> ({self.n_coins} монет, {self._dir_label()}).\n\n"
                    + self.format_status(),
                    chat_id=cid,
                )
            return

        msg = upd.get("message")
        if not msg:
            return
        chat = msg.get("chat", {}) or {}
        user = msg.get("from", {}) or {}
        if not self.is_authorized(chat, user):
            logger.info("Ignored unauthorized Telegram message from chat=%s user=%s", chat.get("id"), user.get("id"))
            return
        cid = str(chat.get("id", ""))
        text = (msg.get("text") or "").strip()
        if not text:
            return
        cmd = text.split()[0].split("@")[0].lower()
        low = text.lower()

        if text == BTN_STATUS or cmd in ("/start", "/status", "/menu") or ("статус" in low and "баланс" in low):
            self.send_message(self.format_status(), chat_id=cid)
        elif text == BTN_POSITIONS or cmd == "/positions" or "открытые позиции" in low:
            self.send_message(self.format_open_positions(), chat_id=cid)
        elif text == self.btn_portfolio or cmd == "/portfolio" or "портфель" in low:
            self.send_message(self.format_portfolio(), chat_id=cid)
        elif text == BTN_HISTORY or cmd == "/history" or "история сделок" in low:
            self.send_message(self.format_history(), chat_id=cid)
        elif text == BTN_TOGGLE_PAUSE or cmd in ("/pause", "/resume") or ("пауза" in low and "старт" in low):
            new_state = not self.journal.is_paused()
            self.journal.set_paused(new_state)
            if new_state:
                self.send_message(
                    "⏸ <b>Vika N2N7 ПОСТАВЛЕН НА ПАУЗУ</b>\n"
                    "Новые сделки открываться не будут. Защитные SL/TP на открытых позициях сохранены.",
                    chat_id=cid,
                )
            else:
                self.send_message("▶️ <b>Vika N2N7 ВОЗОБНОВИЛ РАБОТУ</b>\nПоиск новых входов активен.", chat_id=cid)
        elif text == BTN_PANIC or cmd == "/panic" or "panic close" in low or "закрыть всё" in low:
            confirm_kb = {
                "inline_keyboard": [[
                    {"text": "✅ Да, закрыть всё по рынку", "callback_data": "panic_confirm"},
                    {"text": "❌ Отмена", "callback_data": "panic_cancel"},
                ]]
            }
            self.send_message(
                "🛑 <b>ПОДТВЕРЖДЕНИЕ PANIC CLOSE | Vika N2N7</b>\n"
                "Закрыть ВСЕ открытые позиции по рынку, отменить активные ордера и поставить бота на паузу?",
                reply_markup=confirm_kb,
                chat_id=cid,
            )
