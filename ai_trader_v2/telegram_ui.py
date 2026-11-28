"""
Telegram Control & Live Monitoring Interface rebuilt exclusively for L2_news
(15x Isolated, 5x12%, Cap $5,000 / $2,500 storm, All-Thick-18: 18L + 18S,
2-Step DCA 35/65, BE +0.30% with DCA cancel, Hard-SL-only AC 6h & Probation,
Retail L/S z > +1.0 LONG / z < -1.0 SHORT, US Macro Blackout [T-60m..T+45m]).

ReplyKeyboard Buttons:
  1. 📊 Статус L2_news          | 📈 Позиции и DCA
  2. 🌐 Макро и Паузы            | 🪙 Портфель (18L+18S)
  3. 📜 История сделок           | ⏸ Пауза / ▶️ Старт
  4. 🛑 Panic Close (inline confirmation)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import requests

from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime
from ai_trader_v2.macro_calendar import MacroCalendarStatus
from ai_trader_v2.vika_engine import CoinEvaluation, KEY_SHORT_NAMES
from ai_trader_v2_long.long_engine import LONG_KEY_SHORT_NAMES, LongCoinEvaluation

logger = logging.getLogger("vika.telegram")

BTN_STATUS = "📊 Статус Vika Hybrid"
BTN_POSITIONS = "📈 Позиции и DCA"
BTN_MACRO = "🌐 Макро и Паузы"
BTN_PORTFOLIO_PREFIX = "🪙 Портфель"
BTN_HISTORY = "📜 История сделок"
BTN_TOGGLE_PAUSE = "⏸ Пауза / ▶️ Старт"
BTN_PANIC = "🛑 Panic Close"


def _fmt_until(ts_ms: int, now_ms: int) -> str:
    if ts_ms <= now_ms:
        return "🟢 нет"
    rem_min = max(1, int((ts_ms - now_ms) // 60_000))
    hrs, mins = divmod(rem_min, 60)
    dt_s = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%d.%m %H:%M UTC")
    return f"🧊 до {dt_s} ({hrs}ч {mins:02d}м)"


def _dca_status_badge(pos: Dict[str, Any]) -> str:
    st = str(pos.get("dca_status") or "NONE")
    px = float(pos.get("dca_price") or 0.0)
    qty = float(pos.get("dca_qty") or 0.0)
    if st == "PENDING":
        return f"⏳ 2-я нога Limit @ <code>{px}</code> (qty <code>{qty}</code>)"
    if st == "FILLED":
        return "✅ 2-я нога DCA залита (полный слот)"
    if st == "PARTIAL_FILLED":
        return "✅ 2-я нога DCA частично залита"
    if st == "CANCELLED_BE":
        return "🧹 2-я нога снята при BE +0.30%"
    if st.startswith("CANCELLED"):
        return f"⚪ 2-я нога отменена ({st})"
    return "⚪ без DCA"


class VikaTelegramUI:
    def __init__(
        self,
        cfg: Dict[str, Any],
        executor: VikaExecutor,
        journal: VikaJournal,
        get_latest_evals: Callable[[], Dict[str, CoinEvaluation]],
        get_latest_long_evals: Optional[Callable[[], Dict[str, LongCoinEvaluation]]] = None,
        get_btc_regime: Optional[Callable[[], Optional[BtcL2Regime]]] = None,
        get_macro_status: Optional[Callable[[], MacroCalendarStatus]] = None,
    ) -> None:
        self.cfg = cfg
        self.executor = executor
        self.journal = journal
        self.get_latest_evals = get_latest_evals
        self.get_latest_long_evals = get_latest_long_evals or (lambda: {})
        self.get_btc_regime = get_btc_regime or (lambda: executor.last_btc_regime)
        self.get_macro_status = get_macro_status or (
            lambda: executor.macro_guard.evaluate() if executor.macro_guard else None
        )

        self.n_long = len(cfg.get("long_coins", []))
        self.n_short = len(cfg.get("short_coins", cfg.get("coins", [])))
        self.btn_portfolio = f"{BTN_PORTFOLIO_PREFIX} ({self.n_long}L+{self.n_short}S)"

        self.main_keyboard: Dict[str, Any] = {
            "keyboard": [
                [{"text": BTN_STATUS}, {"text": BTN_POSITIONS}],
                [{"text": BTN_MACRO}, {"text": self.btn_portfolio}],
                [{"text": BTN_HISTORY}, {"text": BTN_TOGGLE_PAUSE}],
                [{"text": BTN_PANIC}],
            ],
            "resize_keyboard": True,
            "is_persistent": True,
        }
        self.bot_commands: List[Dict[str, str]] = [
            {"command": "status", "description": "Vika Hybrid — баланс, 15x, 5×12%, резерв <=85%, BTC и макро"},
            {"command": "positions", "description": "Открытые позиции, статус 2-й ноги DCA и BE +0.30%"},
            {"command": "macro", "description": "Макроокно FOMC/CPI/NFP, паузы AC 6ч и Probation"},
            {"command": "portfolio", "description": f"Пул All-Thick-18 ({self.n_long}L+{self.n_short}S) и Retail L/S"},
            {"command": "history", "description": "Последние 10 сделок (TP / BE+0.30% / Hard SL)"},
            {"command": "pause", "description": "Пауза / Возобновление торговли"},
            {"command": "panic", "description": "Panic Close — закрыть всё и отменить DCA/TP"},
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
        self._send_lock = threading.Lock()
        self._poll_session = requests.Session()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset = 0

    def is_authorized(self, chat: Dict[str, Any], user: Dict[str, Any]) -> bool:
        cid = str(chat.get("id", ""))
        uid = str(user.get("id", ""))
        if self.chat_id and cid == self.chat_id:
            return True
        if self.allowed_users and (uid in self.allowed_users or cid in self.allowed_users):
            return True
        return False

    def register_commands(self) -> bool:
        if not self.enabled:
            return False
        try:
            with self._send_lock:
                r = self._send_session.post(
                    f"{self.api_base}/setMyCommands",
                    json={"commands": self.bot_commands},
                    timeout=15,
                )
            data = r.json()
            ok = bool(data.get("ok"))
            if not ok:
                logger.warning("Telegram setMyCommands failed: %s", data)
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
            "text": text[:4090],
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
            "reply_markup": reply_markup if reply_markup is not None else self.main_keyboard,
        }
        try:
            with self._send_lock:
                r = self._send_session.post(f"{self.api_base}/sendMessage", json=payload, timeout=15)
            data = r.json()
            if not data.get("ok"):
                logger.warning("Telegram sendMessage failed (chat=%s): %s", target_chat, data)
                if parse_mode == "HTML" and r.status_code == 400:
                    # Fallback to plain text if any dynamic string had unexpected HTML characters
                    plain_payload = dict(payload)
                    plain_payload.pop("parse_mode", None)
                    with self._send_lock:
                        r2 = self._send_session.post(f"{self.api_base}/sendMessage", json=plain_payload, timeout=15)
                    if r2.json().get("ok"):
                        return True
                return False
            return True
        except Exception as exc:
            logger.warning("Telegram sendMessage exception (chat=%s): %s", target_chat, exc)
            return False

    def _answer_callback(self, callback_id: str, text: str = "") -> None:
        if not self.enabled:
            return
        try:
            with self._send_lock:
                self._send_session.post(
                    f"{self.api_base}/answerCallbackQuery",
                    json={"callback_query_id": callback_id, "text": text},
                    timeout=10,
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # L2_news View Formatters
    # ------------------------------------------------------------------
    def format_status(self) -> str:
        try:
            ds = self.executor.check_daily_stop()
        except Exception as exc:
            logger.warning("format_status balance refresh fallback: %s", exc)
            eq = self.executor.last_equity
            ds = self.journal.ensure_daily_state(eq)
            pnl = eq - ds["start_equity"]
            ds["current_equity"] = eq
            ds["daily_pnl_usd"] = round(pnl, 2)
            ds["daily_pnl_pct"] = round((pnl / ds["start_equity"] * 100.0) if ds["start_equity"] > 0 else 0.0, 2)

        now_ms = int(time.time() * 1000)
        stats = self.journal.get_total_realized_stats()
        snap = self.executor.get_margin_reserve_snapshot()
        paused = self.journal.is_paused()
        btc = self.get_btc_regime()
        macro_st = self.get_macro_status()

        run_state = "⏸ ПАУЗА" if paused else ("🚨 СТОП -6% UTC" if ds["halted"] else "▶️ В РАБОТЕ")
        btc_line = btc.mode_label if btc else "⏳ BTC 4H+1D сканируется..."
        macro_line = macro_st.short_label if macro_st else "⚪ календарь не подключён"

        ac_l = _fmt_until(self.journal.get_ac_pause_until_ms("LONG"), now_ms)
        ac_s = _fmt_until(self.journal.get_ac_pause_until_ms("SHORT"), now_ms)
        probs = self.journal.get_all_active_probations(now_ms)
        prob_txt = ", ".join(s.replace("USDT", "") for s, _ in probs) if probs else "нет"

        cap_tag = f"${snap['effective_cap_usd']:,.0f}" + (" ⚡ШТОРМ" if snap["in_storm"] else " (шторм $2,500)")
        dca_split = "30/70" if snap["in_storm"] else "35/65"
        res_icon = "🔴" if snap["reserve_pct"] >= 78.0 else ("🟡" if snap["reserve_pct"] >= 60.0 else "🟢")

        lines = [
            f"📊 <b>СТАТУС Vika Hybrid | Bybit {self.executor.mode.upper()}</b>",
            f"Состояние: <b>{run_state}</b> | Плечо: <code>{self.executor.leverage}x {self.executor.margin_mode}</code>",
            f"Режим BTC: {btc_line}",
            f"Макроокно [-60m..+45m]: {macro_line}",
            "",
            f"💰 Баланс: <b>${ds['current_equity']:,.2f}</b> (uPnL: <code>{snap['unr_open']:+,.2f}$</code>)",
            f"📅 День (UTC): <b>{ds['daily_pnl_usd']:+,.2f}$ ({ds['daily_pnl_pct']:+.2f}%)</b> | Лимит: <code>-{self.executor.daily_loss_limit_pct:.0f}%</code>",
            f"🏆 Итого: <b>{stats['pnl_usd']:+,.2f}$</b> | Сделок: <code>{stats['trades']}</code> (WR <code>{stats['wr']:.1f}%</code> | HardSL <code>{stats['sl']}</code> | BE <code>{stats['be']}</code>)",
            "",
            f"🧩 Слоты <b>5×12%</b>: 🟢L <b>{snap['open_long']}/{self.executor.max_same_dir}</b> | 🔴S <b>{snap['open_short']}/{self.executor.max_same_dir}</b> | Всего <b>{snap['open_total']}/{self.executor.max_open}</b>",
            f"🛡 AntiSalvo: <code>≤{self.executor.max_entries_per_bar_l}L / ≤{self.executor.max_entries_per_bar_s}S за бар, ≤{self.executor.max_same_key}/ключ</code>",
            f"💼 Кап слота: <code>{cap_tag}</code> | {res_icon} Резерв Bybit: <b>{snap['reserve_pct']:.1f}% / {snap['limit_pct']:.0f}%</b> (<code>${snap['reserved_usd']:,.1f}</code>)",
            f"🪜 DCA ({dca_split}): <code>{snap['dca_pending']}</code> в стакане | <code>{snap['dca_filled']}</code> залито | BE <code>+1.0% → +0.30%</code> + отмена DCA",
            "",
            f"🧊 Паузы (только Hard SL):",
            f"  • AC {self.executor.ac_pause_hours}ч LONG: {ac_l}",
            f"  • AC {self.executor.ac_pause_hours}ч SHORT: {ac_s}",
            f"  • Probation ({self.executor.probation_pause_days}д): <code>{prob_txt}</code>",
        ]
        if self.executor.stale_warnings:
            lines.append("\n⚠️ <b>Внешние позиции/ордера на бирже:</b>")
            for w in self.executor.stale_warnings[:4]:
                lines.append(f"  • <code>{w}</code>")
        return "\n".join(lines)

    def format_open_positions(self) -> str:
        active = self.journal.get_active_positions()
        snap = self.executor.get_margin_reserve_snapshot()
        header = (
            f"📈 <b>ПОЗИЦИИ И DCA | Vika Hybrid</b>\n"
            f"🟢 LONG: <b>{snap['open_long']}/{self.executor.max_same_dir}</b> | 🔴 SHORT: <b>{snap['open_short']}/{self.executor.max_same_dir}</b> | "
            f"Резерв: <b>{snap['reserve_pct']:.1f}% / {snap['limit_pct']:.0f}%</b>"
        )
        if not active:
            return header + "\n\nСейчас открытых позиций нет."

        ex_map = self.executor.last_exchange_positions
        lines = [header]
        for p in active:
            sym = p["symbol"]
            badge = "🟢 LONG" if p["side"] == "LONG" else "🔴 SHORT"
            k_short = KEY_SHORT_NAMES.get(p["key_name"], LONG_KEY_SHORT_NAMES.get(p["key_name"], p["key_name"]))
            be_txt = "🛡 <b>BE +0.30% активен</b>" if p["be_active"] else f"⏳ BE при <code>{p['be_trigger_price']}</code>"
            storm_tag = " ⚡ШТОРМ" if p.get("in_storm") else ""

            upnl_txt = "—"
            if sym in ex_map:
                upnl = float(ex_map[sym].get("unrealisedPnl") or 0.0)
                margin = max(float(p["margin_usd"]), 1.0)
                upnl_txt = f"<b>{upnl:+,.2f}$ ({upnl / margin * 100.0:+.1f}%)</b>"

            res_m = float(p.get("reserved_margin_usd") or p["margin_usd"])
            lines.append(
                f"\n<b>{sym}</b> {badge} <code>{p['leverage']}x</code>{storm_tag} | <code>{k_short}</code>\n"
                f"  Средняя: <code>{p['entry_price']}</code> | PnL: {upnl_txt}\n"
                f"  TP: <code>{p['tp_price']}</code> | SL: <code>{p['sl_price']}</code> | {be_txt}\n"
                f"  {_dca_status_badge(p)}\n"
                f"  Маржа живая/резерв: <code>${float(p['margin_usd']):,.1f} / ${res_m:,.1f}</code>"
            )
        return "\n".join(lines)

    def format_macro_and_pauses(self) -> str:
        now_ms = int(time.time() * 1000)
        macro_st = self.get_macro_status()
        lines = ["🌐 <b>МАКРОКАЛЕНДАРЬ И ПАУЗЫ | Vika Hybrid</b>"]
        if macro_st is None:
            lines.append("Календарь: <code>отключён</code>")
        else:
            lines.append(f"Статус окна <code>[-{macro_st.pre_minutes}m .. +{macro_st.post_minutes}m]</code>: {macro_st.short_label}")
            hz_warn = " ⚠️ <b>&lt; 3 дн. — обнови!</b>" if macro_st.horizon_warning else " ✅"
            lines.append(f"Горизонт календаря: <code>{macro_st.horizon_days:.1f} дн.</code>{hz_warn}")
            if macro_st.upcoming_events:
                lines.append("\n📅 <b>Ближайшие FOMC / CPI / NFP (UTC):</b>")
                for ev in macro_st.upcoming_events[:6]:
                    dt_s = ev.dt_utc.strftime("%Y-%m-%d %H:%M UTC")
                    lines.append(f"  • <b>{ev.event_type:<4}</b> <code>{dt_s}</code> — {ev.title}")
            else:
                lines.append("\n⚠️ Будущих событий в <code>config/macro_events_utc.json</code> нет!")

        ac_l = _fmt_until(self.journal.get_ac_pause_until_ms("LONG"), now_ms)
        ac_s = _fmt_until(self.journal.get_ac_pause_until_ms("SHORT"), now_ms)
        probs = self.journal.get_all_active_probations(now_ms)
        lines.append(f"\n🧊 <b>AC {self.executor.ac_pause_hours}ч (Anti-Cascade, только Hard SL):</b>")
        lines.append(f"  • 🟢 LONG: {ac_l}")
        lines.append(f"  • 🔴 SHORT: {ac_s}")
        lines.append(f"\n⛔ <b>Coin Probation ({self.executor.probation_sl_count} Hard SL / {self.executor.probation_window_days}д → {self.executor.probation_pause_days}д):</b>")
        if probs:
            for sym, until_ms in probs:
                lines.append(f"  • <b>{sym}</b>: {_fmt_until(until_ms, now_ms)}")
        else:
            lines.append("  • Активных карантинов нет 🟢")
        return "\n".join(lines)

    def format_portfolio(self) -> str:
        l_evals = self.get_latest_long_evals()
        s_evals = self.get_latest_evals()
        btc = self.get_btc_regime()
        now_ms = int(time.time() * 1000)
        prob_syms = {s for s, _ in self.journal.get_all_active_probations(now_ms)}
        ls_max_z = float(self.cfg.get("filters", {}).get("retail_ls_max_abs_z", 1.0))
        allow_long = bool(btc.allow_long) if btc else False
        allow_short = bool(btc.allow_short) if btc else False

        lines = [f"🪙 <b>ПОРТФЕЛЬ All-Thick-18 ({self.n_long}L + {self.n_short}S)</b>"]
        if btc:
            lines.append(btc.mode_label)
        lines.append(
            f"<i>🟢 допущена | ⛔ блок z &gt; +{ls_max_z:.1f} (LONG), z &lt; -{ls_max_z:.1f} (SHORT) / Probation | ⚪ откл. BTC</i>"
        )

        if self.n_long > 0:
            l_rows: List[str] = []
            allowed_l = 0
            for c in self.cfg.get("long_coins", []):
                sym = c["symbol"]
                ks = c["key"].split("_")[0] + ("L" if c["key"].startswith("K") and c["key"].endswith("_L") else "")
                ev = l_evals.get(sym)
                if ev is None:
                    l_rows.append(f"• <code>{sym[:-4]:<9}</code> {ks} | ⏳")
                    continue
                ls_s = f"z={ev.retail_ls_z:+.1f}" if ev.retail_ls_available else "z=n/a"
                in_prob = sym in prob_syms
                ls_blk = ev.retail_ls_available and (ev.retail_ls_z > ls_max_z)
                if not allow_long:
                    ico = "⚪"
                elif in_prob or ls_blk:
                    ico = "⛔"
                else:
                    ico = "🟢"
                    allowed_l += 1
                prob_tag = " [P]" if in_prob else ""
                l_rows.append(f"{ico} <code>{sym[:-4]:<9}</code> {ks:<3} | {ls_s}{prob_tag}")
            l_hdr = (
                f"\n🟢 <b>LONG ({self.n_long} | допущено: {allowed_l}):</b>"
                if allow_long
                else f"\n⚪ <b>LONG ({self.n_long} | откл. BTC):</b>"
            )
            lines.append(l_hdr)
            lines.extend(l_rows)

        if self.n_short > 0:
            s_rows: List[str] = []
            allowed_s = 0
            for c in self.cfg.get("short_coins", self.cfg.get("coins", [])):
                sym = c["symbol"]
                ks = c["key"].split("_")[0]
                ev = s_evals.get(sym)
                if ev is None:
                    s_rows.append(f"• <code>{sym[:-4]:<9}</code> {ks} | ⏳")
                    continue
                ls_s = f"z={ev.retail_ls_z:+.1f}" if ev.retail_ls_available else "z=n/a"
                in_prob = sym in prob_syms
                ls_blk = ev.retail_ls_available and (ev.retail_ls_z < -ls_max_z)
                if not allow_short:
                    ico = "⚪"
                elif in_prob or ls_blk:
                    ico = "⛔"
                else:
                    ico = "🟢"
                    allowed_s += 1
                prob_tag = " [P]" if in_prob else ""
                s_rows.append(f"{ico} <code>{sym[:-4]:<9}</code> {ks:<3} | {ls_s}{prob_tag} | 4H:{ev.slope4h_pct:+.1f}%")
            s_hdr = (
                f"\n🔴 <b>SHORT ({self.n_short} | допущено: {allowed_s}):</b>"
                if allow_short
                else f"\n⚪ <b>SHORT ({self.n_short} | откл. BTC):</b>"
            )
            lines.append(s_hdr)
            lines.extend(s_rows)

        return "\n".join(lines)

    def format_history(self) -> str:
        closed = self.journal.get_recent_closed(limit=10)
        if not closed:
            return "📜 <b>ИСТОРИЯ СДЕЛОК | Vika Hybrid</b>\nЗакрытых сделок пока нет."
        lines = ["📜 <b>ПОСЛЕДНИЕ 10 СДЕЛОК | Vika Hybrid</b>"]
        for r in closed:
            pnl = float(r.get("pnl_usd") or 0.0)
            pct = float(r.get("pnl_pct") or 0.0)
            reason = str(r.get("exit_reason") or "CLOSED")
            icon = "❌" if reason in ("SL", "HARD_SL") else ("🛡" if reason == "BE" else "✅")
            dt = datetime.fromtimestamp(int(r["closed_at"]) / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
            side_s = "L" if r["side"] == "LONG" else "S"
            dca_s = "+DCA" if r.get("dca_status") == "FILLED" else ""
            lines.append(
                f"{icon} <code>{dt}</code> <b>{r['symbol'][:-4]}</b>({side_s}{dca_s}) [{reason}] "
                f"<b>{pnl:+,.2f}$ ({pct:+.1f}%)</b>"
            )
        return "\n".join(lines)

    def format_startup_welcome(self, lev_status: Dict[str, str]) -> str:
        ds = self.executor.check_daily_stop()
        snap = self.executor.get_margin_reserve_snapshot()
        btc = self.get_btc_regime()
        macro_st = self.get_macro_status()
        n_uniq = len(lev_status)
        ok_cnt = sum(1 for v in lev_status.values() if v.startswith("OK") or v.startswith("DRY_RUN"))
        ls_max_z = float(self.cfg.get("filters", {}).get("retail_ls_max_abs_z", 1.0))

        lines = [
            f"🤖 <b>ЗАПУСК: Vika Hybrid (All-Thick-18)</b>",
            f"Биржа: <code>Bybit {self.executor.mode.upper()} | {self.executor.leverage}x {self.executor.margin_mode} ({ok_cnt}/{n_uniq})</code>",
            f"Пул: <b>{self.n_long} LONG + {self.n_short} SHORT</b> | Retail L/S: <code>z &gt; +{ls_max_z:.1f}</code> (LONG), <code>z &lt; -{ls_max_z:.1f}</code> (SHORT)",
            f"Риск: <b>5×12%</b> | Кап <code>${self.executor.max_margin_usd:,.0f}</code> (шторм <code>${self.executor.storm_max_margin_usd:,.0f}</code>) | Резерв <code>≤{self.executor.max_total_margin_pct*100:.0f}%</code>",
            f"DCA: <code>35/65 (-1.5% / SL -4% / TP +1.5%)</code> | Шторм: <code>30/70 (-2% / SL -5% / TP +1.6%)</code>",
            f"BE: <code>+1.0% → +0.30%</code> + авто-снятие DCA | AC 6ч &amp; Probation: <code>только Hard SL</code>",
            f"Баланс: <b>${ds['current_equity']:,.2f}</b> | Резерв: <code>{snap['reserve_pct']:.1f}%</code>",
            f"BTC: {btc.mode_label if btc else '⏳ сканируется'}",
            f"Макро: {macro_st.short_label if macro_st else '⚪ выкл'}",
        ]
        if self.executor.stale_warnings:
            lines.append("\n⚠️ <b>Внешние позиции/ордера (не закрыты):</b>")
            for w in self.executor.stale_warnings[:4]:
                lines.append(f"  • <code>{w}</code>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Polling Loop & Command Router
    # ------------------------------------------------------------------
    def start_polling(self) -> None:
        if not self.enabled:
            return
        try:
            r = self._poll_session.get(f"{self.api_base}/getUpdates", params={"timeout": 1, "offset": -1}, timeout=10).json()
            if r.get("ok") and r.get("result"):
                self._offset = int(r["result"][-1]["update_id"]) + 1
        except Exception as exc:
            logger.warning("Telegram initial offset flush note: %s", exc)
        self._thread = threading.Thread(target=self._poll_loop, name="l2-tg-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def _poll_loop(self) -> None:
        if (
            not self.executor.dry_run
            and getattr(self.executor.client, "has_credentials", False)
            and hasattr(self.executor.client, "ensure_thread_private_client")
        ):
            try:
                self.executor.client.ensure_thread_private_client()
            except Exception as exc:
                logger.warning("Telegram thread private client init warning: %s", exc)
        while not self._stop_evt.is_set():
            try:
                r = self._poll_session.get(
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
                self.send_message(self.executor.panic_close_all(), chat_id=cid)
            elif cdata == "panic_cancel":
                self.send_message("❌ Panic Close отменён. Vika Hybrid продолжает работу.", chat_id=cid)
            else:
                self.send_message("🔄 Интерфейс обновлён до <b>Vika Hybrid</b>.\n\n" + self.format_status(), chat_id=cid)
            return

        msg = upd.get("message")
        if not msg:
            return
        chat = msg.get("chat", {}) or {}
        user = msg.get("from", {}) or {}
        if not self.is_authorized(chat, user):
            return
        cid = str(chat.get("id", ""))
        text = (msg.get("text") or "").strip()
        if not text:
            return
        cmd = text.split()[0].split("@")[0].lower()
        low = text.lower()

        if text == BTN_STATUS or cmd in ("/start", "/status", "/menu") or "статус" in low:
            self.send_message(self.format_status(), chat_id=cid)
        elif text == BTN_POSITIONS or cmd == "/positions" or "позиции" in low:
            self.send_message(self.format_open_positions(), chat_id=cid)
        elif text == BTN_MACRO or cmd == "/macro" or "макро" in low or "паузы" in low:
            self.send_message(self.format_macro_and_pauses(), chat_id=cid)
        elif text == self.btn_portfolio or cmd == "/portfolio" or "портфель" in low:
            self.send_message(self.format_portfolio(), chat_id=cid)
        elif text == BTN_HISTORY or cmd == "/history" or "история" in low:
            self.send_message(self.format_history(), chat_id=cid)
        elif text == BTN_TOGGLE_PAUSE or cmd in ("/pause", "/resume") or ("пауза" in low and "старт" in low):
            new_state = not self.journal.is_paused()
            self.journal.set_paused(new_state)
            if new_state:
                self.send_message(
                    "⏸ <b>Vika Hybrid ПОСТАВЛЕН НА ПАУЗУ</b>\nНовые входы отключены. Защитные SL/TP и BE на открытых позициях работают.",
                    chat_id=cid,
                )
            else:
                self.send_message("▶️ <b>Vika Hybrid ВОЗОБНОВИЛ РАБОТУ</b>\nПоиск входов активен.", chat_id=cid)
        elif text == BTN_PANIC or cmd == "/panic" or "panic" in low or "закрыть всё" in low:
            confirm_kb = {
                "inline_keyboard": [[
                    {"text": "✅ Да, закрыть всё и отменить DCA/TP", "callback_data": "panic_confirm"},
                    {"text": "❌ Отмена", "callback_data": "panic_cancel"},
                ]]
            }
            self.send_message(
                "🛑 <b>ПОДТВЕРЖДЕНИЕ PANIC CLOSE | Vika Hybrid</b>\n"
                "Закрыть ВСЕ позиции по рынку, отменить все DCA/TP ордера и поставить бота на паузу?",
                reply_markup=confirm_kb,
                chat_id=cid,
            )
