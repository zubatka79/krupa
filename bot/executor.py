"""Production Trade Executor (`bot/executor.py`):
- Strictly enforces Isolated Margin + 15x leverage (`ADAUSDT` Variant B) via Bybit v5 API
- Hard US Macro Shield inside the order & position management core:
    * Refuses all new entries during `-30m .. +15m` UTC OR when calendar fail-safe is active
    * Immediately cancels unfilled pending Maker limits during `-30m .. +15m` UTC
    * Force-closes open positions at market at `-30m` before core US news if `uPnL < +0.25%`
- PostOnly Maker Limit entry (Fib 0.50) with atomic server-side SL (2.50%) and 4-bar (60m) TTL
- Places 100% reduce-only PostOnly Limit TP (1.20%) upon entry fill
- 2-Step Smart Break-Even: +0.80% -> SL +0.30% (`BE_ACTIVE`), +0.95% -> SL +0.50% (`BE2_ACTIVE`)
- 100% market close on full 1H EMA20/EMA50 trend flip (`H1_FLIP`)
- Enforces max_daily_stops=3 circuit-breaker and SQLite state persistence (`data/trades.db`)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from bot.bybit_client import BybitClient, round_step_down, round_tick
from bot.news_guard import NewsGuard, minutes_to_next_event, read_news_state
from bot.risk_manager import RiskManager
from bot.scanner import TradeSignal

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso(now: Optional[datetime] = None) -> str:
    return (now or utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_today_str(now: Optional[datetime] = None) -> str:
    return (now or utc_now()).astimezone(timezone.utc).strftime("%Y-%m-%d")


class TradeExecutor:
    """Manages production Maker/TP/SL/2-Step-BE/H1-Flip/Macro-Shield lifecycle & SQLite state."""

    def __init__(
        self,
        client: BybitClient,
        risk_manager: RiskManager,
        config: Dict[str, Any],
        db_path: str | Path = "data/trades.db",
        dry_run: bool = False,
        notifier: Optional[Callable[[str], None]] = None,
        news_guard: Optional[NewsGuard] = None,
        trend_provider: Optional[Callable[[str], str]] = None,
        news_state_path: str | Path = "data/news_state.json",
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.config = config
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run or (not client.has_credentials)
        self.notifier = notifier
        self.news_guard = news_guard
        self.trend_provider = trend_provider
        self.news_state_path = Path(news_state_path)
        self._finalize_attempts: Dict[int, int] = {}
        self._init_db()

    def update_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.risk_manager.update_config(config)

    def _notify(self, text: str) -> None:
        logger.info(text)
        if self.notifier:
            try:
                self.notifier(text)
            except Exception as exc:
                logger.warning("Notifier error: %s", exc)

    def _get_macro_shield(self, now: Optional[datetime] = None, allow_network_refresh: bool = True) -> Dict[str, Any]:
        if not bool(self.config.get("macro_shield_enabled", True)):
            return {
                "entry_allowed": True,
                "cancel_pending_limits": False,
                "protect_open_positions": False,
                "fail_safe_active": False,
                "reason": None,
            }
        if self.news_guard is not None:
            return self.news_guard.evaluate_macro_shield(now=now, allow_network_refresh=allow_network_refresh)
        return read_news_state(self.news_state_path)

    # ------------------------------------------------------------------
    # SQLite Schema & Persistence
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    expires_at_utc TEXT,
                    entry_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp1_price REAL NOT NULL,
                    tp2_price REAL NOT NULL,
                    be_trigger_price REAL DEFAULT 0.0,
                    be_price REAL NOT NULL,
                    be2_trigger_price REAL DEFAULT 0.0,
                    be2_price REAL DEFAULT 0.0,
                    qty REAL NOT NULL,
                    tp1_qty REAL NOT NULL,
                    tp2_qty REAL NOT NULL,
                    remaining_qty REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    margin_usdt REAL NOT NULL,
                    notional_usdt REAL NOT NULL,
                    score REAL NOT NULL,
                    score_details_json TEXT,
                    entry_order_id TEXT,
                    tp1_order_id TEXT,
                    tp2_order_id TEXT,
                    exit_reason TEXT,
                    gross_pnl_usdt REAL DEFAULT 0.0,
                    fees_usdt REAL DEFAULT 0.0,
                    net_pnl_usdt REAL DEFAULT 0.0,
                    net_pnl_pct REAL DEFAULT 0.0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date_utc TEXT PRIMARY KEY,
                    start_equity REAL NOT NULL,
                    realized_pnl_usdt REAL DEFAULT 0.0,
                    trades_count INTEGER DEFAULT 0,
                    sl_count INTEGER DEFAULT 0,
                    is_halted INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS news_pauses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    pause_until_utc TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
                """
            )
            existing_trade_cols = {r["name"] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
            for col_name, col_def in (
                ("expires_at_utc", "TEXT"),
                ("be_trigger_price", "REAL DEFAULT 0.0"),
                ("be2_trigger_price", "REAL DEFAULT 0.0"),
                ("be2_price", "REAL DEFAULT 0.0"),
            ):
                if col_name not in existing_trade_cols:
                    conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}")
            existing_daily_cols = {r["name"] for r in conn.execute("PRAGMA table_info(daily_stats)").fetchall()}
            if "sl_count" not in existing_daily_cols:
                conn.execute("ALTER TABLE daily_stats ADD COLUMN sl_count INTEGER DEFAULT 0")
            conn.commit()

    def log_news_pause(self, layer: str, pause_until_utc: str, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO news_pauses (created_at, pause_until_utc, layer, reason) VALUES (?, ?, ?, ?)",
                (utc_now_iso(), pause_until_utc, layer, reason),
            )
            conn.commit()

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Return all PENDING, OPEN, TP1_HIT, BE_ACTIVE, and BE2_ACTIVE trades."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status IN ('PENDING', 'OPEN', 'TP1_HIT', 'BE_ACTIVE', 'BE2_ACTIVE') ORDER BY id ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trades(self) -> List[Dict[str, Any]]:
        return self.get_active_trades()

    def is_symbol_in_cooldown(self, symbol: str, now: Optional[datetime] = None) -> bool:
        cooldown_min = int(self.config.get("cooldown_bars", 1)) * 15
        with self._connect() as conn:
            row = conn.execute(
                "SELECT exit_time FROM trades WHERE symbol = ? AND status = 'CLOSED' AND exit_time IS NOT NULL "
                "ORDER BY id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if not row or not row["exit_time"]:
            return False
        try:
            ex_dt = datetime.fromisoformat(str(row["exit_time"]).replace("Z", "+00:00"))
            return ((now or utc_now()) - ex_dt) < timedelta(minutes=cooldown_min)
        except ValueError:
            return False

    def ensure_daily_stats(self, current_equity: float, now: Optional[datetime] = None) -> Dict[str, Any]:
        today = utc_today_str(now)
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM daily_stats WHERE date_utc = ?", (today,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO daily_stats (date_utc, start_equity, realized_pnl_usdt, trades_count, sl_count, is_halted) "
                    "VALUES (?, ?, 0.0, 0, 0, 0)",
                    (today, current_equity),
                )
                conn.commit()
                return {
                    "date_utc": today,
                    "start_equity": current_equity,
                    "realized_pnl_usdt": 0.0,
                    "trades_count": 0,
                    "sl_count": 0,
                    "is_halted": 0,
                }
            return dict(row)

    def _record_daily_close(self, net_pnl_usdt: float, is_sl: bool, halted: bool, now: Optional[datetime] = None) -> None:
        today = utc_today_str(now)
        with self._connect() as conn:
            conn.execute(
                "UPDATE daily_stats SET realized_pnl_usdt = realized_pnl_usdt + ?, "
                "trades_count = trades_count + 1, sl_count = sl_count + ?, "
                "is_halted = MAX(is_halted, ?) WHERE date_utc = ?",
                (net_pnl_usdt, 1 if is_sl else 0, 1 if halted else 0, today),
            )
            conn.commit()

    def get_current_equity(self, fallback_paper_equity: float = 1000.0) -> float:
        if self.client.has_credentials:
            try:
                bal = self.client.get_wallet_balance("USDT")
                eq = float(bal.get("equity", 0.0))
                if eq > 0:
                    return eq
            except Exception as exc:
                if not self.dry_run:
                    raise
                logger.warning("Dry-run live balance fetch failed (%s), using fallback equity.", exc)
        if self.dry_run:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COALESCE(SUM(net_pnl_usdt), 0.0) AS pnl FROM trades WHERE status = 'CLOSED'"
                ).fetchone()
                realized = float(row["pnl"]) if row else 0.0
            return max(10.0, fallback_paper_equity + realized)
        bal = self.client.get_wallet_balance("USDT")
        return float(bal.get("equity", 0.0))

    # ------------------------------------------------------------------
    # Production Entry: Hard Macro Shield Gate + Isolated 15x + PostOnly Maker + Server SL
    # ------------------------------------------------------------------
    def open_position(
        self,
        signal: TradeSignal,
        current_equity: float,
        now: Optional[datetime] = None,
        allow_network_refresh: bool = True,
    ) -> Optional[int]:
        cur_now = now or utc_now()

        # HARD CORE GATE: US Macro Shield (-30m..+15m UTC OR fail-safe stale calendar)
        shield = self._get_macro_shield(now=cur_now, allow_network_refresh=allow_network_refresh)
        if not shield.get("entry_allowed", False):
            lvl = logging.ERROR if shield.get("fail_safe_active") else logging.WARNING
            logger.log(
                lvl,
                "HARD BLOCKED %s %s in TradeExecutor.open_position: %s",
                signal.symbol, signal.direction, shield.get("reason"),
            )
            return None

        ttl_min_gate = int(self.config.get("maker_ttl_bars", 4)) * 15
        pre_min_gate = int(self.config.get("macro_pre_window_min", 30))
        mins_to_event = minutes_to_next_event(shield, now=cur_now)
        if mins_to_event is not None and mins_to_event < pre_min_gate + ttl_min_gate:
            logger.warning(
                "Blocked %s %s: core macro event in %.0f min, limit TTL (%d min) would overlap the pre-news window.",
                signal.symbol, signal.direction, mins_to_event, ttl_min_gate,
            )
            return None

        daily = self.ensure_daily_stats(current_equity, now=cur_now)
        active = self.get_active_trades()

        max_daily_stops = int(self.config.get("max_daily_stops", 3))
        if int(daily.get("sl_count", 0)) >= max_daily_stops or int(daily.get("is_halted", 0)) == 1:
            logger.warning("Blocked %s: daily stop limit reached (%d/%d SLs).", signal.symbol, daily.get("sl_count", 0), max_daily_stops)
            return None

        halted_dd, dd_pct = self.risk_manager.is_daily_loss_halted(float(daily["start_equity"]), current_equity)
        if halted_dd:
            logger.warning("Blocked %s: daily equity drawdown limit reached (%.2f%%).", signal.symbol, dd_pct)
            return None

        if self.is_symbol_in_cooldown(signal.symbol, now=cur_now):
            logger.info("Blocked %s: post-trade cooldown active.", signal.symbol)
            return None

        max_open = int(self.config.get("max_open_positions", 1))
        max_same = int(self.config.get("max_same_direction", 1))
        if len(active) >= max_open:
            logger.info("Blocked %s: max active positions reached (%d/%d).", signal.symbol, len(active), max_open)
            return None
        if any(t["symbol"] == signal.symbol for t in active):
            logger.info("Blocked %s: position/order already active.", signal.symbol)
            return None
        if sum(1 for t in active if t["direction"] == signal.direction) >= max_same:
            logger.info("Blocked %s %s: max same-direction limit reached (%d).", signal.symbol, signal.direction, max_same)
            return None

        leverage = int(self.config.get("leverage", 15))
        margin_pct = float(self.config.get("margin_pct_per_trade", 15.0))
        margin_usdt = current_equity * (margin_pct / 100.0)
        notional = margin_usdt * leverage
        max_notional = float(self.config.get("max_notional_usdt", 0.0) or 0.0)
        if max_notional > 0.0 and notional > max_notional:
            notional = max_notional
            margin_usdt = notional / leverage

        # Global guard across all bots sharing the Bybit account; fails closed
        max_portfolio_pos = int(self.config.get("max_portfolio_open_positions", 3))
        max_portfolio_margin_pct = float(self.config.get("max_portfolio_margin_pct", 45.0))
        if not self.dry_run and self.client.has_credentials:
            try:
                live_pos = self.client.get_positions()
                live_orders = self.client.get_open_orders()
            except Exception as exc:
                logger.warning("Blocked %s: could not verify global portfolio on Bybit (%s).", signal.symbol, exc)
                return None
            other_open = [p for p in live_pos if str(p.get("symbol", "")).upper() != signal.symbol]
            if len(other_open) >= max_portfolio_pos:
                logger.info(
                    "Blocked %s: global portfolio open positions limit reached on Bybit (%d/%d open: %s).",
                    signal.symbol,
                    len(other_open),
                    max_portfolio_pos,
                    [p.get("symbol") for p in other_open],
                )
                return None
            used_im = sum(float(p.get("positionIM") or 0.0) for p in other_open)
            for o in live_orders:
                if str(o.get("symbol", "")).upper() == signal.symbol or o.get("reduceOnly"):
                    continue
                try:
                    used_im += float(o.get("qty") or 0.0) * float(o.get("price") or 0.0) / max(leverage, 1)
                except (TypeError, ValueError):
                    continue
            if used_im + margin_usdt > current_equity * max_portfolio_margin_pct / 100.0:
                logger.info(
                    "Blocked %s: account margin %.2f + %.2f USDT would exceed %.1f%% of equity.",
                    signal.symbol, used_im, margin_usdt, max_portfolio_margin_pct,
                )
                return None

        used_margin = sum(float(t["margin_usdt"]) for t in active)
        if used_margin + margin_usdt > current_equity * 0.95:
            logger.info("Blocked %s: insufficient free isolated margin.", signal.symbol)
            return None

        entry_price = float(signal.entry_price)
        spec = signal.spec
        if spec is not None:
            qty = round_step_down(notional / entry_price, spec.qty_step)
            if qty < spec.min_order_qty or (qty * entry_price) < max(spec.min_notional, 5.0):
                logger.info("Rejected %s: qty %.6f below Bybit minimums.", signal.symbol, qty)
                return None
            notional = qty * entry_price
            margin_usdt = notional / leverage
        else:
            qty = notional / entry_price

        tp_frac = (float(signal.tp_pct) if signal.tp_pct is not None else float(self.config.get("tp_pct", 1.20))) / 100.0
        sl_frac = (float(signal.sl_pct) if signal.sl_pct > 0 else float(self.config.get("sl_pct", 2.50))) / 100.0
        be1_tr_frac = (float(signal.be1_trigger_pct) if signal.be1_trigger_pct is not None else float(self.config.get("smart_be1_trigger_pct", 0.80))) / 100.0
        be1_off_frac = (float(signal.be1_offset_pct) if signal.be1_offset_pct is not None else float(self.config.get("smart_be1_offset_pct", 0.30))) / 100.0
        be2_tr_frac = (float(signal.be2_trigger_pct) if signal.be2_trigger_pct is not None else float(self.config.get("smart_be2_trigger_pct", 0.95))) / 100.0
        be2_off_frac = (float(signal.be2_offset_pct) if signal.be2_offset_pct is not None else float(self.config.get("smart_be2_offset_pct", 0.50))) / 100.0
        is_long = signal.direction == "LONG"

        if is_long:
            sl_price = entry_price * (1.0 - sl_frac)
            tp_price = float(signal.tp_price) if signal.tp_price is not None else entry_price * (1.0 + tp_frac)
            be1_trigger = entry_price * (1.0 + be1_tr_frac)
            be1_price = entry_price * (1.0 + be1_off_frac)
            be2_trigger = entry_price * (1.0 + be2_tr_frac) if be2_tr_frac > 0 else 0.0
            be2_price = entry_price * (1.0 + be2_off_frac) if be2_tr_frac > 0 else 0.0
        else:
            sl_price = entry_price * (1.0 + sl_frac)
            tp_price = float(signal.tp_price) if signal.tp_price is not None else entry_price * (1.0 - tp_frac)
            be1_trigger = entry_price * (1.0 - be1_tr_frac)
            be1_price = entry_price * (1.0 - be1_off_frac)
            be2_trigger = entry_price * (1.0 - be2_tr_frac) if be2_tr_frac > 0 else 0.0
            be2_price = entry_price * (1.0 - be2_off_frac) if be2_tr_frac > 0 else 0.0

        if spec is not None and spec.tick_size > 0:
            entry_price = round_tick(entry_price, spec.tick_size)
            sl_price = round_tick(sl_price, spec.tick_size)
            tp_price = round_tick(tp_price, spec.tick_size)
            be1_trigger = round_tick(be1_trigger, spec.tick_size)
            be1_price = round_tick(be1_price, spec.tick_size)
            if be2_trigger > 0:
                be2_trigger = round_tick(be2_trigger, spec.tick_size)
                be2_price = round_tick(be2_price, spec.tick_size)

        ttl_bars = int(self.config.get("maker_ttl_bars", 4))
        ttl_min = ttl_bars * 15
        expires_iso = utc_now_iso(cur_now + timedelta(minutes=ttl_min))
        entry_side = "Buy" if is_long else "Sell"
        exec_mode = "paper" if self.dry_run else str(self.config.get("mode", "testnet"))
        entry_order_id = f"DRY-MAKER-{int(cur_now.timestamp())}"

        if not self.dry_run:
            self.client.ensure_isolated_margin_and_leverage(signal.symbol, leverage)
            res = self.client.place_limit_entry_with_sl(
                symbol=signal.symbol,
                side=entry_side,
                qty=qty,
                price=entry_price,
                stop_loss=sl_price,
            )
            entry_order_id = str(res.get("orderId", ""))

        try:
            trade_id = self._insert_pending_trade(
                signal, exec_mode, cur_now, expires_iso, entry_price, sl_price, tp_price,
                be1_trigger, be1_price, be2_trigger, be2_price, qty, leverage, margin_usdt, notional, entry_order_id,
            )
        except Exception:
            if not self.dry_run and entry_order_id:
                logger.error("DB insert failed after placing %s, cancelling order %s.", signal.symbol, entry_order_id)
                self.client.cancel_order(signal.symbol, entry_order_id)
            raise

        dir_badge = "🟢 LONG" if is_long else "🔴 SHORT"
        self._notify(
            f"⏳ ВЫСТАВЛЕН ЛИМИТНЫЙ ОРДЕР ({exec_mode.upper()}) | {signal.symbol} {dir_badge}\n"
            f"Плечо: {leverage}x Isolated | Маржа: ${margin_usdt:.2f}\n"
            f"Цена входа (Limit): {entry_price} (TTL {ttl_bars} св. / {ttl_min} мин)\n"
            f"Тейк (TP): {tp_price} (+{tp_frac*100:.2f}%) | Стоп (SL): {sl_price} (-{sl_frac*100:.2f}%)\n"
            f"БУ1: {be1_trigger} -> {be1_price} | БУ2: {be2_trigger} -> {be2_price}\n"
            f"💰 Баланс: ${current_equity:,.2f}"
        )
        return trade_id

    def _insert_pending_trade(
        self, signal: TradeSignal, exec_mode: str, cur_now: datetime, expires_iso: str,
        entry_price: float, sl_price: float, tp_price: float,
        be1_trigger: float, be1_price: float, be2_trigger: float, be2_price: float,
        qty: float, leverage: int, margin_usdt: float, notional: float, entry_order_id: str,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO trades (
                    symbol, direction, status, mode, entry_time, expires_at_utc,
                    entry_price, sl_price, tp1_price, tp2_price,
                    be_trigger_price, be_price, be2_trigger_price, be2_price,
                    qty, tp1_qty, tp2_qty, remaining_qty,
                    leverage, margin_usdt, notional_usdt,
                    score, score_details_json, entry_order_id, fees_usdt
                ) VALUES (?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.0, ?, ?, ?, ?, ?, ?, ?, 0.0)
                """,
                (
                    signal.symbol, signal.direction, exec_mode, utc_now_iso(cur_now), expires_iso,
                    entry_price, sl_price, tp_price, tp_price,
                    be1_trigger, be1_price, be2_trigger, be2_price,
                    qty, qty, qty, leverage, margin_usdt, notional,
                    signal.score, json.dumps(signal.breakdown), entry_order_id,
                ),
            )
            trade_id = int(cur.lastrowid)
            conn.commit()
        return trade_id

    # ------------------------------------------------------------------
    # Position Lifecycle Sync: Macro Shield + TTL + 2-Step Smart BE + 1H Flip
    # ------------------------------------------------------------------
    def sync_positions(
        self,
        now: Optional[datetime] = None,
        price_overrides: Optional[Dict[str, float]] = None,
        trend_overrides: Optional[Dict[str, str]] = None,
        allow_network_refresh: bool = True,
    ) -> Dict[str, Any]:
        """Manage pending limits, macro-shield cancellations & pre-news closes, 2-step BE, and 1H flip."""
        cur_now = now or utc_now()
        summary = {"cancelled_macro": 0, "closed_macro": 0, "closed_h1_flip": 0, "be1_moved": 0, "be2_moved": 0}
        active = self.get_active_trades()
        if not active:
            return summary

        shield = self._get_macro_shield(now=cur_now, allow_network_refresh=allow_network_refresh)
        cancel_limits_macro = bool(shield.get("cancel_pending_limits", False))
        protect_open_macro = bool(shield.get("protect_open_positions", False))
        protect_upnl_thresh = float(self.config.get("macro_protect_upnl_pct", 0.25))
        smart_be_enabled = bool(self.config.get("smart_be_enabled", True))
        exit_on_h1_flip = bool(self.config.get("exit_on_h1_flip", True))

        if self.dry_run:
            return self._sync_paper_positions(
                active=active,
                now=cur_now,
                shield=shield,
                cancel_limits_macro=cancel_limits_macro,
                protect_open_macro=protect_open_macro,
                protect_upnl_thresh=protect_upnl_thresh,
                smart_be_enabled=smart_be_enabled,
                exit_on_h1_flip=exit_on_h1_flip,
                price_overrides=price_overrides,
                trend_overrides=trend_overrides,
                summary=summary,
            )

        live_positions = {p["symbol"]: p for p in self.client.get_positions()}
        tickers: Optional[Dict[str, Dict[str, float]]] = None

        for trade in active:
            sym = trade["symbol"]
            pos = live_positions.get(sym)
            pos_size = float(pos.get("size") or 0.0) if pos else 0.0

            # 1. Handle PENDING Maker Limit Entry
            if trade["status"] == "PENDING":
                order = self.client.get_order_status(sym, trade["entry_order_id"]) if trade["entry_order_id"] else None
                ord_status = str(order.get("orderStatus", "")) if order else ""

                if ord_status == "Filled":
                    self._on_maker_filled(trade, pos_size if pos_size > 0 else float(trade["qty"]))
                    continue

                if ord_status in ("Cancelled", "Rejected", "Deactivated", "PartiallyFilledCanceled"):
                    if pos_size > 0:
                        self._on_maker_filled(trade, pos_size)
                    else:
                        self._mark_cancelled(trade["id"], f"Exchange status: {ord_status}", now=cur_now)
                    continue

                if pos_size > 0 and not order:
                    self._on_maker_filled(trade, pos_size)
                    continue

                exp_dt = datetime.fromisoformat(str(trade["expires_at_utc"]).replace("Z", "+00:00")) if trade.get("expires_at_utc") else None
                expired = exp_dt is not None and cur_now >= exp_dt
                if not (cancel_limits_macro or expired):
                    # Partially filled orders keep working until filled, expired or blocked by the shield
                    continue

                if trade["entry_order_id"]:
                    self.client.cancel_order(sym, trade["entry_order_id"])
                filled_now = self._live_position_size(sym)
                if filled_now > 0:
                    self._on_maker_filled(trade, filled_now)
                    continue
                if cancel_limits_macro:
                    self._mark_cancelled(trade["id"], f"US Macro Shield cancel: {shield.get('reason')}", now=cur_now)
                    summary["cancelled_macro"] += 1
                else:
                    self._mark_cancelled(trade["id"], f"Maker TTL expired ({self.config.get('maker_ttl_bars', 4)} bars)", now=cur_now)
                continue

            # 2. Handle OPEN / BE_ACTIVE / BE2_ACTIVE live positions
            if pos_size <= 0.0:
                try:
                    self.client.cancel_all_orders(sym)
                except Exception:
                    pass
                self._finalize_live_closed(trade, now=cur_now)
                continue

            price = float(price_overrides[sym]) if (price_overrides and sym in price_overrides) else float(pos.get("markPrice") or 0.0)
            if price <= 0:
                if tickers is None:
                    tickers = {} if price_overrides else self.client.get_tickers()
                if sym in tickers:
                    price = tickers[sym]["last_price"]
            is_long = trade["direction"] == "LONG"
            ep = float(trade["entry_price"])
            upnl_pct = ((price - ep) / ep * 100.0) if (is_long and price > 0) else (((ep - price) / ep * 100.0) if price > 0 else 0.0)

            # 2A. HARD US MACRO SHIELD (-30m..0m): close open position at market if uPnL < +0.25%
            if protect_open_macro and price > 0 and upnl_pct < protect_upnl_thresh:
                self._force_close_market(trade, price, pos_size, "MACRO_PRE_NEWS", now=cur_now)
                summary["closed_macro"] += 1
                continue

            # 2B. 100% close on full 1H EMA20/EMA50 flip against position
            if exit_on_h1_flip and price > 0:
                h1_trend = self._resolve_1h_trend(sym, trend_overrides)
                if (is_long and h1_trend == "BEAR") or ((not is_long) and h1_trend == "BULL"):
                    self._force_close_market(trade, price, pos_size, "H1_FLIP", now=cur_now)
                    summary["closed_h1_flip"] += 1
                    continue

            if price > 0 and self._ensure_tp_order(trade, pos_size, price, now=cur_now):
                continue

            # 2C. 2-Step Smart Break-Even (+0.80% -> +0.30%, +0.95% -> +0.50%)
            if smart_be_enabled and price > 0:
                be2_trig = float(trade.get("be2_trigger_price") or 0.0)
                be1_trig = float(trade.get("be_trigger_price") or 0.0)
                if trade["status"] in ("OPEN", "BE_ACTIVE") and be2_trig > 0 and ((price >= be2_trig) if is_long else (price <= be2_trig)):
                    if self._activate_smart_be_step(trade, step=2):
                        summary["be2_moved"] += 1
                elif trade["status"] == "OPEN" and be1_trig > 0 and ((price >= be1_trig) if is_long else (price <= be1_trig)):
                    if self._activate_smart_be_step(trade, step=1):
                        summary["be1_moved"] += 1

        return summary

    def _resolve_1h_trend(self, sym: str, trend_overrides: Optional[Dict[str, str]] = None) -> str:
        if trend_overrides and sym in trend_overrides:
            return str(trend_overrides[sym]).upper()
        if self.trend_provider is not None:
            try:
                return str(self.trend_provider(sym)).upper()
            except Exception as exc:
                logger.warning("1H trend provider error for %s: %s", sym, exc)
        return "NEUTRAL"

    def _force_close_market(self, trade: Dict[str, Any], exit_price: float, qty: float, reason: str, now: Optional[datetime] = None) -> None:
        sym = trade["symbol"]
        close_side = "Sell" if trade["direction"] == "LONG" else "Buy"
        if not self.dry_run:
            try:
                self.client.cancel_all_orders(sym)
            except Exception as exc:
                logger.warning("cancel_all_orders before %s close on %s failed: %s", reason, sym, exc)
            try:
                self.client.close_position_market(symbol=sym, side=close_side, qty=qty)
            except Exception as exc:
                logger.error("Failed force market close (%s) on %s: %s", reason, sym, exc)
                return
            # Booked on the next sync from the exchange closed-PnL record
            with self._connect() as conn:
                conn.execute("UPDATE trades SET exit_reason = ? WHERE id = ?", (reason, trade["id"]))
                conn.commit()
            return

        gross = self.risk_manager.leg_gross_pnl(trade["direction"], float(trade["entry_price"]), exit_price, qty)
        ex_fee = self.risk_manager.fee_usdt(qty * exit_price, is_maker=False)
        self._commit_closed(trade, reason, gross, ex_fee, now=now)

    def _on_maker_filled(self, trade: Dict[str, Any], fill_qty: float) -> None:
        sym = trade["symbol"]
        close_side = "Sell" if trade["direction"] == "LONG" else "Buy"
        tp_price = float(trade["tp1_price"])
        tp_order_id = f"DRY-TP-{int(utc_now().timestamp())}"

        if not self.dry_run:
            try:
                res = self.client.place_limit_tp_order(symbol=sym, side=close_side, qty=fill_qty, price=tp_price)
                tp_order_id = str(res.get("orderId", ""))
            except Exception as exc:
                logger.error("Failed placing 100%% PostOnly Limit TP for %s: %s", sym, exc)
                tp_order_id = ""

        entry_fee = self.risk_manager.fee_usdt(fill_qty * float(trade["entry_price"]), is_maker=True)
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET status = 'OPEN', qty = ?, tp1_qty = ?, remaining_qty = ?, "
                "tp1_order_id = ?, fees_usdt = ? WHERE id = ?",
                (fill_qty, fill_qty, fill_qty, tp_order_id, entry_fee, trade["id"]),
            )
            conn.commit()

        is_long = str(trade.get("direction", "")).upper() == "LONG"
        dir_badge = "🟢 LONG" if is_long else "🔴 SHORT"
        lev = int(trade.get("leverage") or self.config.get("leverage", 10))
        margin_val = float(trade.get("margin_usdt") or 0.0)
        ep_val = float(trade.get("entry_price") or 0.0)
        tp_pct_val = (abs(tp_price - ep_val) / ep_val * 100.0) if ep_val > 0 else float(self.config.get("tp_pct", 1.2))
        cur_eq = self.get_current_equity()
        tp_note = "" if tp_order_id else " (не выставлен, повторю на следующей синхронизации)"
        self._notify(
            f"🚀 СДЕЛКА НАЧАЛАСЬ | {sym} {dir_badge}\n"
            f"Плечо: {lev}x Isolated | Маржа: ${margin_val:.2f} | Объём: {fill_qty}\n"
            f"Цена входа: {trade['entry_price']}\n"
            f"Тейк (TP): {tp_price} (+{tp_pct_val:.2f}%){tp_note} | Стоп (SL): {trade['sl_price']}\n"
            f"💰 Баланс: ${cur_eq:,.2f}"
        )

    def _activate_smart_be_step(self, trade: Dict[str, Any], step: int) -> bool:
        sym = trade["symbol"]
        ep = float(trade.get("entry_price") or 0.0)
        target_sl = float(trade["be2_price"]) if step == 2 else float(trade["be_price"])
        new_status = "BE2_ACTIVE" if step == 2 else "BE_ACTIVE"
        lock_pct = (abs(target_sl - ep) / ep * 100.0) if ep > 0 else float(
            self.config.get("smart_be2_offset_pct", 0.50) if step == 2 else self.config.get("smart_be1_offset_pct", 0.30)
        )
        label = f"BE{step} (+{lock_pct:.2f}%)"

        if not self.dry_run:
            try:
                self.client.set_trading_stop(symbol=sym, stop_loss=target_sl)
            except Exception as exc:
                logger.error("Failed moving SL to %s on %s: %s", label, sym, exc)
                return False

        with self._connect() as conn:
            conn.execute("UPDATE trades SET status = ?, sl_price = ? WHERE id = ?", (new_status, target_sl, trade["id"]))
            conn.commit()
        is_long = str(trade.get("direction", "")).upper() == "LONG"
        dir_badge = "🟢 LONG" if is_long else "🔴 SHORT"
        lev = int(trade.get("leverage") or self.config.get("leverage", 10))
        self._notify(
            f"🛡 БЕЗУБЫТОК {label} | {sym} {dir_badge}\n"
            f"Плечо: {lev}x Isolated | Стоп переведён на: {target_sl}"
        )
        return True

    def _mark_cancelled(self, trade_id: int, reason: str, now: Optional[datetime] = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET status = 'CANCELLED', exit_time = ?, exit_reason = ?, remaining_qty = 0.0 WHERE id = ?",
                (utc_now_iso(now), reason, trade_id),
            )
            conn.commit()
        logger.info("Pending Maker trade #%d cancelled: %s", trade_id, reason)

    def _live_position_size(self, symbol: str) -> float:
        try:
            for p in self.client.get_positions(symbol=symbol):
                if p.get("symbol") == symbol:
                    return float(p.get("size") or 0.0)
        except Exception as exc:
            logger.warning("Position size check failed for %s: %s", symbol, exc)
        return 0.0

    def _ensure_tp_order(self, trade: Dict[str, Any], pos_size: float, price: float, now: Optional[datetime] = None) -> bool:
        """Keep a reduce-only TP for the full live size. Returns True if the position was closed instead."""
        sym = trade["symbol"]
        is_long = trade["direction"] == "LONG"
        tp_price = float(trade["tp1_price"])
        tp_id = str(trade.get("tp1_order_id") or "")
        if tp_id:
            order = self.client.get_order_status(sym, tp_id)
            status = str(order.get("orderStatus", "")) if order else ""
            if status in ("Filled", "PartiallyFilled"):
                return False
            if status in ("New", "Untriggered"):
                order_qty = float(order.get("qty") or 0.0)
                if abs(order_qty - pos_size) <= max(1e-9, pos_size * 1e-6):
                    return False
                self.client.cancel_order(sym, tp_id)
            tp_id = ""

        beyond_tp = price >= tp_price if is_long else price <= tp_price
        if beyond_tp:
            # A PostOnly TP behind the market is rejected, so take the profit at market
            self._force_close_market(trade, price, pos_size, "TP_MARKET", now=now)
            return True

        close_side = "Sell" if is_long else "Buy"
        try:
            res = self.client.place_limit_tp_order(symbol=sym, side=close_side, qty=pos_size, price=tp_price)
            tp_id = str(res.get("orderId", ""))
        except Exception as exc:
            logger.error("TP re-placement failed for %s: %s", sym, exc)
            return False
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET tp1_order_id = ?, qty = ?, tp1_qty = ?, remaining_qty = ? WHERE id = ?",
                (tp_id, pos_size, pos_size, pos_size, trade["id"]),
            )
            conn.commit()
        trade["tp1_order_id"] = tp_id
        trade["qty"] = pos_size
        logger.info("TP for %s re-placed at %s for qty %s.", sym, tp_price, pos_size)
        return False

    def _fetch_closed_pnl(self, trade: Dict[str, Any]) -> Optional[Dict[str, float]]:
        entry_dt = datetime.fromisoformat(str(trade["entry_time"]).replace("Z", "+00:00"))
        start_ms = int(entry_dt.timestamp() * 1000)
        try:
            rows = self.client.get_closed_pnl(trade["symbol"], start_ms=start_ms)
        except Exception as exc:
            logger.warning("closed-pnl fetch failed for %s: %s", trade["symbol"], exc)
            return None
        rows = [r for r in rows if int(r.get("createdTime") or 0) >= start_ms]
        if not rows:
            return None
        net = sum(float(r.get("closedPnl") or 0.0) for r in rows)
        qty = sum(float(r.get("closedSize") or 0.0) for r in rows)
        exit_value = sum(float(r.get("cumExitValue") or 0.0) for r in rows)
        fees: Optional[float] = None
        if all("openFee" in r and "closeFee" in r for r in rows):
            fees = sum(float(r.get("openFee") or 0.0) + float(r.get("closeFee") or 0.0) for r in rows)
        return {
            "net": net,
            "qty": qty,
            "exit_price": exit_value / qty if qty > 0 else 0.0,
            "fees": fees if fees is not None else float("nan"),
        }

    def _finalize_live_closed(self, trade: Dict[str, Any], now: Optional[datetime] = None) -> None:
        sym = trade["symbol"]
        tp_filled = False
        if trade.get("tp1_order_id"):
            ord_tp = self.client.get_order_status(sym, trade["tp1_order_id"])
            if ord_tp and ord_tp.get("orderStatus") == "Filled":
                tp_filled = True

        if tp_filled:
            exit_p, reason, is_mk = float(trade["tp1_price"]), "TP", True
        elif trade.get("exit_reason"):
            exit_p, reason, is_mk = float(trade["sl_price"]), str(trade["exit_reason"]), False
        elif trade["status"] == "BE2_ACTIVE":
            exit_p, reason, is_mk = float(trade["be2_price"]), "BE2", False
        elif trade["status"] == "BE_ACTIVE":
            exit_p, reason, is_mk = float(trade["be_price"]), "BE", False
        else:
            exit_p, reason, is_mk = float(trade["sl_price"]), "SL", False

        actual = self._fetch_closed_pnl(trade)
        if actual is None:
            # The exchange publishes closed-PnL with a short delay; wait a few syncs before estimating
            attempts = self._finalize_attempts.get(trade["id"], 0) + 1
            self._finalize_attempts[trade["id"]] = attempts
            if attempts < 5:
                return
            logger.warning("No closed-PnL record for trade #%s, booking estimated PnL.", trade["id"])
        self._finalize_attempts.pop(trade["id"], None)

        qty = float(trade["qty"])
        entry_p = float(trade["entry_price"])
        if actual is not None:
            if actual["exit_price"] > 0:
                exit_p = actual["exit_price"]
            if actual["qty"] > 0:
                qty = actual["qty"]
            gross = self.risk_manager.leg_gross_pnl(trade["direction"], entry_p, exit_p, qty)
            fees = actual["fees"] if actual["fees"] == actual["fees"] else gross - actual["net"]
            if reason == "SL" and actual["net"] >= 0:
                reason = "EXTERNAL"
            self._commit_closed(trade, reason, gross, 0.0, now=now, total_fees_override=fees, net_override=actual["net"])
            return

        gross = self.risk_manager.leg_gross_pnl(trade["direction"], entry_p, exit_p, qty)
        ex_fee = self.risk_manager.fee_usdt(qty * exit_p, is_maker=is_mk)
        self._commit_closed(trade, reason, gross, ex_fee, now=now)

    def _sync_paper_positions(
        self,
        active: List[Dict[str, Any]],
        now: datetime,
        shield: Dict[str, Any],
        cancel_limits_macro: bool,
        protect_open_macro: bool,
        protect_upnl_thresh: float,
        smart_be_enabled: bool,
        exit_on_h1_flip: bool,
        price_overrides: Optional[Dict[str, float]],
        trend_overrides: Optional[Dict[str, str]],
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        tickers = {} if price_overrides else self.client.get_tickers()

        for trade in active:
            sym = trade["symbol"]
            if price_overrides and sym in price_overrides:
                price = float(price_overrides[sym])
            else:
                t = tickers.get(sym)
                if not t or t["last_price"] <= 0:
                    continue
                price = float(t["last_price"])
            is_long = trade["direction"] == "LONG"

            if trade["status"] == "PENDING":
                if cancel_limits_macro:
                    self._mark_cancelled(trade["id"], f"US Macro Shield cancel (paper): {shield.get('reason')}", now=now)
                    summary["cancelled_macro"] += 1
                    continue
                if (price < float(trade["entry_price"])) if is_long else (price > float(trade["entry_price"])):
                    self._on_maker_filled(trade, float(trade["qty"]))
                else:
                    exp_dt = datetime.fromisoformat(str(trade["expires_at_utc"]).replace("Z", "+00:00")) if trade.get("expires_at_utc") else None
                    if exp_dt is not None and now >= exp_dt:
                        self._mark_cancelled(trade["id"], "Maker TTL expired (paper)", now=now)
                continue

            if trade["status"] in ("OPEN", "BE_ACTIVE", "BE2_ACTIVE"):
                ep = float(trade["entry_price"])
                upnl_pct = ((price - ep) / ep * 100.0) if is_long else ((ep - price) / ep * 100.0)

                sl_hit = (price <= float(trade["sl_price"])) if is_long else (price >= float(trade["sl_price"]))
                tp_hit = (price >= float(trade["tp1_price"])) if is_long else (price <= float(trade["tp1_price"]))
                if sl_hit:
                    reason = "BE2" if trade["status"] == "BE2_ACTIVE" else ("BE" if trade["status"] == "BE_ACTIVE" else "SL")
                    exit_p = float(trade["sl_price"])
                    gross = self.risk_manager.leg_gross_pnl(trade["direction"], ep, exit_p, float(trade["qty"]))
                    ex_fee = self.risk_manager.fee_usdt(float(trade["qty"]) * exit_p, is_maker=False)
                    self._commit_closed(trade, reason, gross, ex_fee, now=now)
                elif tp_hit:
                    exit_p = float(trade["tp1_price"])
                    gross = self.risk_manager.leg_gross_pnl(trade["direction"], ep, exit_p, float(trade["qty"]))
                    ex_fee = self.risk_manager.fee_usdt(float(trade["qty"]) * exit_p, is_maker=True)
                    self._commit_closed(trade, "TP", gross, ex_fee, now=now)
                elif protect_open_macro and upnl_pct < protect_upnl_thresh:
                    self._force_close_market(trade, price, float(trade["qty"]), "MACRO_PRE_NEWS", now=now)
                    summary["closed_macro"] += 1
                elif exit_on_h1_flip:
                    h1_trend = self._resolve_1h_trend(sym, trend_overrides)
                    if (is_long and h1_trend == "BEAR") or ((not is_long) and h1_trend == "BULL"):
                        self._force_close_market(trade, price, float(trade["qty"]), "H1_FLIP", now=now)
                        summary["closed_h1_flip"] += 1
                        continue
                if smart_be_enabled and trade["status"] in ("OPEN", "BE_ACTIVE"):
                    be2_trig = float(trade.get("be2_trigger_price") or 0.0)
                    be1_trig = float(trade.get("be_trigger_price") or 0.0)
                    if be2_trig > 0 and ((price >= be2_trig) if is_long else (price <= be2_trig)):
                        if self._activate_smart_be_step(trade, step=2):
                            summary["be2_moved"] += 1
                    elif trade["status"] == "OPEN" and be1_trig > 0 and ((price >= be1_trig) if is_long else (price <= be1_trig)):
                        if self._activate_smart_be_step(trade, step=1):
                            summary["be1_moved"] += 1
        return summary

    def _commit_closed(
        self,
        trade: Dict[str, Any],
        reason: str,
        gross: float,
        exit_fee: float,
        now: Optional[datetime] = None,
        total_fees_override: Optional[float] = None,
        net_override: Optional[float] = None,
    ) -> None:
        total_fees = total_fees_override if total_fees_override is not None else float(trade["fees_usdt"]) + exit_fee
        net_pnl = net_override if net_override is not None else gross - total_fees
        margin = max(float(trade["margin_usdt"]), 1e-9)
        net_pct_margin = (net_pnl / margin) * 100.0

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET status = 'CLOSED', exit_time = ?, remaining_qty = 0.0,
                    exit_reason = ?, gross_pnl_usdt = ?, fees_usdt = ?,
                    net_pnl_usdt = ?, net_pnl_pct = ?
                WHERE id = ?
                """,
                (utc_now_iso(now), reason, round(gross, 6), round(total_fees, 6), round(net_pnl, 6), round(net_pct_margin, 4), trade["id"]),
            )
            conn.commit()

        equity = self.get_current_equity()
        # If this close is the first event of the day, the day started before its PnL was realized
        daily = self.ensure_daily_stats(equity - net_pnl, now=now)
        is_sl = reason == "SL"
        max_stops = int(self.config.get("max_daily_stops", 3))
        halted_dd, _ = self.risk_manager.is_daily_loss_halted(float(daily["start_equity"]), equity)
        halted = halted_dd or ((int(daily.get("sl_count", 0)) + (1 if is_sl else 0)) >= max_stops)
        self._record_daily_close(net_pnl, is_sl=is_sl, halted=halted, now=now)

        is_long = str(trade.get("direction", "")).upper() == "LONG"
        dir_badge = "🟢 LONG" if is_long else "🔴 SHORT"
        lev = int(trade.get("leverage") or self.config.get("leverage", 10))
        res_icon = "✅" if net_pnl >= 0 else "❌"
        self._notify(
            f"{res_icon} СДЕЛКА ЗАКРЫТА ({reason}) | {trade['symbol']} {dir_badge}\n"
            f"Плечо: {lev}x Isolated | Вход: {trade['entry_price']}\n"
            f"Результат: {net_pnl:+.2f} USDT ({net_pct_margin:+.2f}% к марже) | Комиссия: ${total_fees:.2f}\n"
            f"💰 Баланс: ${equity:,.2f}"
        )
