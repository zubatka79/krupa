"""
Bybit v5 Execution & 20-second Exchange Sync Engine for VIKA_N2_N7.

Features:
  - Cross Margin + 10x leverage setup across all 10 coins
  - Atomic server-side TP and SL attached directly in Bybit place_order
  - Closed-candle BE StopLoss modification on Bybit (set_trading_stop)
  - 20-second sync loop tracking fills, TTL cancellations, server-side TP/BE/SL exits,
    live wallet balance for 10% compounding, and -6.0% UTC daily hard stop
  - Paper mode simulation fallback (when --paper or dry_run=True)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from bot.bybit_client import BybitClient, InstrumentSpec, round_step_down, round_tick
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.vika_engine import CoinEvaluation, M15_MS

logger = logging.getLogger("vika.executor")


def _is_entry_limit_order(order: Dict[str, Any]) -> bool:
    """Return True strictly for opening Limit entry orders (never TP/SL or reduceOnly)."""
    if str(order.get("orderType", "")).lower() != "limit":
        return False
    reduce_only = order.get("reduceOnly")
    if reduce_only is True or str(reduce_only).lower() == "true":
        return False
    stop_type = str(order.get("stopOrderType", "") or "").strip()
    if stop_type and stop_type.upper() != "UNKNOWN":
        return False
    order_filter = str(order.get("orderFilter", "Order") or "Order").strip()
    if order_filter not in ("Order", ""):
        return False
    return True


class VikaExecutor:
    def __init__(
        self,
        cfg: Dict[str, Any],
        client: BybitClient,
        journal: VikaJournal,
        notifier: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.journal = journal
        self.notify = notifier or (lambda msg: None)

        self.strategy_name = str(cfg.get("strategy_name", "VIKA_N2_N7"))
        self.mode = str(cfg.get("mode", "demo")).lower()
        self.dry_run = bool(cfg.get("dry_run", False)) or (self.mode == "paper")
        self.leverage = int(cfg.get("leverage", 10))
        self.margin_pct = float(cfg.get("margin_pct_per_trade", 10.0)) / 100.0
        self.max_open = int(cfg.get("max_open_positions", 8))
        self.max_total_margin_pct = float(cfg.get("max_total_margin_pct", 80.0)) / 100.0
        self.daily_loss_limit_pct = float(cfg.get("daily_loss_limit_pct", 6.0))
        self.ttl_bars = int(cfg.get("limit_entry_ttl_bars", 3))
        self.ttl_sec = int(cfg.get("limit_entry_ttl_sec", 2700))
        self.cooldown_bars = int(cfg.get("cooldown_bars_after_close", 4))

        self.coin_cfg_map: Dict[str, Dict[str, Any]] = {c["symbol"]: c for c in cfg["coins"]}
        self.specs: Dict[str, InstrumentSpec] = {}
        self.last_equity: float = 2000.0
        self.last_available: float = 2000.0
        self.last_exchange_positions: Dict[str, Dict[str, Any]] = {}
        self.stale_warnings: List[str] = []
        self.startup_cancelled_orders: List[str] = []

    # ------------------------------------------------------------------
    # Startup Setup: Instruments, Cross Margin + 10x Leverage, Stale Check
    # ------------------------------------------------------------------
    def load_instruments(self) -> Dict[str, InstrumentSpec]:
        all_specs = self.client.get_linear_instruments()
        self.specs = {}
        missing = []
        for sym in self.coin_cfg_map:
            if sym in all_specs:
                self.specs[sym] = all_specs[sym]
            else:
                missing.append(sym)
        if missing:
            raise RuntimeError(f"Missing symbols on Bybit Linear USDT: {missing}")
        return self.specs

    def ensure_cross_margin_and_leverage(self) -> Dict[str, str]:
        """Enforce Cross Margin (tradeMode=0 / UTA REGULAR_MARGIN) and 10x leverage for all 10 coins."""
        results: Dict[str, str] = {}
        if self.dry_run or not self.client.has_credentials:
            for sym in self.coin_cfg_map:
                results[sym] = f"DRY_RUN ({self.leverage}x Cross)"
            return results

        priv = self.client._require_private()
        # 1. Ensure UTA account-level Cross Margin (REGULAR_MARGIN)
        try:
            resp = priv.set_margin_mode(setMarginMode="REGULAR_MARGIN")
            rc = resp.get("retCode")
            if rc not in (0, 110026, 100028, 3400045):
                logger.info("set_margin_mode(REGULAR_MARGIN) response: %s", resp)
        except Exception as exc:
            logger.info("UTA set_margin_mode note: %s", exc)

        # 2. Enforce 10x leverage (+ classic cross mode if non-UTA) per symbol
        lev_str = str(self.leverage)
        for sym in self.coin_cfg_map:
            try:
                try:
                    priv.switch_margin_mode(
                        category="linear",
                        symbol=sym,
                        tradeMode=0,  # 0 = Cross Margin
                        buyLeverage=lev_str,
                        sellLeverage=lev_str,
                    )
                except Exception:
                    pass  # UTA accounts ignore per-symbol switch_margin_mode (100028)
                self.client.set_leverage(sym, self.leverage)
                results[sym] = f"OK ({self.leverage}x Cross)"
            except Exception as exc:
                results[sym] = f"ERR ({exc})"
                logger.error("Failed 10x Cross setup for %s: %s", sym, exc)
        return results

    def check_preexisting_positions(self) -> List[str]:
        """Inspect exchange for existing positions/orders NOT tracked in VIKA journal. Never force-close!"""
        warnings: List[str] = []
        if self.dry_run or not self.client.has_credentials:
            self.stale_warnings = []
            return warnings

        tracked_syms = {p["symbol"] for p in self.journal.get_active_positions()}
        try:
            ex_pos = self.client.get_positions()
            ex_ord = self.client.get_open_orders()
        except Exception as exc:
            msg = f"Не удалось сверить открытые позиции на Bybit: {exc}"
            logger.warning(msg)
            return [msg]

        for p in ex_pos:
            sym = p.get("symbol", "")
            if sym not in tracked_syms:
                warnings.append(
                    f"Позиция {sym} {p.get('side')} qty={p.get('size')} entry={p.get('avgPrice')} uPnL={p.get('unrealisedPnl')}"
                )
        for o in ex_ord:
            sym = o.get("symbol", "")
            if sym not in tracked_syms:
                warnings.append(
                    f"Ордер {sym} {o.get('side')} {o.get('orderType')} qty={o.get('qty')} price={o.get('price')}"
                )

        self.stale_warnings = warnings
        if warnings:
            logger.warning("Pre-existing untracked exchange items found: %s", warnings)
        return warnings

    def cleanup_stale_entry_orders(self) -> List[str]:
        """
        Startup guard: inspect all open active orders (non-TP/SL, non-reduceOnly) on Bybit
        and immediately cancel any Limit entry orders older than ttl_sec (45m / 2700s).
        Partially filled positions remain open under server-side TP/SL.
        """
        cancelled: List[str] = []
        self.startup_cancelled_orders = []
        if self.dry_run or not self.client.has_credentials:
            return cancelled

        now_ms = int(time.time() * 1000)
        ttl_ms = self.ttl_sec * 1000
        try:
            ex_ord_list = self.client.get_open_orders()
            ex_pos_list = self.client.get_positions()
        except Exception as exc:
            logger.warning("Startup stale-order check failed: %s", exc)
            return cancelled

        ex_pos_map = {p["symbol"]: p for p in ex_pos_list}
        active_journal = self.journal.get_active_positions()
        tracked_by_oid = {
            str(p.get("order_id") or ""): p for p in active_journal if p.get("order_id")
        }

        for o in ex_ord_list:
            if not _is_entry_limit_order(o):
                continue
            sym = str(o.get("symbol", ""))
            oid = str(o.get("orderId", ""))
            if not sym or not oid:
                continue
            created_ms = int(o.get("createdTime") or 0)
            age_ms = (now_ms - created_ms) if created_ms > 0 else (ttl_ms + 1)
            jpos = tracked_by_oid.get(oid)

            if age_ms >= ttl_ms:
                self.client.cancel_order(sym, oid)
                age_min = max(1, int(age_ms // 60_000))
                note = f"{sym} ({o.get('side')} Limit @ {o.get('price')}, возраст {age_min}m)"
                cancelled.append(note)
                logger.warning("Startup cancelled stale entry limit order: %s [orderId=%s]", note, oid)

                if jpos is not None:
                    if sym in ex_pos_map:
                        ep_ex = ex_pos_map[sym]
                        fill_px = float(ep_ex.get("avgPrice") or jpos["entry_price"])
                        fill_qty = float(ep_ex.get("size") or jpos["qty"])
                        fill_margin = round(fill_qty * fill_px / self.leverage, 2)
                        self.journal.update_open_fill(jpos["id"], fill_px, fill_qty, fill_margin, clear_order_id=True)
                        self._ensure_server_tpsl(sym, jpos, ep_ex)
                    else:
                        self.journal.mark_cancelled(jpos["id"], "STALE_ON_RESTART")

        self.startup_cancelled_orders = cancelled
        return cancelled

    # ------------------------------------------------------------------
    # Live Balance & Daily Risk Check (-6.0% UTC)
    # ------------------------------------------------------------------
    def refresh_balance(self) -> Tuple[float, float]:
        if self.dry_run or not self.client.has_credentials:
            stats = self.journal.get_total_realized_stats()
            self.last_equity = max(50.0, 2000.0 + float(stats["pnl_usd"]))
            used_m = sum(float(p["margin_usd"]) for p in self.journal.get_active_positions())
            self.last_available = max(0.0, self.last_equity - used_m)
            return self.last_equity, self.last_available

        bal = self.client.get_wallet_balance("USDT")
        self.last_equity = float(bal["equity"])
        used_m = sum(float(p["margin_usd"]) for p in self.journal.get_active_positions())
        raw_avail = float(bal["available"])
        self.last_available = min(raw_avail, max(0.0, self.last_equity - used_m))
        return self.last_equity, self.last_available

    def check_daily_stop(self) -> Dict[str, Any]:
        eq, _ = self.refresh_balance()
        ds = self.journal.ensure_daily_state(eq)
        start_eq = ds["start_equity"]
        daily_pnl_usd = eq - start_eq
        daily_pnl_pct = (daily_pnl_usd / start_eq * 100.0) if start_eq > 0 else 0.0

        if (not ds["halted"]) and daily_pnl_pct <= -self.daily_loss_limit_pct:
            self.journal.mark_daily_halted(notified=True)
            ds["halted"] = True
            msg = (
                f"🚨 <b>ДНЕВНОЙ СТОП -{self.daily_loss_limit_pct:.1f}% UTC АКТИВИРОВАН</b>\n"
                f"Стратегия: <b>{self.strategy_name}</b>\n"
                f"Баланс на 00:00 UTC: <code>${start_eq:,.2f}</code>\n"
                f"Текущий баланс: <code>${eq:,.2f}</code>\n"
                f"Дневной PnL: <b>{daily_pnl_usd:+,.2f} USDT ({daily_pnl_pct:+.2f}%)</b>\n"
                f"Новые входы заблокированы до 00:00 UTC. Защитные SL/TP на открытых позициях сохранены."
            )
            logger.warning("Daily loss limit hit: %.2f%%", daily_pnl_pct)
            self.notify(msg)

        ds["current_equity"] = eq
        ds["daily_pnl_usd"] = round(daily_pnl_usd, 2)
        ds["daily_pnl_pct"] = round(daily_pnl_pct, 2)
        return ds

    # ------------------------------------------------------------------
    # Atomic Order Placement (Limit GTC with Server-Side TP & SL)
    # ------------------------------------------------------------------
    def _place_entry_with_tpsl(
        self, symbol: str, side: str, qty: float, price: float, tp_price: float, sl_price: float
    ) -> str:
        if self.dry_run:
            return f"PAPER-{symbol}-{int(time.time() * 1000)}"

        priv = self.client._require_private()
        bybit_side = "Buy" if side == "LONG" else "Sell"
        resp = priv.place_order(
            category="linear",
            symbol=symbol,
            side=bybit_side,
            orderType="Limit",
            qty=str(qty),
            price=str(price),
            takeProfit=str(tp_price),
            stopLoss=str(sl_price),
            tpTriggerBy="MarkPrice",
            slTriggerBy="MarkPrice",
            tpslMode="Full",
            timeInForce="GTC",
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit place_order failed for {symbol}: {resp}")
        return str(resp["result"].get("orderId", ""))

    def open_signal(self, ev: CoinEvaluation) -> Optional[int]:
        """Attempt to open a position for a passed VIKA_N2_N7 signal."""
        sym = ev.symbol
        side = ev.final_signal
        if side not in ("LONG", "SHORT"):
            return None

        # Prevent duplicate signal processing on the same closed 15m bar
        if self.journal.get_last_signal_bar_ts(sym) >= ev.bar_ts:
            return None
        self.journal.set_last_signal_bar_ts(sym, ev.bar_ts)

        if self.journal.is_paused():
            logger.info("Skip %s %s: bot is PAUSED", sym, side)
            return None

        ds = self.check_daily_stop()
        if ds["halted"]:
            logger.info("Skip %s %s: daily -6%% stop active", sym, side)
            return None

        now_ms = int(time.time() * 1000)
        if now_ms < self.journal.get_cooldown_until_ms(sym):
            logger.info("Skip %s %s: symbol in cooldown", sym, side)
            return None

        active = self.journal.get_active_positions()
        if any(p["symbol"] == sym for p in active):
            logger.info("Skip %s %s: already active", sym, side)
            return None
        if len(active) >= self.max_open:
            logger.info("Skip %s %s: max slots %d/%d reached", sym, side, len(active), self.max_open)
            return None

        eq = self.last_equity
        if eq < 50.0:
            logger.warning("Skip %s %s: equity too low ($%.2f)", sym, side, eq)
            return None

        used_margin = sum(float(p["margin_usd"]) for p in active)
        margin_usd = eq * self.margin_pct
        if (used_margin + margin_usd) > eq * (self.max_total_margin_pct + 0.005):
            logger.info("Skip %s %s: margin cap 80%% reached", sym, side)
            return None

        spec = self.specs[sym]
        coin_cfg = self.coin_cfg_map[sym]
        entry_price = round_tick(ev.close_price, spec.tick_size)
        notional = margin_usd * self.leverage
        qty = round_step_down(notional / entry_price, spec.qty_step)
        if qty < spec.min_order_qty or (qty * entry_price) < max(spec.min_notional, 5.0):
            logger.warning("Skip %s %s: qty %.6f below min lot/notional", sym, side, qty)
            return None

        tp_pct = float(coin_cfg["tp_pct"]) / 100.0
        sl_pct = float(coin_cfg["sl_pct"]) / 100.0
        be_trig_pct = float(coin_cfg["be_trigger_pct"]) / 100.0
        be_lock_pct = float(coin_cfg["be_lock_pct"]) / 100.0

        if side == "LONG":
            tp_price = round_tick(entry_price * (1.0 + tp_pct), spec.tick_size)
            sl_price = round_tick(entry_price * (1.0 - sl_pct), spec.tick_size)
            be_trigger_price = round_tick(entry_price * (1.0 + be_trig_pct), spec.tick_size)
            be_lock_price = round_tick(entry_price * (1.0 + be_lock_pct), spec.tick_size)
        else:
            tp_price = round_tick(entry_price * (1.0 - tp_pct), spec.tick_size)
            sl_price = round_tick(entry_price * (1.0 + sl_pct), spec.tick_size)
            be_trigger_price = round_tick(entry_price * (1.0 - be_trig_pct), spec.tick_size)
            be_lock_price = round_tick(entry_price * (1.0 - be_lock_pct), spec.tick_size)

        order_id = self._place_entry_with_tpsl(sym, side, qty, entry_price, tp_price, sl_price)
        placed_ms = int(time.time() * 1000)
        bar_expire_ts = ev.bar_ts + (self.ttl_bars + 1) * M15_MS
        time_expire_ts = placed_ms + self.ttl_sec * 1000
        expire_ts = min(bar_expire_ts, time_expire_ts) if bar_expire_ts > placed_ms else time_expire_ts
        init_status = "OPEN" if self.dry_run else "PENDING"

        filters_snap = {
            "adx15": ev.adx15,
            "bbw15": ev.bbw15,
            "atr15_pct": ev.atr15_pct,
            "funding_bps": ev.funding_bps,
            "div4h": ev.div4h_label,
            "rvol": ev.rvol,
        }
        pos_id = self.journal.add_position(
            symbol=sym,
            side=side,
            key_name=ev.key,
            status=init_status,
            order_id=order_id,
            entry_price=entry_price,
            qty=qty,
            margin_usd=round(qty * entry_price / self.leverage, 2),
            leverage=self.leverage,
            tp_price=tp_price,
            sl_price=sl_price,
            be_trigger_price=be_trigger_price,
            be_lock_price=be_lock_price,
            entry_bar_ts=ev.bar_ts,
            expire_ts=expire_ts,
            filters=filters_snap,
        )

        slots_used = len(active) + 1
        badge = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
        mode_tag = " [PAPER]" if self.dry_run else (" [DEMO]" if self.mode == "demo" else "")
        msg = (
            f"🚀 <b>ВХОД В СДЕЛКУ{mode_tag} | {sym} {badge}</b>\n"
            f"Ключ: <code>{ev.key}</code> | Слоты: <b>{slots_used}/{self.max_open}</b>\n"
            f"Вход (Limit): <code>{entry_price}</code>\n"
            f"TP: <code>{tp_price}</code> (+{coin_cfg['tp_pct']}%) | SL: <code>{sl_price}</code> (-{coin_cfg['sl_pct']}%)\n"
            f"BE: при закрытии <code>{be_trigger_price}</code> -> SL <code>{be_lock_price}</code>\n"
            f"Объём: <code>{qty}</code> (~<code>${qty * entry_price:,.1f}</code>) | Маржа: <code>${margin_usd:,.1f}</code> (10% × {self.leverage}x)\n"
            f"Фильтры: ADX15=<code>{ev.adx15:.1f}</code> | BBW15=<code>{ev.bbw15:.2f}%</code> | "
            f"ATR15=<code>{ev.atr15_pct:.2f}%</code> | FR=<code>{ev.funding_bps:+.2f}bps</code> | 4H Div: <code>{ev.div4h_label}</code>"
        )
        logger.info("Opened %s %s (%s) id=%s order=%s", sym, side, ev.key, pos_id, order_id)
        self.notify(msg)
        return pos_id

    # ------------------------------------------------------------------
    # Exchange Sync (every 20 seconds) & 45m (3-bar) Limit Order TTL
    # ------------------------------------------------------------------
    def _is_order_expired(self, pos: Dict[str, Any], now_ms: int) -> bool:
        """Return True if entry limit order reached 45m (2700s) or 3 closed 15m bars."""
        ttl_ms = self.ttl_sec * 1000
        age_ms = now_ms - int(pos.get("opened_at") or 0)
        return (age_ms >= ttl_ms) or (now_ms >= int(pos.get("expire_ts") or 0))

    def _ensure_server_tpsl(self, sym: str, pos: Dict[str, Any], ep_ex: Dict[str, Any]) -> None:
        """Ensure server-side TP and SL remain attached on Bybit for an open position."""
        ex_sl = float(ep_ex.get("stopLoss") or 0.0)
        ex_tp = float(ep_ex.get("takeProfit") or 0.0)
        if ex_sl > 0 and ex_tp > 0:
            return
        try:
            priv = self.client._require_private()
            priv.set_trading_stop(
                category="linear",
                symbol=sym,
                takeProfit=str(pos["tp_price"]),
                stopLoss=str(pos["sl_price"]),
                tpTriggerBy="MarkPrice",
                slTriggerBy="MarkPrice",
                tpslMode="Full",
                positionIdx=0,
            )
            logger.info("Restored missing server-side TP/SL for %s", sym)
        except Exception as exc:
            logger.warning("Failed to restore TP/SL for %s: %s", sym, exc)

    def _notify_ttl_cancelled(
        self,
        pos: Dict[str, Any],
        *,
        partial_fill: bool = False,
        fill_qty: float = 0.0,
        fill_px: float = 0.0,
    ) -> None:
        sym = pos["symbol"]
        badge = "🟢 LONG" if pos["side"] == "LONG" else "🔴 SHORT"
        ttl_min = self.ttl_sec // 60
        if partial_fill:
            msg = (
                f"⌛ <b>Лимитный ордер по {sym} отменён по тайм-ауту {ttl_min}m ({self.ttl_bars} свечи) — остаток</b>\n"
                f"Направление: {badge} | Ключ: <code>{pos['key_name']}</code>\n"
                f"Неисполненный остаток отменён. Открытая часть позиции "
                f"(<code>{fill_qty}</code> @ <code>{fill_px}</code>) оставлена под защитой серверных TP/SL."
            )
        else:
            slots_used = len(self.journal.get_active_positions())
            msg = (
                f"⌛ <b>Лимитный ордер по {sym} отменён по тайм-ауту {ttl_min}m ({self.ttl_bars} свечи)</b>\n"
                f"Направление: {badge} | Ключ: <code>{pos['key_name']}</code>\n"
                f"Цена лимитки: <code>{pos['entry_price']}</code> | Слот освобождён (<b>{slots_used}/{self.max_open}</b>)"
            )
        self.notify(msg)

    def sync_with_exchange(self) -> None:
        """
        20-second sync loop:
          - Full fills & partial fills (preserving TP/SL protection)
          - Auto-cancel unfilled entry limit orders (or unfilled partial remainders) after 45m (3 x 15m bars)
          - Server-side TP/BE/SL exit reconciliation
          - Live wallet balance & -6.0% UTC daily hard stop
        """
        if self.dry_run or not self.client.has_credentials:
            self.check_daily_stop()
            return

        try:
            ex_pos_list = self.client.get_positions()
            ex_ord_list = self.client.get_open_orders()
        except Exception as exc:
            logger.warning("Exchange sync error: %s", exc)
            return

        ex_pos_map = {p["symbol"]: p for p in ex_pos_list}
        ex_ord_map = {str(o.get("orderId", "")): o for o in ex_ord_list if o.get("orderId")}
        self.last_exchange_positions = ex_pos_map
        now_ms = int(time.time() * 1000)

        for pos in self.journal.get_active_positions():
            sym = pos["symbol"]
            pos_id = pos["id"]
            status = pos["status"]
            oid = str(pos.get("order_id") or "")
            has_pos = sym in ex_pos_map
            has_ord = bool(oid and oid in ex_ord_map)
            expired = self._is_order_expired(pos, now_ms)

            if status == "PENDING":
                if has_pos and has_ord:
                    # Partial fill detected: position is open AND remainder limit order is still resting
                    ep_ex = ex_pos_map[sym]
                    fill_px = float(ep_ex.get("avgPrice") or pos["entry_price"])
                    fill_qty = float(ep_ex.get("size") or pos["qty"])
                    fill_margin = round(fill_qty * fill_px / self.leverage, 2)
                    if expired:
                        # Cancel ONLY unfilled remainder; keep open filled portion protected by server-side TP/SL
                        self.client.cancel_order(sym, oid)
                        self.journal.update_open_fill(pos_id, fill_px, fill_qty, fill_margin, clear_order_id=True)
                        self._ensure_server_tpsl(sym, pos, ep_ex)
                        logger.info(
                            "Cancelled expired partial-fill remainder for %s (45m TTL); kept open qty=%s @ %s",
                            sym, fill_qty, fill_px,
                        )
                        self._notify_ttl_cancelled(pos, partial_fill=True, fill_qty=fill_qty, fill_px=fill_px)
                    else:
                        # Transition to OPEN while retaining order_id so remainder is watched until 45m TTL
                        self.journal.update_open_fill(pos_id, fill_px, fill_qty, fill_margin, clear_order_id=False)
                        self._ensure_server_tpsl(sym, pos, ep_ex)
                        logger.info(
                            "Partial fill active on %s: qty=%s @ %s (watching remainder until 45m TTL)",
                            sym, fill_qty, fill_px,
                        )
                elif has_pos and not has_ord:
                    # Fully filled!
                    ep_ex = ex_pos_map[sym]
                    fill_px = float(ep_ex.get("avgPrice") or pos["entry_price"])
                    fill_qty = float(ep_ex.get("size") or pos["qty"])
                    fill_margin = round(fill_qty * fill_px / self.leverage, 2)
                    self.journal.update_open_fill(pos_id, fill_px, fill_qty, fill_margin, clear_order_id=True)
                    self._ensure_server_tpsl(sym, pos, ep_ex)
                    logger.info("Pending order filled on exchange: %s qty=%s @ %s", sym, fill_qty, fill_px)
                elif expired:
                    # Unfilled and >= 45m (3 bars): cancel limit order and verify no last-ms fill occurred
                    if has_ord:
                        self.client.cancel_order(sym, oid)
                    recheck = []
                    try:
                        recheck = self.client.get_positions(sym)
                    except Exception:
                        pass
                    if recheck:
                        ep_ex = recheck[0]
                        fill_px = float(ep_ex.get("avgPrice") or pos["entry_price"])
                        fill_qty = float(ep_ex.get("size") or pos["qty"])
                        fill_margin = round(fill_qty * fill_px / self.leverage, 2)
                        self.journal.update_open_fill(pos_id, fill_px, fill_qty, fill_margin, clear_order_id=True)
                        self._ensure_server_tpsl(sym, pos, ep_ex)
                        logger.info("Order filled just before 45m TTL cancel: %s qty=%s @ %s", sym, fill_qty, fill_px)
                    else:
                        self.journal.mark_cancelled(pos_id, "TTL_EXPIRED")
                        logger.info("Cancelled expired limit entry for %s (45m / 3-bar TTL)", sym)
                        self._notify_ttl_cancelled(pos, partial_fill=False)
                elif not has_ord:
                    # Order disappeared without open position -> cancelled externally or filled & instantly closed
                    self._reconcile_missing_position(pos, now_ms)

            elif status == "OPEN":
                if not has_pos:
                    if has_ord:
                        # Clean up any leftover partial-fill entry order if the filled part already hit TP/SL
                        self.client.cancel_order(sym, oid)
                    self._reconcile_missing_position(pos, now_ms)
                else:
                    ep_ex = ex_pos_map[sym]
                    if oid:
                        fill_px = float(ep_ex.get("avgPrice") or pos["entry_price"])
                        fill_qty = float(ep_ex.get("size") or pos["qty"])
                        fill_margin = round(fill_qty * fill_px / self.leverage, 2)
                        if has_ord:
                            if expired:
                                # Partial fill reached 45m TTL -> cancel ONLY unfilled remainder, keep open part!
                                self.client.cancel_order(sym, oid)
                                self.journal.update_open_fill(pos_id, fill_px, fill_qty, fill_margin, clear_order_id=True)
                                logger.info(
                                    "Cancelled expired partial-fill remainder for OPEN %s (45m TTL); kept qty=%s @ %s",
                                    sym, fill_qty, fill_px,
                                )
                                self._notify_ttl_cancelled(pos, partial_fill=True, fill_qty=fill_qty, fill_px=fill_px)
                            else:
                                self.journal.update_open_fill(pos_id, fill_px, fill_qty, fill_margin, clear_order_id=False)
                        else:
                            # Remainder completed filling!
                            self.journal.update_open_fill(pos_id, fill_px, fill_qty, fill_margin, clear_order_id=True)
                            logger.info("Partial fill completed for %s: final qty=%s @ %s", sym, fill_qty, fill_px)
                    self._ensure_server_tpsl(sym, pos, ep_ex)

        self.check_daily_stop()

    def _reconcile_missing_position(self, pos: Dict[str, Any], now_ms: int) -> None:
        sym = pos["symbol"]
        pos_id = pos["id"]
        opened_at = int(pos["opened_at"])
        pnl_usd = 0.0
        exit_px = float(pos["entry_price"])
        found_closed = False

        try:
            closed_list = self.client.get_closed_pnl(sym, start_ms=max(0, opened_at - 60_000), limit=10)
            for item in closed_list:
                upd = int(item.get("updatedTime") or item.get("createdTime") or 0)
                if upd >= opened_at - 30_000:
                    pnl_usd = float(item.get("closedPnl") or 0.0)
                    exit_px = float(item.get("avgExitPrice") or exit_px)
                    found_closed = True
                    break
        except Exception as exc:
            logger.warning("get_closed_pnl failed for %s: %s", sym, exc)

        if pos["status"] == "PENDING" and not found_closed:
            self.journal.mark_cancelled(pos_id, "CANCELLED_EXTERNAL")
            return

        if pnl_usd > 0:
            reason = "BE" if (pos["be_active"] and abs(exit_px - pos["be_lock_price"]) < abs(exit_px - pos["tp_price"])) else "TP"
        elif pnl_usd < 0:
            reason = "SL"
        else:
            reason = "BE" if pos["be_active"] else "CLOSED"

        margin = max(float(pos["margin_usd"]), 1.0)
        pnl_pct = pnl_usd / margin * 100.0
        self.journal.mark_closed(pos_id, exit_px, reason, round(pnl_usd, 2), round(pnl_pct, 2))
        self.journal.set_cooldown_until_ms(sym, now_ms + self.cooldown_bars * M15_MS)

        ds = self.check_daily_stop()
        icon = "✅" if pnl_usd >= 0 else "❌"
        badge = "🟢 LONG" if pos["side"] == "LONG" else "🔴 SHORT"
        msg = (
            f"{icon} <b>СДЕЛКА ЗАКРЫТА ({reason}) | {sym} {badge}</b>\n"
            f"Ключ: <code>{pos['key_name']}</code>\n"
            f"Вход: <code>{pos['entry_price']}</code> -> Выход: <code>{exit_px}</code>\n"
            f"Результат: <b>{pnl_usd:+,.2f} USDT ({pnl_pct:+.2f}% к марже)</b>\n"
            f"Новый баланс: <code>${ds['current_equity']:,.2f}</code> | "
            f"Дневной PnL: <code>{ds['daily_pnl_usd']:+,.2f} USDT ({ds['daily_pnl_pct']:+.2f}%)</code>"
        )
        logger.info("Closed %s (%s): pnl=%.2f USDT", sym, reason, pnl_usd)
        self.notify(msg)

    # ------------------------------------------------------------------
    # Closed-15m-Bar Breakeven Check (strictly identical to backtest)
    # ------------------------------------------------------------------
    def check_breakeven_on_closed_bar(self, sym: str, closed_bar_ts: int, closed_price: float) -> None:
        """If open position's closed 15m candle passes be_trigger_price, move Bybit SL to be_lock_price."""
        for pos in self.journal.get_open_positions():
            if pos["symbol"] != sym or pos["be_active"]:
                continue
            if closed_bar_ts <= int(pos["entry_bar_ts"]):
                continue

            is_l = pos["side"] == "LONG"
            trig = float(pos["be_trigger_price"])
            lock_px = float(pos["be_lock_price"])
            hit = (closed_price >= trig) if is_l else (closed_price <= trig)
            if not hit:
                continue

            if not self.dry_run and self.client.has_credentials:
                try:
                    self.client.set_trading_stop(sym, lock_px)
                except Exception as exc:
                    logger.error("Failed to move SL to BE for %s: %s", sym, exc)
                    continue

            self.journal.mark_be_active(pos["id"], lock_px)
            badge = "🟢 LONG" if is_l else "🔴 SHORT"
            msg = (
                f"🛡 <b>ПЕРЕВОД В БЕЗУБЫТОК (BE) | {sym} {badge}</b>\n"
                f"Цена закрытия 15m: <code>{closed_price}</code>\n"
                f"Новый серверный SL в плюсе: <code>{lock_px}</code> (вход <code>{pos['entry_price']}</code>)"
            )
            logger.info("Moved %s SL to BE: %s", sym, lock_px)
            self.notify(msg)

    # ------------------------------------------------------------------
    # Panic Close All
    # ------------------------------------------------------------------
    def panic_close_all(self) -> str:
        """Cancel all open orders, market-close all open positions, and pause the bot."""
        self.journal.set_paused(True)
        closed_count = 0
        cancelled_count = 0
        errors: List[str] = []

        if not self.dry_run and self.client.has_credentials:
            for sym in self.coin_cfg_map:
                try:
                    self.client.cancel_all_orders(sym)
                    cancelled_count += 1
                except Exception:
                    pass
            try:
                for p in self.client.get_positions():
                    sym = p["symbol"]
                    sz = float(p.get("size") or 0.0)
                    if sz <= 0:
                        continue
                    close_side = "Sell" if p.get("side") == "Buy" else "Buy"
                    try:
                        self.client.close_position_market(sym, close_side, sz)
                        closed_count += 1
                    except Exception as exc:
                        errors.append(f"{sym}: {exc}")
            except Exception as exc:
                errors.append(f"get_positions: {exc}")

        for pos in self.journal.get_active_positions():
            if pos["status"] == "PENDING":
                self.journal.mark_cancelled(pos["id"], "PANIC_CLOSE")
            else:
                self.journal.mark_closed(pos["id"], float(pos["entry_price"]), "PANIC_CLOSE", 0.0, 0.0)

        time.sleep(1.0)
        eq, _ = self.refresh_balance()
        summary = (
            f"🛑 <b>PANIC CLOSE ВЫПОЛНЕН</b>\n"
            f"Закрыто позиций: <b>{closed_count}</b>\n"
            f"Бот переведён в режим: <b>⏸ ПАУЗА</b>\n"
            f"Актуальный баланс: <code>${eq:,.2f}</code>"
        )
        if errors:
            summary += "\n⚠️ Ошибки: " + "; ".join(errors)
        logger.warning("PANIC CLOSE executed: closed=%d errors=%s", closed_count, errors)
        return summary
