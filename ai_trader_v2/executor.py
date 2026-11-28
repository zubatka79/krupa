"""
Bybit v5 Execution & 20-second Exchange Sync Engine for L2_news (Certified Step 17).

Features:
  - Isolated Margin + 15x leverage across All-Thick-18 (18 LONG + 18 SHORT)
  - 5x12% slots (max_open=7, max_same_direction=5, max_same_key=2, AntiSalvo <=2 LONG / <=2 SHORT per bar)
  - Margin cap $5,000 ($2,500 in VolStorm 1D ATR > 4%), strict 100% Bybit margin reserve <= 85%
  - 2-Step Limit DCA (35/65 normal: -1.5% DCA / -4.0% SL / +1.5% TP; 30/70 storm: -2.0% / -5.0% / +1.6%)
  - Atomic server-side SL attached directly in Leg 1 Market IOC entry + resting Leg 2 Limit GTC DCA + Limit Maker TP
  - Closed-candle BE +0.30% (trigger +1.00%): moves SL to +0.30%, synchronously cancels pending Leg 2 DCA limit,
    and frees its reserved margin.
  - Strict Hard SL vs BE separation: AC 6h pause (per side) & Coin Probation (2 Hard SL / 21d -> 14d) trigger
    ONLY on real Hard SL (be_active=0), NEVER on BE exits!
  - US Macro Calendar Guard [T-60m .. T+45m] blocking ONLY new entries.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime
from ai_trader_v2.macro_calendar import MacroCalendarGuard, MacroCalendarStatus
from ai_trader_v2.vika_engine import CoinEvaluation, M15_MS
from bot.bybit_client import BybitClient, InstrumentSpec, fmt_num, round_step_down, round_tick

logger = logging.getLogger("vika.executor")


def _is_entry_limit_order(order: Dict[str, Any]) -> bool:
    """Return True strictly for opening/DCA Limit orders (never TP/SL or reduceOnly)."""
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


def _is_reduce_only_limit_tp(order: Dict[str, Any], symbol: str, close_side: str) -> bool:
    """Return True if order is an active reduceOnly Limit TP order for `symbol`."""
    if str(order.get("symbol", "")) != symbol:
        return False
    if str(order.get("side", "")) != close_side:
        return False
    if str(order.get("orderType", "")).lower() != "limit":
        return False
    reduce_only = order.get("reduceOnly")
    if not (reduce_only is True or str(reduce_only).lower() == "true"):
        return False
    stop_type = str(order.get("stopOrderType", "") or "").strip()
    if stop_type and stop_type.upper() != "UNKNOWN":
        return False
    return True


def _collect_all_symbols(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Collect unique symbols from long_coins, short_coins, and legacy coins list."""
    out: Dict[str, Dict[str, Any]] = {}
    for bucket in ("coins", "short_coins", "long_coins"):
        for c in cfg.get(bucket, []):
            sym = str(c["symbol"])
            if sym not in out:
                out[sym] = dict(c)
    return out


class VikaExecutor:
    def __init__(
        self,
        cfg: Dict[str, Any],
        client: BybitClient,
        journal: VikaJournal,
        notifier: Optional[Callable[[str], None]] = None,
        macro_guard: Optional[MacroCalendarGuard] = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.journal = journal
        self.notify = notifier or (lambda msg: None)
        self.macro_guard = macro_guard

        self.strategy_name = str(cfg.get("strategy_name", "Vika Hybrid"))
        self.mode = str(cfg.get("mode", "demo")).lower()
        self.dry_run = bool(cfg.get("dry_run", False)) or (self.mode == "paper")
        self.allowed_directions = str(cfg.get("allowed_directions", "long_short")).strip().lower()
        self.margin_mode = str(cfg.get("margin_mode", "ISOLATED")).strip().upper()
        self.leverage = int(cfg.get("leverage", 15))
        self.entry_order_type = str(cfg.get("entry_order_type", "Market")).strip().capitalize()
        self.tp_order_type = str(cfg.get("tp_order_type", "Limit")).strip().capitalize()

        self.margin_pct = float(cfg.get("margin_pct_per_trade", 12.0)) / 100.0
        self.max_margin_usd = float(cfg.get("max_margin_usd", 5000.0))
        self.storm_max_margin_usd = float(cfg.get("storm_max_margin_usd", 2500.0))
        self.max_open = int(cfg.get("max_open_positions", 7))
        self.max_same_dir = int(cfg.get("max_same_direction", 5))
        self.max_same_key = int(cfg.get("max_same_key", 2))
        self.max_entries_per_bar_l = int(cfg.get("max_entries_per_bar_long", cfg.get("max_entries_per_bar", 2)))
        self.max_entries_per_bar_s = int(cfg.get("max_entries_per_bar_short", 2))
        self.max_total_margin_pct = float(cfg.get("max_total_margin_pct", 85.0)) / 100.0
        self.margin_fee_reserve_mult = float(cfg.get("margin_fee_reserve_mult", 1.01))
        self.daily_loss_limit_pct = float(cfg.get("daily_loss_limit_pct", 6.0))

        self.cooldown_bars_long = int(cfg.get("cooldown_bars_long", 3))
        self.cooldown_bars_short = int(cfg.get("cooldown_bars_short", cfg.get("cooldown_bars_after_close", 4)))
        self.cooldown_bars = self.cooldown_bars_short
        self.ac_pause_hours = int(cfg.get("ac_pause_hours", 6))
        self.probation_sl_count = int(cfg.get("probation_sl_count", 2))
        self.probation_window_days = int(cfg.get("probation_window_days", 21))
        self.probation_pause_days = int(cfg.get("probation_pause_days", 14))
        self.ttl_bars = int(cfg.get("limit_entry_ttl_bars", 3))
        self.ttl_sec = int(cfg.get("limit_entry_ttl_sec", 2700))

        dca_cfg = cfg.get("dca", {}) or {}
        self.dca_enabled = bool(dca_cfg.get("enabled", True))
        self.dca_leg1_pct = float(dca_cfg.get("leg1_pct", 35.0)) / 100.0
        self.dca_step_pct = float(dca_cfg.get("step_pct", 1.5)) / 100.0
        self.dca_sl_pct = float(dca_cfg.get("sl_pct", 4.0)) / 100.0
        self.dca_tp_pct = float(dca_cfg.get("tp_pct", 1.5)) / 100.0
        self.be_trigger_pct = float(dca_cfg.get("be_trigger_pct", 1.0)) / 100.0
        self.be_lock_pct = float(dca_cfg.get("be_lock_pct", 0.30)) / 100.0

        self.storm_size_mult = float(dca_cfg.get("storm_size_mult", 0.50))
        self.storm_leg1_pct = float(dca_cfg.get("storm_leg1_pct", 30.0)) / 100.0
        self.storm_step_pct = float(dca_cfg.get("storm_step_pct", 2.0)) / 100.0
        self.storm_sl_pct = float(dca_cfg.get("storm_sl_pct", 5.0)) / 100.0
        self.storm_tp_pct = float(dca_cfg.get("storm_tp_pct", 1.6)) / 100.0

        self.coin_cfg_map: Dict[str, Dict[str, Any]] = _collect_all_symbols(cfg)
        self.specs: Dict[str, InstrumentSpec] = {}
        self._lock = threading.RLock()
        self.last_wallet_balance: float = 2000.0
        self.last_equity: float = 2000.0
        self.last_available: float = 2000.0
        self.last_exchange_positions: Dict[str, Dict[str, Any]] = {}
        self.last_btc_regime: Optional[BtcL2Regime] = None
        self.stale_warnings: List[str] = []
        self.startup_cancelled_orders: List[str] = []

        self._bar_entry_counts: Dict[Tuple[int, str], int] = {}
        self._last_margin_warn_ts: float = 0.0
        self._last_news_notified_iso: str = ""
        self._last_horizon_warn_ts: float = 0.0

    # ------------------------------------------------------------------
    # Startup Setup: Instruments, Isolated 15x Leverage, Stale Check
    # ------------------------------------------------------------------
    def load_instruments(self) -> Dict[str, InstrumentSpec]:
        with self._lock:
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

    def ensure_margin_and_leverage(self) -> Dict[str, str]:
        """Enforce Isolated Margin (tradeMode=1) and configured leverage (15x) for all pool symbols."""
        with self._lock:
            results: Dict[str, str] = {}
            mode_lbl = "Isolated" if self.margin_mode == "ISOLATED" else "Cross"
            trade_mode = 1 if self.margin_mode == "ISOLATED" else 0
            if self.dry_run or not self.client.has_credentials:
                for sym in self.coin_cfg_map:
                    results[sym] = f"DRY_RUN ({self.leverage}x {mode_lbl})"
                return results

            priv = self.client._require_private()
            if self.margin_mode == "ISOLATED":
                try:
                    resp = priv.set_margin_mode(setMarginMode="ISOLATED_MARGIN")
                    rc = resp.get("retCode")
                    if rc not in (0, 110026, 100028, 3400045):
                        logger.info("set_margin_mode(ISOLATED_MARGIN) response: %s", resp)
                except Exception as exc:
                    logger.info("UTA set_margin_mode(ISOLATED_MARGIN) note: %s", exc)
            else:
                try:
                    priv.set_margin_mode(setMarginMode="REGULAR_MARGIN")
                except Exception:
                    pass

            lev_str = str(self.leverage)
            for sym in self.coin_cfg_map:
                try:
                    try:
                        priv.switch_margin_mode(
                            category="linear",
                            symbol=sym,
                            tradeMode=trade_mode,
                            buyLeverage=lev_str,
                            sellLeverage=lev_str,
                        )
                    except Exception:
                        pass
                    self.client.set_leverage(sym, self.leverage)
                    results[sym] = f"OK ({self.leverage}x {mode_lbl})"
                except Exception as exc:
                    results[sym] = f"ERR ({exc})"
                    logger.error("Failed %dx %s setup for %s: %s", self.leverage, mode_lbl, sym, exc)
            return results

    # Backward-compatible alias
    ensure_cross_margin_and_leverage = ensure_margin_and_leverage

    def check_preexisting_positions(self) -> List[str]:
        with self._lock:
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

    def _calc_levels_for(
        self, sym: str, side: str, base_px: float, *, in_storm: bool, tp_pct: Optional[float] = None
    ) -> Tuple[float, float, float, float, float]:
        spec = self.specs.get(sym)
        tick = spec.tick_size if spec else 0.0001
        if self.dca_enabled:
            dca_step = self.storm_step_pct if in_storm else self.dca_step_pct
            sl_pct = self.storm_sl_pct if in_storm else self.dca_sl_pct
            eff_tp = tp_pct if tp_pct is not None else (self.storm_tp_pct if in_storm else self.dca_tp_pct)
        else:
            dca_step = 0.0
            sl_pct = self.dca_sl_pct
            eff_tp = tp_pct if tp_pct is not None else self.dca_tp_pct
        if side == "LONG":
            return (
                round_tick(base_px * (1.0 + eff_tp), tick),
                round_tick(base_px * (1.0 - sl_pct), tick),
                round_tick(base_px * (1.0 + self.be_trigger_pct), tick),
                round_tick(base_px * (1.0 + self.be_lock_pct), tick),
                round_tick(base_px * (1.0 - dca_step), tick) if dca_step > 0 else 0.0,
            )
        return (
            round_tick(base_px * (1.0 - eff_tp), tick),
            round_tick(base_px * (1.0 + sl_pct), tick),
            round_tick(base_px * (1.0 - self.be_trigger_pct), tick),
            round_tick(base_px * (1.0 - self.be_lock_pct), tick),
            round_tick(base_px * (1.0 + dca_step), tick) if dca_step > 0 else 0.0,
        )

    def recover_pending_open_positions(
        self,
        ex_pos_list: Optional[List[Dict[str, Any]]] = None,
        ex_ord_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Recover any PENDING_OPEN drafts left by a crash/restart mid-entry before stale cleanup runs."""
        with self._lock:
            pending_opens = self.journal.get_pending_open_positions()
            if not pending_opens or self.dry_run or not self.client.has_credentials:
                return
            try:
                if ex_pos_list is None:
                    ex_pos_list = self.client.get_positions()
                if ex_ord_list is None:
                    ex_ord_list = self.client.get_open_orders()
            except Exception as exc:
                logger.warning("Failed recovering PENDING_OPEN positions: %s", exc)
                return

            ex_pos_map = {p["symbol"]: p for p in ex_pos_list}
            now_ms = int(time.time() * 1000)

            for pos in pending_opens:
                sym = pos["symbol"]
                side = pos["side"]
                bybit_side = "Buy" if side == "LONG" else "Sell"
                if sym in ex_pos_map:
                    ep_ex = ex_pos_map[sym]
                    entry_price = float(ep_ex.get("avgPrice") or pos["entry_price"])
                    fill_qty = float(ep_ex.get("size") or pos["qty"])
                    in_storm = bool(pos.get("in_storm"))
                    tp_pct = float(pos.get("tp_pct") or 1.5) / 100.0
                    tp_price, sl_price, be_trig_px, be_lock_px, dca_px = self._calc_levels_for(
                        sym, side, entry_price, in_storm=in_storm, tp_pct=tp_pct
                    )
                    if float(ep_ex.get("stopLoss") or 0.0) <= 0:
                        try:
                            self.client.set_trading_stop(sym, sl_price)
                        except Exception as exc:
                            logger.warning("Recovery SL restore failed for %s: %s", sym, exc)
                    else:
                        sl_price = float(ep_ex.get("stopLoss") or sl_price)

                    self._ensure_server_tpsl(
                        sym,
                        {**pos, "tp_price": tp_price, "sl_price": sl_price, "qty": fill_qty},
                        ep_ex,
                        ex_ord_list,
                    )

                    matching_dca = [
                        o for o in ex_ord_list
                        if str(o.get("symbol", "")) == sym
                        and str(o.get("side", "")) == bybit_side
                        and _is_entry_limit_order(o)
                    ]
                    orig_leg1_qty = float(pos["qty"])
                    planned_dca_qty = float(pos.get("dca_qty") or 0.0)
                    dca_oid = ""
                    dca_qty = planned_dca_qty
                    dca_status = "NONE"
                    dca_filled_qty = 0.0

                    if matching_dca:
                        ord_d = matching_dca[0]
                        dca_oid = str(ord_d.get("orderId") or "")
                        dca_px = float(ord_d.get("price") or dca_px)
                        dca_qty = float(ord_d.get("qty") or planned_dca_qty)
                        dca_status = "PENDING"
                        dca_filled_qty = max(0.0, fill_qty - orig_leg1_qty)
                    elif fill_qty > orig_leg1_qty * 1.15:
                        dca_status = "FILLED"
                        dca_filled_qty = max(0.0, fill_qty - orig_leg1_qty)
                        dca_qty = planned_dca_qty if planned_dca_qty > 0 else dca_filled_qty
                    elif self.dca_enabled and planned_dca_qty > 0 and dca_px > 0:
                        try:
                            dca_oid = self._place_dca_limit_order(sym, side, planned_dca_qty, dca_px)
                            dca_status = "PENDING"
                        except Exception as exc:
                            logger.error("Recovery failed placing Leg-2 DCA for %s: %s", sym, exc)

                    actual_m = round(fill_qty * entry_price / self.leverage, 2)
                    rem_dca = max(0.0, dca_qty - dca_filled_qty) if dca_status == "PENDING" else 0.0
                    dca_m = round(dca_qty * dca_px / self.leverage, 2) if dca_qty > 0 and dca_px > 0 else 0.0
                    res_total = round(actual_m + (rem_dca * dca_px / self.leverage if rem_dca > 0 and dca_px > 0 else 0.0), 2)

                    self.journal.finalize_pending_open(
                        pos["id"],
                        order_id=str(pos.get("order_id") or ""),
                        entry_price=entry_price,
                        qty=fill_qty,
                        margin_usd=actual_m,
                        reserved_margin_usd=res_total,
                        tp_price=tp_price,
                        sl_price=sl_price,
                        be_trigger_price=be_trig_px,
                        be_lock_price=be_lock_px,
                        dca_order_id=dca_oid,
                        dca_price=dca_px if dca_qty > 0 else 0.0,
                        dca_qty=dca_qty,
                        dca_margin_usd=dca_m,
                        dca_status=dca_status,
                        dca_filled_qty=dca_filled_qty,
                    )
                    logger.info(
                        "Recovered PENDING_OPEN %s %s id=%s -> OPEN (qty=%s dca=%s/%s)",
                        sym, side, pos["id"], fill_qty, dca_status, dca_oid,
                    )
                else:
                    for o in ex_ord_list:
                        if str(o.get("symbol", "")) == sym and o.get("orderId"):
                            try:
                                self.client.cancel_order(sym, str(o["orderId"]))
                            except Exception as exc:
                                logger.warning("Recovery cancel order failed for %s: %s", sym, exc)
                    self._reconcile_missing_position(pos, now_ms)

    def cleanup_stale_entry_orders(self) -> List[str]:
        """On startup, cancel ONLY untracked stale Limit orders (>45m); preserve active L2_news DCA limits!"""
        with self._lock:
            cancelled: List[str] = []
            self.startup_cancelled_orders = []
            if self.dry_run or not self.client.has_credentials:
                return cancelled

            now_ms = int(time.time() * 1000)
            ttl_ms = self.ttl_sec * 1000
            try:
                ex_ord_list = self.client.get_open_orders()
            except Exception as exc:
                logger.warning("Startup stale-order check failed: %s", exc)
                return cancelled

            self.recover_pending_open_positions(ex_ord_list=ex_ord_list)

            active_journal = self.journal.get_active_positions()
            active_dca_oids = {str(p.get("dca_order_id") or "") for p in active_journal if p.get("dca_order_id")}
            pending_open_syms = {str(p["symbol"]) for p in active_journal if p.get("status") == "PENDING_OPEN"}

            for o in ex_ord_list:
                if not _is_entry_limit_order(o):
                    continue
                sym = str(o.get("symbol", ""))
                oid = str(o.get("orderId", ""))
                if not sym or not oid or oid in active_dca_oids or sym in pending_open_syms:
                    continue  # Never cancel an active L2_news Leg-2 DCA limit or PENDING_OPEN entry!
                created_ms = int(o.get("createdTime") or 0)
                age_ms = (now_ms - created_ms) if created_ms > 0 else (ttl_ms + 1)
                if age_ms >= ttl_ms:
                    try:
                        res_cancel = self.client.cancel_order(sym, oid)
                    except Exception as exc:
                        logger.warning("Startup cancel stale order failed for %s (%s): %s", sym, oid, exc)
                        continue
                    if res_cancel is False:
                        continue
                    age_min = max(1, int(age_ms // 60_000))
                    note = f"{sym} ({o.get('side')} Limit @ {o.get('price')}, возраст {age_min}m)"
                    cancelled.append(note)
                    logger.warning("Startup cancelled stale entry limit order: %s [orderId=%s]", note, oid)

            self.startup_cancelled_orders = cancelled
            return cancelled

    # ------------------------------------------------------------------
    # Live Balance, Margin Reserve (<= 85%), Macro Guard & Daily Stop
    # ------------------------------------------------------------------
    def refresh_balance(self) -> Tuple[float, float]:
        with self._lock:
            res_m = sum(float(p.get("reserved_margin_usd") or p["margin_usd"]) for p in self.journal.get_active_positions())
            if self.dry_run or not self.client.has_credentials:
                stats = self.journal.get_total_realized_stats()
                wb = max(50.0, 2000.0 + float(stats["pnl_usd"]))
                unr = sum(float(p.get("unrealisedPnl") or 0.0) for p in self.last_exchange_positions.values())
                self.last_wallet_balance = wb
                self.last_equity = max(0.0, wb + unr)
                self.last_available = max(0.0, wb + min(0.0, unr) - res_m)
                return self.last_equity, self.last_available

            bal = self.client.get_wallet_balance("USDT")
            wb = float(bal.get("wallet_balance", bal["equity"]))
            tot_eq = float(bal.get("total_equity", bal["equity"]))
            self.last_wallet_balance = wb
            self.last_equity = tot_eq
            raw_avail = float(bal["available"])
            free_base_est = min(wb, tot_eq)
            self.last_available = min(raw_avail, max(0.0, free_base_est - res_m))
            return self.last_equity, self.last_available

    def get_margin_reserve_snapshot(self) -> Dict[str, Any]:
        """Compute honest Bybit 100% initial margin reserve ratio (live positions + resting Leg-2 DCA limits)."""
        with self._lock:
            active = self.journal.get_active_positions()
            wb = max(float(self.last_wallet_balance), 0.0)
            tot_eq = max(float(self.last_equity), 0.0)
            unr_pos = sum(float(p.get("unrealisedPnl") or 0.0) for p in self.last_exchange_positions.values())
            unr_bal = tot_eq - wb
            unr_open = unr_pos if abs(unr_pos) > 1e-9 else unr_bal
            free_base = max(0.0, wb + min(0.0, unr_open))
            reserved_usd = sum(float(p.get("reserved_margin_usd") or p["margin_usd"]) for p in active)
            live_margin_usd = sum(float(p["margin_usd"]) for p in active)
            reserve_pct = (reserved_usd / free_base * 100.0) if free_base > 0 else 100.0
            in_storm = bool(self.last_btc_regime.in_storm) if self.last_btc_regime else False
            eff_cap = self.storm_max_margin_usd if in_storm else self.max_margin_usd
            n_long = sum(1 for p in active if p["side"] == "LONG")
            n_short = sum(1 for p in active if p["side"] == "SHORT")
            dca_pending = sum(1 for p in active if p.get("dca_status") == "PENDING")
            dca_filled = sum(1 for p in active if p.get("dca_status") == "FILLED")
            return {
                "wallet_balance": round(wb, 2),
                "total_equity": round(tot_eq, 2),
                "equity": round(tot_eq, 2),
                "unr_open": round(unr_open, 2),
                "free_base": round(free_base, 2),
                "reserved_usd": round(reserved_usd, 2),
                "live_margin_usd": round(live_margin_usd, 2),
                "reserve_pct": round(reserve_pct, 1),
                "limit_pct": round(self.max_total_margin_pct * 100.0, 1),
                "effective_cap_usd": eff_cap,
                "in_storm": in_storm,
                "open_total": len(active),
                "open_long": n_long,
                "open_short": n_short,
                "dca_pending": dca_pending,
                "dca_filled": dca_filled,
            }

    def check_macro_calendar(self, *, notify_events: bool = True) -> Optional[MacroCalendarStatus]:
        with self._lock:
            if self.macro_guard is None:
                return None
            st = self.macro_guard.evaluate()
            now_mono = time.monotonic()
            if st.horizon_warning and (now_mono - self._last_horizon_warn_ts) >= 12 * 3600:
                self._last_horizon_warn_ts = now_mono
                logger.warning(
                    "Macro calendar horizon warning: %.2f days remaining (threshold %.1fd)!",
                    st.horizon_days, self.macro_guard.horizon_warn_days,
                )
                if notify_events:
                    self.notify(
                        f"⚠️ <b>КАЛЕНДАРЬ НОВОСТЕЙ: МАЛЫЙ ГОРИЗОНТ ({st.horizon_days:.1f} дн.)</b>\n"
                        f"До конца <code>config/macro_events_utc.json</code> осталось меньше 3 дней.\n"
                        f"Запусти <code>python update_macro_calendar.py</code> или проверь cron."
                    )
            if st.in_blackout and st.active_event is not None:
                if notify_events and self._last_news_notified_iso != st.active_event.iso_utc:
                    self._last_news_notified_iso = st.active_event.iso_utc
                    end_s = st.blackout_end_utc.strftime("%H:%M UTC") if st.blackout_end_utc else "?"
                    self.notify(
                        f"📅 <b>МАКРО-ПАУЗА [{st.active_event.event_type}] | Входы заблокированы</b>\n"
                        f"Событие: <b>{st.active_event.title}</b>\n"
                        f"Релиз: <code>{st.active_event.iso_utc}</code> | Окно: <code>[-{st.pre_minutes}m .. +{st.post_minutes}m]</code>\n"
                        f"Разблокировка входов: <b>{end_s}</b> (открытые позиции, DCA и BE работают штатно)."
                    )
            return st

    def check_daily_stop(self) -> Dict[str, Any]:
        with self._lock:
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
                    f"Баланс на 00:00 UTC: <code>${start_eq:,.2f}</code> → Текущий: <code>${eq:,.2f}</code>\n"
                    f"Дневной PnL: <b>{daily_pnl_usd:+,.2f} USDT ({daily_pnl_pct:+.2f}%)</b>\n"
                    f"Новые входы заблокированы до 00:00 UTC. Защитные SL/TP сохранены."
                )
                logger.warning("Daily loss limit hit: %.2f%%", daily_pnl_pct)
                self.notify(msg)

            ds["current_equity"] = eq
            ds["daily_pnl_usd"] = round(daily_pnl_usd, 2)
            ds["daily_pnl_pct"] = round(daily_pnl_pct, 2)
            return ds

    # ------------------------------------------------------------------
    # Order Placement: Leg 1 Market IOC + Atomic SL + Leg 2 Limit DCA + Limit Maker TP
    # ------------------------------------------------------------------
    def _place_market_entry_with_sl(
        self, symbol: str, side: str, qty: float, sl_price: float
    ) -> str:
        """Open Leg 1 immediately via Market IOC with atomic server-side Market Stop-Loss."""
        if self.dry_run:
            return f"PAPER-MKT-{symbol}-{int(time.time() * 1000)}"
        priv = self.client._require_private()
        bybit_side = "Buy" if side == "LONG" else "Sell"
        resp = priv.place_order(
            category="linear",
            symbol=symbol,
            side=bybit_side,
            orderType="Market",
            qty=fmt_num(qty),
            stopLoss=fmt_num(sl_price),
            slTriggerBy="MarkPrice",
            tpslMode="Full",
            timeInForce="IOC",
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit Market place_order failed for {symbol}: {resp}")
        return str(resp["result"].get("orderId", ""))

    def _place_dca_limit_order(self, symbol: str, side: str, qty: float, price: float) -> str:
        """Place resting Leg 2 Limit GTC DCA order (Maker 0.02%, reduceOnly=False)."""
        if self.dry_run or not self.client.has_credentials:
            return f"PAPER-DCA-{symbol}-{int(time.time() * 1000)}"
        priv = self.client._require_private()
        bybit_side = "Buy" if side == "LONG" else "Sell"
        resp = priv.place_order(
            category="linear",
            symbol=symbol,
            side=bybit_side,
            orderType="Limit",
            qty=fmt_num(qty),
            price=fmt_num(price),
            timeInForce="GTC",
            reduceOnly=False,
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(f"Bybit Leg-2 DCA Limit failed for {symbol}: {resp}")
        oid = str(resp["result"].get("orderId", ""))
        logger.info("Placed Leg-2 DCA Limit %s %s qty=%s @ %s id=%s", symbol, bybit_side, qty, price, oid)
        return oid

    def _place_limit_tp(self, symbol: str, side: str, qty: float, tp_price: float) -> Optional[str]:
        """Place resting reduceOnly Limit GTC Take-Profit order in the book (Maker 0.02%)."""
        if self.dry_run or not self.client.has_credentials:
            return f"PAPER-TP-{symbol}"
        close_side = "Sell" if side == "LONG" else "Buy"
        try:
            res = self.client.place_limit_tp_order(symbol, close_side, qty, tp_price)
            oid = str(res.get("orderId", ""))
            logger.info("Placed Limit Maker TP for %s (%s qty=%s @ %s) id=%s", symbol, close_side, qty, tp_price, oid)
            return oid
        except Exception as exc:
            logger.error("Failed placing Limit Maker TP for %s @ %s: %s (falling back to server TP)", symbol, tp_price, exc)
            try:
                priv = self.client._require_private()
                priv.set_trading_stop(
                    category="linear",
                    symbol=symbol,
                    takeProfit=fmt_num(tp_price),
                    tpTriggerBy="MarkPrice",
                    tpslMode="Full",
                    positionIdx=0,
                )
            except Exception as exc2:
                logger.error("Fallback set_trading_stop TP also failed for %s: %s", symbol, exc2)
            return None

    def _cancel_existing_limit_tps(self, symbol: str, side: str, ex_ord_list: Optional[List[Dict[str, Any]]] = None) -> None:
        if self.dry_run or not self.client.has_credentials:
            return
        close_side = "Sell" if side == "LONG" else "Buy"
        orders = ex_ord_list if ex_ord_list is not None else self.client.get_open_orders(symbol)
        for o in orders:
            if _is_reduce_only_limit_tp(o, symbol, close_side) and o.get("orderId"):
                try:
                    self.client.cancel_order(symbol, str(o["orderId"]))
                except Exception as exc:
                    logger.warning("Failed cancelling old Limit TP %s for %s: %s", o.get("orderId"), symbol, exc)

    # ------------------------------------------------------------------
    # Open Signal (L2_news: 5x12%, 15x, Cap $5k/$2.5k, AntiSalvo, Macro, 2-Step DCA)
    # ------------------------------------------------------------------
    def open_signal(
        self,
        ev: Any,
        *,
        btc_regime: Optional[BtcL2Regime] = None,
    ) -> Optional[int]:
        with self._lock:
            sym = str(ev.symbol)
            side = str(ev.final_signal or "")
            if side not in ("LONG", "SHORT"):
                return None

            if side == "LONG" and self.allowed_directions not in ("long_short", "only_long"):
                return None
            if side == "SHORT" and self.allowed_directions not in ("long_short", "only_short"):
                return None

            regime = btc_regime or self.last_btc_regime
            if regime is not None:
                if side == "LONG" and not regime.allow_long:
                    logger.info("Skip %s LONG: BTC 4H+1D regime blocks LONG (%s)", sym, regime.mode_label)
                    return None
                if side == "SHORT" and not regime.allow_short:
                    logger.info("Skip %s SHORT: BTC 4H+1D regime blocks SHORT (%s)", sym, regime.mode_label)
                    return None

            # Macro calendar blackout [T-60m .. T+45m]
            macro_st = self.check_macro_calendar(notify_events=True)
            if macro_st is not None and macro_st.in_blackout:
                ev_t = macro_st.active_event.event_type if macro_st.active_event else "MACRO"
                logger.info("Skip %s %s: US macro blackout active [%s]", sym, side, ev_t)
                return None

            if self.journal.get_last_signal_bar_ts(sym, side) >= int(ev.bar_ts):
                return None
            if self.journal.is_paused():
                logger.info("Skip %s %s: bot is PAUSED", sym, side)
                return None

            ds = self.check_daily_stop()
            if ds["halted"]:
                logger.info("Skip %s %s: daily -%.1f%% stop active", sym, side, self.daily_loss_limit_pct)
                return None

            now_ms = int(time.time() * 1000)
            if now_ms < self.journal.get_ac_pause_until_ms(side):
                logger.info("Skip %s %s: AC %dh pause active for %s", sym, side, self.ac_pause_hours, side)
                return None
            if now_ms < self.journal.get_coin_probation_until_ms(sym):
                logger.info("Skip %s %s: symbol in Coin Probation (%dd quarantine)", sym, side, self.probation_pause_days)
                return None
            if now_ms < self.journal.get_cooldown_until_ms(sym, side):
                logger.info("Skip %s %s: symbol in post-close cooldown", sym, side)
                return None

            # AntiSalvo: max 2 LONG / max 2 SHORT per closed 15m bar
            bar_key = (int(ev.bar_ts), side)
            bar_lim = self.max_entries_per_bar_l if side == "LONG" else self.max_entries_per_bar_s
            used_bar = self._bar_entry_counts.get(bar_key, 0)
            if bar_lim > 0 and used_bar >= bar_lim:
                logger.info("Skip %s %s: AntiSalvo max %d/%d entries per bar reached", sym, side, used_bar, bar_lim)
                return None

            active = self.journal.get_active_positions()
            if any(p["symbol"] == sym for p in active):
                logger.info("Skip %s %s: symbol already active", sym, side)
                return None
            if len(active) >= self.max_open:
                logger.info("Skip %s %s: max total slots %d/%d reached", sym, side, len(active), self.max_open)
                return None
            same_dir_cnt = sum(1 for p in active if p["side"] == side)
            if same_dir_cnt >= self.max_same_dir:
                logger.info("Skip %s %s: AntiSalvo max_same_direction %d/%d reached", sym, side, same_dir_cnt, self.max_same_dir)
                return None
            same_key_cnt = sum(1 for p in active if p["side"] == side and p["key_name"] == ev.key)
            if same_key_cnt >= self.max_same_key:
                logger.info("Skip %s %s: max_same_key %d/%d reached for %s", sym, side, same_key_cnt, self.max_same_key, ev.key)
                return None

            wb = self.last_wallet_balance
            if wb < 50.0 or self.last_equity < 50.0:
                logger.warning("Skip %s %s: balance too low (wb=$%.2f eq=$%.2f)", sym, side, wb, self.last_equity)
                return None

            in_storm = bool(regime.in_storm) if regime is not None else False
            s_mult = self.storm_size_mult if in_storm else 1.0
            eff_cap = self.storm_max_margin_usd if in_storm else self.max_margin_usd
            full_margin = min(wb * self.margin_pct * s_mult, eff_cap)
            if full_margin < 10.0:
                return None

            # STRICT 100% Bybit Initial Margin Reserve Check (live positions + resting Leg-2 DCA limits <= 85% free_base)
            snap = self.get_margin_reserve_snapshot()
            free_base = snap["free_base"]
            reserved_now = snap["reserved_usd"]
            projected_pct = ((reserved_now + full_margin) * self.margin_fee_reserve_mult / max(free_base, 1.0)) * 100.0
            if (reserved_now + full_margin) * self.margin_fee_reserve_mult > free_base * self.max_total_margin_pct:
                logger.warning(
                    "Skip %s %s: Bybit margin reserve %.1f%% > %.1f%% (reserved=$%.1f + new=$%.1f, free=$%.1f)",
                    sym, side, projected_pct, self.max_total_margin_pct * 100.0, reserved_now, full_margin, free_base,
                )
                now_mono = time.monotonic()
                if (now_mono - self._last_margin_warn_ts) >= 1800:
                    self._last_margin_warn_ts = now_mono
                    self.notify(
                        f"⚠️ <b>ЛИМИТ РЕЗЕРВА МАРЖИ BYBIT ({self.max_total_margin_pct*100:.0f}%)</b>\n"
                        f"Вход <code>{sym} {side}</code> отклонён: резерв достиг бы <b>{projected_pct:.1f}%</b>\n"
                        f"Зарезервировано: <code>${reserved_now:,.1f}</code> | Свободная база: <code>${free_base:,.1f}</code>"
                    )
                return None

            spec = self.specs[sym]
            ref_price = round_tick(float(ev.close_price), spec.tick_size)

            if self.dca_enabled:
                leg1_frac = self.storm_leg1_pct if in_storm else self.dca_leg1_pct
                dca_step = self.storm_step_pct if in_storm else self.dca_step_pct
                sl_pct = self.storm_sl_pct if in_storm else self.dca_sl_pct
                tp_pct = self.storm_tp_pct if in_storm else self.dca_tp_pct
            else:
                leg1_frac = 1.0
                dca_step = 0.0
                sl_pct = self.dca_sl_pct
                tp_pct = self.dca_tp_pct

            m1 = full_margin * leg1_frac
            m2 = full_margin * (1.0 - leg1_frac)
            qty1 = round_step_down((m1 * self.leverage) / ref_price, spec.qty_step)
            if qty1 < spec.min_order_qty or (qty1 * ref_price) < max(spec.min_notional, 5.0):
                logger.warning("Skip %s %s: Leg-1 qty %.6f below min lot/notional", sym, side, qty1)
                return None

            tp_price, sl_price, be_trig_px, be_lock_px, dca_px = self._calc_levels_for(
                sym, side, ref_price, in_storm=in_storm, tp_pct=tp_pct
            )
            entry_price = ref_price
            fill_qty = qty1

            # 0. Persist PENDING_OPEN draft to SQLite BEFORE sending market entry order (crash/restart atomic guard)
            dca_qty_plan = 0.0
            if self.dca_enabled and m2 >= 5.0 and dca_px > 0:
                cand_dq = round_step_down((m2 * self.leverage) / dca_px, spec.qty_step)
                if cand_dq >= spec.min_order_qty and (cand_dq * dca_px) >= max(spec.min_notional, 5.0):
                    dca_qty_plan = cand_dq
            m1_plan = round(qty1 * ref_price / self.leverage, 2)
            dca_m_plan = round(dca_qty_plan * dca_px / self.leverage, 2) if dca_qty_plan > 0 else 0.0
            res_plan = round(m1_plan + dca_m_plan, 2)
            filters_snap = {
                "rvol": float(getattr(ev, "rvol", 0.0)),
                "retail_ls_z": float(getattr(ev, "retail_ls_z", 0.0)),
                "funding_bps": float(getattr(ev, "funding_bps", 0.0)),
                "in_storm": in_storm,
                "dca_step_pct": round(dca_step * 100.0, 2),
            }
            pos_id = self.journal.add_position(
                symbol=sym,
                side=side,
                key_name=str(ev.key),
                status="PENDING_OPEN",
                order_id="",
                entry_price=ref_price,
                qty=qty1,
                margin_usd=m1_plan,
                reserved_margin_usd=res_plan,
                leverage=self.leverage,
                tp_price=tp_price,
                sl_price=sl_price,
                be_trigger_price=be_trig_px,
                be_lock_price=be_lock_px,
                entry_bar_ts=int(ev.bar_ts),
                expire_ts=0,
                filters=filters_snap,
                dca_order_id="",
                dca_price=dca_px if dca_qty_plan > 0 else 0.0,
                dca_qty=dca_qty_plan,
                dca_filled_qty=0.0,
                dca_margin_usd=dca_m_plan,
                dca_status="PENDING" if dca_qty_plan > 0 else "NONE",
                in_storm=in_storm,
                tp_pct=round(tp_pct * 100.0, 2),
            )

            # 1. Leg 1 Market IOC with atomic server-side Market SL
            mkt_oid = ""
            try:
                mkt_oid = self._place_market_entry_with_sl(sym, side, qty1, sl_price)
            except Exception as exc:
                if self.dry_run or "Bybit Market place_order failed" in str(exc):
                    self.journal.mark_cancelled(pos_id, "ORDER_REJECTED")
                    raise
                try:
                    ex_chk = self.client.get_positions(sym)
                except Exception:
                    self.journal.set_last_signal_bar_ts(sym, int(ev.bar_ts), side)
                    raise exc
                if not ex_chk:
                    self.journal.mark_cancelled(pos_id, "ORDER_FAILED")
                    raise

            self.journal.set_last_signal_bar_ts(sym, int(ev.bar_ts), side)

            # 2. Confirm actual Leg-1 fill price & qty from exchange
            if not self.dry_run and self.client.has_credentials:
                for _ in range(3):
                    time.sleep(0.35)
                    try:
                        ex_p = self.client.get_positions(sym)
                        if ex_p:
                            avg_p = float(ex_p[0].get("avgPrice") or 0.0)
                            ex_sz = float(ex_p[0].get("size") or 0.0)
                            if avg_p > 0:
                                entry_price = avg_p
                            if ex_sz > 0:
                                fill_qty = ex_sz
                            break
                    except Exception as exc:
                        logger.warning("Post-market fill lookup retry for %s: %s", sym, exc)

            tp_price, actual_sl, be_trig_px, be_lock_px, dca_px = self._calc_levels_for(
                sym, side, entry_price, in_storm=in_storm, tp_pct=tp_pct
            )
            if (not self.dry_run) and self.client.has_credentials and abs(actual_sl - sl_price) >= spec.tick_size:
                try:
                    self.client.set_trading_stop(sym, actual_sl)
                    sl_price = actual_sl
                except Exception as exc:
                    logger.warning("Keeping initial atomic SL %s for %s (%s)", sl_price, sym, exc)

            # 3. Place resting Limit Maker TP for Leg-1 qty
            if self.tp_order_type == "Limit":
                self._place_limit_tp(sym, side, fill_qty, tp_price)

            # 4. Place resting Leg-2 Limit GTC DCA order (65% normal / 70% storm)
            dca_oid = ""
            dca_qty = 0.0
            dca_status = "NONE"
            actual_m1 = round(fill_qty * entry_price / self.leverage, 2)
            reserved_total = actual_m1
            if self.dca_enabled and m2 >= 5.0 and dca_px > 0:
                dca_qty = round_step_down((m2 * self.leverage) / dca_px, spec.qty_step)
                if dca_qty >= spec.min_order_qty and (dca_qty * dca_px) >= max(spec.min_notional, 5.0):
                    try:
                        dca_oid = self._place_dca_limit_order(sym, side, dca_qty, dca_px)
                        dca_status = "PENDING"
                        reserved_total = round(actual_m1 + (dca_qty * dca_px / self.leverage), 2)
                    except Exception as exc:
                        logger.error("Failed placing Leg-2 DCA limit for %s: %s (Leg 1 stays protected by SL/TP)", sym, exc)
                        dca_qty = 0.0

            self.journal.finalize_pending_open(
                pos_id,
                order_id=mkt_oid,
                entry_price=entry_price,
                qty=fill_qty,
                margin_usd=actual_m1,
                reserved_margin_usd=reserved_total,
                tp_price=tp_price,
                sl_price=sl_price,
                be_trigger_price=be_trig_px,
                be_lock_price=be_lock_px,
                dca_order_id=dca_oid,
                dca_price=dca_px if dca_qty > 0 else 0.0,
                dca_qty=dca_qty,
                dca_margin_usd=round(dca_qty * dca_px / self.leverage, 2) if dca_qty > 0 else 0.0,
                dca_status=dca_status,
                dca_filled_qty=0.0,
            )

            self._bar_entry_counts[bar_key] = used_bar + 1
            if len(self._bar_entry_counts) > 32:
                for old_k in sorted(self._bar_entry_counts.keys())[:-16]:
                    self._bar_entry_counts.pop(old_k, None)

            snap_after = self.get_margin_reserve_snapshot()
            badge = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
            mode_tag = " [PAPER]" if self.dry_run else (" [DEMO]" if self.mode == "demo" else "")
            storm_tag = " ⚡ШТОРМ" if in_storm else ""
            dca_line = (
                f"2-я нога DCA ({int(round((1-leg1_frac)*100))}%): <code>Limit @ {dca_px}</code> (qty <code>{dca_qty}</code>)"
                if dca_status == "PENDING"
                else "2-я нога DCA: <code>выкл</code>"
            )
            ls_txt = f"{getattr(ev, 'retail_ls_z', 0.0):+.2f}" if getattr(ev, "retail_ls_available", False) else "n/a"
            msg = (
                f"🚀 <b>ВХОД Vika Hybrid{mode_tag}{storm_tag} | {sym} {badge}</b>\n"
                f"Ключ: <code>{ev.key}</code> | {side}: <b>{same_dir_cnt + 1}/{self.max_same_dir}</b> | Всего: <b>{len(active) + 1}/{self.max_open}</b>\n"
                f"1-я нога ({int(round(leg1_frac*100))}% Market): <code>{entry_price}</code> (qty <code>{fill_qty}</code>)\n"
                f"{dca_line}\n"
                f"TP (Maker): <code>{tp_price}</code> (+{tp_pct*100:.1f}%) | SL (атом.): <code>{sl_price}</code> (-{sl_pct*100:.1f}%)\n"
                f"BE: <code>+1.0% → +0.30%</code> (<code>{be_lock_px}</code>) + отмена DCA\n"
                f"Резерв слота: <code>${reserved_total:,.1f}</code> ({self.leverage}x Isolated) | "
                f"Резерв счёта: <b>{snap_after['reserve_pct']:.1f}% / {self.max_total_margin_pct*100:.0f}%</b>\n"
                f"Фильтры: Retail L/S z=<code>{ls_txt}</code> | RVOL=<code>{getattr(ev, 'rvol', 0.0):.2f}</code> | FR=<code>{getattr(ev, 'funding_bps', 0.0):+.2f}bps</code>"
            )
            logger.info(
                "Opened Vika Hybrid %s %s (%s) id=%s leg1=%s@%s dca=%s@%s sl=%s tp=%s res=$%.1f (%.1f%%)",
                sym, side, ev.key, pos_id, fill_qty, entry_price, dca_qty, dca_px, sl_price, tp_price,
                reserved_total, snap_after["reserve_pct"],
            )
            self.notify(msg)
            return pos_id

    # ------------------------------------------------------------------
    # Exchange Sync (every 20s): DCA Fill Detection, TP/SL Guard & Exit Reconciliation
    # ------------------------------------------------------------------
    def _ensure_server_tpsl(
        self,
        sym: str,
        pos: Dict[str, Any],
        ep_ex: Dict[str, Any],
        ex_ord_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        ex_sl = float(ep_ex.get("stopLoss") or 0.0)
        ex_tp = float(ep_ex.get("takeProfit") or 0.0)
        if ex_sl <= 0:
            try:
                self.client.set_trading_stop(sym, float(pos["sl_price"]))
                logger.info("Restored missing server-side SL for %s @ %s", sym, pos["sl_price"])
            except Exception as exc:
                logger.warning("Failed to restore SL for %s: %s", sym, exc)

        if ex_tp > 0:
            return
        if self.tp_order_type == "Limit":
            close_side = "Sell" if pos["side"] == "LONG" else "Buy"
            orders = ex_ord_list if ex_ord_list is not None else []
            has_limit_tp = any(_is_reduce_only_limit_tp(o, sym, close_side) for o in orders)
            if not has_limit_tp:
                ex_sz = float(ep_ex.get("size") or pos["qty"])
                if ex_sz > 0:
                    self._place_limit_tp(sym, pos["side"], ex_sz, float(pos["tp_price"]))

    def _compute_dca_targets(
        self, pos: Dict[str, Any], ep_ex: Dict[str, Any]
    ) -> Tuple[float, float, float, float, float, float]:
        """Recompute TP and BE levels from updated exchange avgPrice & size."""
        sym = pos["symbol"]
        side = pos["side"]
        spec = self.specs.get(sym)
        tick = spec.tick_size if spec else 0.0001
        new_ep = float(ep_ex.get("avgPrice") or pos["entry_price"])
        new_qty = float(ep_ex.get("size") or pos["qty"])
        new_margin = round(new_qty * new_ep / self.leverage, 2)
        tp_pct = float(pos.get("tp_pct") or 1.5) / 100.0
        if side == "LONG":
            new_tp = round_tick(new_ep * (1.0 + tp_pct), tick)
            new_be_trig = round_tick(new_ep * (1.0 + self.be_trigger_pct), tick)
            new_be_lock = round_tick(new_ep * (1.0 + self.be_lock_pct), tick)
        else:
            new_tp = round_tick(new_ep * (1.0 - tp_pct), tick)
            new_be_trig = round_tick(new_ep * (1.0 - self.be_trigger_pct), tick)
            new_be_lock = round_tick(new_ep * (1.0 - self.be_lock_pct), tick)
        return new_ep, new_qty, new_margin, new_tp, new_be_trig, new_be_lock

    def _replace_limit_tp_for_qty(
        self,
        sym: str,
        side: str,
        qty: float,
        tp_price: float,
        ex_ord_list: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._cancel_existing_limit_tps(sym, side, ex_ord_list)
        if self.tp_order_type == "Limit":
            self._place_limit_tp(sym, side, qty, tp_price)

    def _handle_dca_fill_on_exchange(
        self,
        pos: Dict[str, Any],
        ep_ex: Dict[str, Any],
        ex_ord_list: List[Dict[str, Any]],
        *,
        dca_filled_qty: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Recompute TP (+1.5%/+1.6%) and BE (+1.0%/+0.30%) from new avgPrice after Leg-2 DCA fills."""
        sym = pos["symbol"]
        side = pos["side"]
        new_ep, new_qty, new_margin, new_tp, new_be_trig, new_be_lock = self._compute_dca_targets(pos, ep_ex)
        tp_pct = float(pos.get("tp_pct") or 1.5) / 100.0
        tot_dca_filled = (
            float(dca_filled_qty)
            if dca_filled_qty is not None
            else float(pos.get("dca_filled_qty") or 0.0) + max(0.0, new_qty - float(pos["qty"]))
        )

        # Replace Leg-1 Limit Maker TP with full-size Limit Maker TP at recalculated avgPrice
        self._replace_limit_tp_for_qty(sym, side, new_qty, new_tp, ex_ord_list)

        self.journal.update_dca_filled(
            pos["id"],
            new_entry_price=new_ep,
            new_qty=new_qty,
            new_margin_usd=new_margin,
            new_tp_price=new_tp,
            new_be_trigger_price=new_be_trig,
            new_be_lock_price=new_be_lock,
            dca_status="FILLED",
            dca_filled_qty=tot_dca_filled,
        )
        pos = dict(pos)
        pos.update(
            entry_price=new_ep, qty=new_qty, margin_usd=new_margin, reserved_margin_usd=new_margin,
            tp_price=new_tp, be_trigger_price=new_be_trig, be_lock_price=new_be_lock,
            dca_order_id="", dca_status="FILLED", dca_filled_qty=tot_dca_filled,
        )

        badge = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
        msg = (
            f"➕ <b>ЗАЛИТА 2-Я НОГА DCA | {sym} {badge}</b>\n"
            f"Новая средняя цена: <code>{new_ep}</code> | Полный объём: <code>{new_qty}</code>\n"
            f"Пересчитан TP (Maker): <code>{new_tp}</code> (+{tp_pct*100:.1f}%) | SL сохранён: <code>{pos['sl_price']}</code>\n"
            f"Новый BE: <code>{new_be_trig} → {new_be_lock} (+0.30%)</code> | Маржа: <code>${new_margin:,.1f}</code>"
        )
        logger.info("Leg-2 DCA filled for %s %s: avg=%s qty=%s new_tp=%s", sym, side, new_ep, new_qty, new_tp)
        self.notify(msg)
        return pos

    def _apply_partial_dca_fill(
        self,
        pos: Dict[str, Any],
        ep_ex: Dict[str, Any],
        dca_ord: Dict[str, Any],
        ex_ord_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Track partial Leg-2 DCA execution while DCA remainder is still resting in the book."""
        sym = pos["symbol"]
        side = pos["side"]
        prev_qty = float(pos["qty"])
        new_ep, new_qty, new_margin, new_tp, new_be_trig, new_be_lock = self._compute_dca_targets(pos, ep_ex)
        delta_qty = max(0.0, new_qty - prev_qty)
        new_dca_filled = float(pos.get("dca_filled_qty") or 0.0) + delta_qty

        leaves_raw = dca_ord.get("leavesQty")
        if leaves_raw is not None and str(leaves_raw) != "":
            rem_dca_qty = max(0.0, float(leaves_raw))
        elif dca_ord.get("cumExecQty") is not None and str(dca_ord.get("cumExecQty")) != "":
            rem_dca_qty = max(0.0, float(dca_ord.get("qty") or pos.get("dca_qty") or 0.0) - float(dca_ord["cumExecQty"]))
        else:
            rem_dca_qty = max(0.0, float(pos.get("dca_qty") or 0.0) - new_dca_filled)

        dca_px = float(pos.get("dca_price") or dca_ord.get("price") or 0.0)
        new_reserved = round(new_margin + (rem_dca_qty * dca_px / self.leverage if dca_px > 0 else 0.0), 2)

        self._replace_limit_tp_for_qty(sym, side, new_qty, new_tp, ex_ord_list)
        self.journal.update_dca_partial(
            pos["id"],
            new_entry_price=new_ep,
            new_qty=new_qty,
            new_margin_usd=new_margin,
            new_reserved_margin_usd=new_reserved,
            new_tp_price=new_tp,
            new_be_trigger_price=new_be_trig,
            new_be_lock_price=new_be_lock,
            dca_filled_qty=new_dca_filled,
        )
        pos = dict(pos)
        pos.update(
            entry_price=new_ep,
            qty=new_qty,
            margin_usd=new_margin,
            reserved_margin_usd=new_reserved,
            tp_price=new_tp,
            be_trigger_price=new_be_trig,
            be_lock_price=new_be_lock,
            dca_filled_qty=new_dca_filled,
        )
        logger.info(
            "Partial Leg-2 DCA fill for %s %s: qty=%s->%s avg=%s new_tp=%s rem_dca=%s",
            sym, side, prev_qty, new_qty, new_ep, new_tp, rem_dca_qty,
        )
        return pos

    def sync_with_exchange(self) -> None:
        with self._lock:
            if self.dry_run or not self.client.has_credentials:
                self.check_daily_stop()
                return

            try:
                ex_pos_list = self.client.get_positions()
                ex_ord_list = self.client.get_open_orders()
            except Exception as exc:
                logger.warning("Exchange sync error: %s", exc)
                return

            self.recover_pending_open_positions(ex_pos_list=ex_pos_list, ex_ord_list=ex_ord_list)

            ex_pos_map = {p["symbol"]: p for p in ex_pos_list}
            ex_ord_map = {str(o.get("orderId", "")): o for o in ex_ord_list if o.get("orderId")}
            self.last_exchange_positions = ex_pos_map
            now_ms = int(time.time() * 1000)

            for pos in self.journal.get_active_positions():
                sym = pos["symbol"]
                has_pos = sym in ex_pos_map
                dca_oid = str(pos.get("dca_order_id") or "")
                dca_pending = (pos.get("dca_status") == "PENDING") and bool(dca_oid)
                has_dca_ord = bool(dca_oid and dca_oid in ex_ord_map)

                if not has_pos:
                    if has_dca_ord:
                        try:
                            self.client.cancel_order(sym, dca_oid)
                        except Exception as exc:
                            logger.warning("Failed cancelling leftover DCA limit for closed %s: %s", sym, exc)
                    self._cancel_existing_limit_tps(sym, pos["side"], ex_ord_list)
                    self._reconcile_missing_position(pos, now_ms)
                    continue

                ep_ex = ex_pos_map[sym]
                ex_sz = float(ep_ex.get("size") or 0.0)
                ex_avg = float(ep_ex.get("avgPrice") or pos["entry_price"])
                prev_qty = float(pos["qty"])
                spec = self.specs.get(sym)
                step = spec.qty_step if spec and spec.qty_step > 0 else 1e-6
                eps_qty = max(step * 0.5, prev_qty * 1e-6, 1e-9)
                sym_ords_for_check = ex_ord_list
                tp_replaced = False

                if ex_sz > prev_qty + eps_qty:
                    if dca_pending and has_dca_ord:
                        pos = self._apply_partial_dca_fill(pos, ep_ex, ex_ord_map[dca_oid], ex_ord_list)
                        tp_replaced = True
                    elif dca_pending and not has_dca_ord:
                        pos = self._handle_dca_fill_on_exchange(pos, ep_ex, ex_ord_list)
                        tp_replaced = True
                    else:
                        # Size grew without active DCA tracking: keep TP qty synchronized
                        new_ep, new_qty, new_margin, new_tp, new_be_trig, new_be_lock = self._compute_dca_targets(pos, ep_ex)
                        self._replace_limit_tp_for_qty(sym, pos["side"], new_qty, new_tp, ex_ord_list)
                        tp_replaced = True
                        self.journal.update_dca_filled(
                            pos["id"],
                            new_entry_price=new_ep,
                            new_qty=new_qty,
                            new_margin_usd=new_margin,
                            new_tp_price=new_tp,
                            new_be_trigger_price=new_be_trig,
                            new_be_lock_price=new_be_lock,
                            dca_status=str(pos.get("dca_status") or "FILLED"),
                        )
                        pos = dict(pos)
                        pos.update(
                            entry_price=new_ep, qty=new_qty, margin_usd=new_margin, reserved_margin_usd=new_margin,
                            tp_price=new_tp, be_trigger_price=new_be_trig, be_lock_price=new_be_lock,
                        )
                elif dca_pending and (not has_dca_ord):
                    prev_dca_filled = float(pos.get("dca_filled_qty") or 0.0)
                    if prev_dca_filled > eps_qty:
                        self.journal.mark_dca_cancelled(pos["id"], "PARTIAL_FILLED", float(pos["margin_usd"]))
                        pos = dict(pos)
                        pos.update(dca_order_id="", dca_status="PARTIAL_FILLED", reserved_margin_usd=float(pos["margin_usd"]))
                        logger.info("Leg-2 DCA remainder for %s gone; preserved partial fill (qty=%s).", sym, pos["qty"])
                    else:
                        self.journal.mark_dca_cancelled(pos["id"], "CANCELLED_EXT", float(pos["margin_usd"]))
                        pos = dict(pos)
                        pos.update(dca_order_id="", dca_status="CANCELLED_EXT", reserved_margin_usd=float(pos["margin_usd"]))
                        logger.warning("Pending Leg-2 DCA limit for %s disappeared unfilled; freed reserved margin.", sym)
                elif 0 < ex_sz < prev_qty - eps_qty:
                    # Partial Limit TP reduction: cancel resting DCA order immediately so it doesn't re-add
                    if dca_oid or pos.get("dca_status") == "PENDING":
                        if dca_oid:
                            try:
                                self.client.cancel_order(sym, dca_oid)
                            except Exception as exc:
                                logger.warning("Failed cancelling DCA limit on partial TP for %s: %s", sym, exc)
                    new_margin = round(ex_sz * ex_avg / self.leverage, 2)
                    self.journal.update_partial_tp_reduce(pos["id"], new_qty=ex_sz, new_margin_usd=new_margin, cancel_dca=True)
                    pos = dict(pos)
                    pos.update(
                        qty=ex_sz,
                        margin_usd=new_margin,
                        reserved_margin_usd=new_margin,
                        dca_order_id="",
                        dca_status="CANCELLED_TP_PARTIAL" if pos.get("dca_status") == "PENDING" else pos.get("dca_status"),
                    )
                    logger.info("Partial TP reduction for %s %s: remaining qty=%s (cancelled pending DCA).", sym, pos["side"], ex_sz)

                if tp_replaced:
                    try:
                        sym_ords_for_check = self.client.get_open_orders(sym)
                    except Exception:
                        sym_ords_for_check = ex_ord_list
                self._ensure_server_tpsl(sym, pos, ep_ex, sym_ords_for_check)

            self.check_daily_stop()

    def _fetch_position_closed_pnl(self, pos: Dict[str, Any]) -> Tuple[bool, float, float]:
        """Sum ALL partial closedPnl records belonging to this position (weighted avg exit price)."""
        sym = pos["symbol"]
        opened_at = int(pos["opened_at"])
        fallback_px = float(pos["entry_price"])
        if self.dry_run or not self.client.has_credentials:
            return False, 0.0, fallback_px
        try:
            closed_list = self.client.get_closed_pnl(sym, start_ms=max(0, opened_at - 60_000), limit=50)
        except Exception as exc:
            logger.warning("get_closed_pnl failed for %s: %s", sym, exc)
            return False, 0.0, fallback_px

        last_closed_ms = self.journal.get_last_closed_at_ms(sym)
        if last_closed_ms > 0 and last_closed_ms < opened_at:
            cutoff_ms = max(opened_at - 30_000, last_closed_ms + 1)
        elif last_closed_ms > 0:
            cutoff_ms = opened_at
        else:
            cutoff_ms = opened_at - 30_000

        matched: List[Dict[str, Any]] = []
        for item in closed_list:
            upd = int(item.get("updatedTime") or item.get("createdTime") or 0)
            if upd >= cutoff_ms:
                matched.append(item)

        if not matched:
            return False, 0.0, fallback_px

        total_pnl = sum(float(it.get("closedPnl") or 0.0) for it in matched)
        w_sum = 0.0
        px_w_sum = 0.0
        px_fallback: List[float] = []
        for it in matched:
            px = float(it.get("avgExitPrice") or 0.0)
            sz = float(it.get("closedSize") or it.get("qty") or 0.0)
            if px > 0:
                px_fallback.append(px)
                if sz > 0:
                    w_sum += sz
                    px_w_sum += sz * px
        if w_sum > 0:
            exit_px = px_w_sum / w_sum
        elif px_fallback:
            exit_px = sum(px_fallback) / len(px_fallback)
        else:
            exit_px = fallback_px
        return True, total_pnl, exit_px

    def _reconcile_missing_position(self, pos: Dict[str, Any], now_ms: int) -> None:
        sym = pos["symbol"]
        pos_id = pos["id"]
        side = pos["side"]
        found_closed, pnl_usd, exit_px = self._fetch_position_closed_pnl(pos)

        if pos["status"] in ("PENDING", "PENDING_OPEN") and not found_closed:
            self.journal.mark_cancelled(pos_id, "CANCELLED_EXTERNAL")
            return

        # STRICT L2_news SEPARATION (Step 16/17): BE exits NEVER trigger AC 6h or Coin Probation!
        is_be_active = bool(pos.get("be_active"))
        is_hard_sl = False
        if is_be_active:
            dist_be = abs(exit_px - float(pos["be_lock_price"]))
            dist_tp = abs(exit_px - float(pos["tp_price"]))
            reason = "TP" if (pnl_usd > 0 and dist_tp < dist_be) else "BE"
        elif pnl_usd > 0:
            reason = "TP"
        else:
            reason = "HARD_SL"
            is_hard_sl = True

        margin = max(float(pos["margin_usd"]), 1.0)
        pnl_pct = pnl_usd / margin * 100.0
        cd_bars = self.cooldown_bars_long if side == "LONG" else self.cooldown_bars_short
        ac_until = now_ms + self.ac_pause_hours * 3_600_000
        prob_hit, sl_cnt, prob_until = self.journal.record_position_exit_atomic(
            pos_id=pos_id,
            symbol=sym,
            side=side,
            now_ms=now_ms,
            exit_price=exit_px,
            exit_reason=reason,
            pnl_usd=round(pnl_usd, 2),
            pnl_pct=round(pnl_pct, 2),
            cooldown_until_ms=now_ms + cd_bars * M15_MS,
            is_hard_sl=is_hard_sl,
            ac_pause_until_ms=ac_until,
            probation_window_days=self.probation_window_days,
            probation_threshold=self.probation_sl_count,
            probation_pause_days=self.probation_pause_days,
        )

        pause_note = ""
        if is_hard_sl:
            ac_dt = datetime.fromtimestamp(ac_until / 1000, tz=timezone.utc).strftime("%H:%M UTC")
            pause_note = f"\n🧊 <b>AC {self.ac_pause_hours}ч ({side}):</b> пауза до <code>{ac_dt}</code>"
            if prob_hit:
                pr_dt = datetime.fromtimestamp(prob_until / 1000, tz=timezone.utc).strftime("%d.%m %H:%M UTC")
                pause_note += f"\n⛔ <b>Probation ({sym}):</b> {sl_cnt}/{self.probation_sl_count} Hard SL → карантин {self.probation_pause_days}д до <code>{pr_dt}</code>"
            else:
                pause_note += f" | Probation {sym}: <code>{sl_cnt}/{self.probation_sl_count}</code> Hard SL за {self.probation_window_days}д"
        elif reason == "BE":
            pause_note = "\n🛡 <i>Выход по BE +0.30%: AC 6ч и Probation НЕ затронуты.</i>"

        ds = self.check_daily_stop()
        icon = "❌" if is_hard_sl else ("🛡" if reason == "BE" else "✅")
        badge = "🟢 LONG" if side == "LONG" else "🔴 SHORT"
        dca_lbl = " (2-я нога залита)" if pos.get("dca_status") in ("FILLED", "PARTIAL_FILLED") else ""
        msg = (
            f"{icon} <b>ЗАКРЫТИЕ ({reason}) | {sym} {badge}{dca_lbl}</b>\n"
            f"Ключ: <code>{pos['key_name']}</code>\n"
            f"Вход: <code>{pos['entry_price']}</code> → Выход: <code>{exit_px}</code>\n"
            f"Результат: <b>{pnl_usd:+,.2f} USDT ({pnl_pct:+.2f}% к марже)</b>\n"
            f"Баланс: <code>${ds['current_equity']:,.2f}</code> | День: <code>{ds['daily_pnl_usd']:+,.2f} ({ds['daily_pnl_pct']:+.2f}%)</code>"
            f"{pause_note}"
        )
        logger.info("Closed %s %s (%s): pnl=%.2f USDT hard_sl=%s", sym, side, reason, pnl_usd, is_hard_sl)
        self.notify(msg)

    # ------------------------------------------------------------------
    # Closed-15m-Bar Breakeven (+1.00% -> +0.30%) + Synchronous Leg-2 DCA Limit Cancel
    # ------------------------------------------------------------------
    def check_breakeven_on_closed_bar(self, sym: str, closed_bar_ts: int, closed_price: float) -> bool:
        """
        On closed 15m candle:
          - If close passes be_trigger_price (+1.00% from current avg entry):
            1) Synchronously cancel any pending Leg-2 DCA limit order on Bybit FIRST.
            2) Re-verify live exchange position (size & avgPrice) against 110001/race fills before moving SL.
            3) Move server-side SL on Bybit to be_lock_price (+0.30%) & update SQLite state.
        """
        with self._lock:
            moved_any = False
            for pos in self.journal.get_open_positions():
                if pos["symbol"] != sym or pos["be_active"]:
                    continue
                if closed_bar_ts <= int(pos["entry_bar_ts"]):
                    continue

                is_l = pos["side"] == "LONG"
                spec = self.specs.get(sym)
                tick = spec.tick_size if spec else 0.0001
                step = spec.qty_step if spec and spec.qty_step > 0 else 1e-6
                eps_qty = max(step * 0.5, float(pos["qty"]) * 1e-6, 1e-9)

                # Pre-sync if Leg-2 DCA already filled/partially filled before BE check
                if not self.dry_run and self.client.has_credentials:
                    try:
                        ex_pos_pre = self.client.get_positions(sym)
                    except Exception as exc:
                        logger.warning("Pre-BE position lookup failed for %s: %s", sym, exc)
                        continue
                    if not ex_pos_pre:
                        continue
                    ep_pre = ex_pos_pre[0]
                    sz_pre = float(ep_pre.get("size") or 0.0)
                    if pos.get("dca_status") == "PENDING" and sz_pre > float(pos["qty"]) + eps_qty:
                        try:
                            sym_ords = self.client.get_open_orders(sym)
                        except Exception as exc:
                            logger.warning("Pre-BE open orders lookup failed for %s: %s", sym, exc)
                            continue
                        dca_oid_pre = str(pos.get("dca_order_id") or "")
                        dca_ord_pre = next((o for o in sym_ords if str(o.get("orderId", "")) == dca_oid_pre), None)
                        if dca_ord_pre is not None:
                            pos = self._apply_partial_dca_fill(pos, ep_pre, dca_ord_pre, sym_ords)
                        else:
                            pos = self._handle_dca_fill_on_exchange(pos, ep_pre, sym_ords)
                        eps_qty = max(step * 0.5, float(pos["qty"]) * 1e-6, 1e-9)

                trig = float(pos["be_trigger_price"])
                hit = (closed_price >= trig) if is_l else (closed_price <= trig)
                if not hit:
                    continue

                dca_cancelled = False
                freed_margin = 0.0
                dca_oid = str(pos.get("dca_order_id") or "")
                if pos.get("dca_status") == "PENDING" and dca_oid:
                    old_reserved = float(pos.get("reserved_margin_usd") or pos["margin_usd"])
                    if not self.dry_run and self.client.has_credentials:
                        try:
                            cancel_res = self.client.cancel_order(sym, dca_oid)
                        except Exception as exc:
                            logger.error("Failed cancelling Leg-2 DCA before BE for %s (%s): %s", sym, dca_oid, exc)
                            continue
                        try:
                            ex_pos_after = self.client.get_positions(sym)
                        except Exception as exc:
                            logger.error("Failed re-checking exchange position after DCA cancel for %s: %s", sym, exc)
                            continue
                        if not ex_pos_after:
                            continue
                        ep_after = ex_pos_after[0]
                        sz_after = float(ep_after.get("size") or 0.0)
                        avg_after = float(ep_after.get("avgPrice") or pos["entry_price"])
                        if sz_after > float(pos["qty"]) + eps_qty or (
                            cancel_res is False and abs(avg_after - float(pos["entry_price"])) >= tick
                        ):
                            delta_qty = max(0.0, sz_after - float(pos["qty"]))
                            tot_dca_filled = float(pos.get("dca_filled_qty") or 0.0) + delta_qty
                            new_ep, new_qty, new_margin, new_tp, new_be_trig, new_be_lock = self._compute_dca_targets(pos, ep_after)
                            try:
                                sym_ords_after = self.client.get_open_orders(sym)
                            except Exception:
                                sym_ords_after = []
                            self._replace_limit_tp_for_qty(sym, pos["side"], new_qty, new_tp, sym_ords_after)
                            orig_dca_qty = float(pos.get("dca_qty") or 0.0)
                            is_full_dca = (
                                (cancel_res is False and sz_after > float(pos["qty"]) + eps_qty and (orig_dca_qty <= 0 or tot_dca_filled >= orig_dca_qty * 0.99))
                                or (orig_dca_qty > 0 and tot_dca_filled >= orig_dca_qty * 0.99)
                            )
                            if is_full_dca:
                                self.journal.update_dca_filled(
                                    pos["id"],
                                    new_entry_price=new_ep,
                                    new_qty=new_qty,
                                    new_margin_usd=new_margin,
                                    new_tp_price=new_tp,
                                    new_be_trigger_price=new_be_trig,
                                    new_be_lock_price=new_be_lock,
                                    dca_status="FILLED",
                                    dca_filled_qty=tot_dca_filled,
                                )
                                pos = dict(pos)
                                pos.update(
                                    entry_price=new_ep, qty=new_qty, margin_usd=new_margin, reserved_margin_usd=new_margin,
                                    tp_price=new_tp, be_trigger_price=new_be_trig, be_lock_price=new_be_lock,
                                    dca_order_id="", dca_status="FILLED", dca_filled_qty=tot_dca_filled,
                                )
                                dca_cancelled = False
                            else:
                                self.journal.update_dca_partial(
                                    pos["id"],
                                    new_entry_price=new_ep,
                                    new_qty=new_qty,
                                    new_margin_usd=new_margin,
                                    new_reserved_margin_usd=new_margin,
                                    new_tp_price=new_tp,
                                    new_be_trigger_price=new_be_trig,
                                    new_be_lock_price=new_be_lock,
                                    dca_filled_qty=tot_dca_filled,
                                )
                                pos = dict(pos)
                                pos.update(
                                    entry_price=new_ep, qty=new_qty, margin_usd=new_margin, reserved_margin_usd=new_margin,
                                    tp_price=new_tp, be_trigger_price=new_be_trig, be_lock_price=new_be_lock,
                                    dca_filled_qty=tot_dca_filled,
                                )
                                dca_cancelled = True
                                freed_margin = max(0.0, old_reserved - new_margin)

                            trig = new_be_trig
                            still_hit = (closed_price >= trig) if is_l else (closed_price <= trig)
                            if not still_hit:
                                if dca_cancelled:
                                    self.journal.mark_dca_cancelled(pos["id"], "PARTIAL_FILLED", new_margin)
                                logger.info(
                                    "BE skipped after concurrent DCA fill for %s: closed=%s vs new_trig=%s",
                                    sym, closed_price, trig,
                                )
                                continue
                        else:
                            dca_cancelled = True
                            freed_margin = max(0.0, old_reserved - float(pos["margin_usd"]))
                    else:
                        dca_cancelled = True
                        freed_margin = max(0.0, old_reserved - float(pos["margin_usd"]))

                lock_px = float(pos["be_lock_price"])
                if not self.dry_run and self.client.has_credentials:
                    try:
                        self.client.set_trading_stop(sym, lock_px)
                    except Exception as exc:
                        logger.error("Failed moving SL to BE +0.30%% for %s: %s", sym, exc)
                        if dca_cancelled and pos.get("dca_status") == "PENDING":
                            rem_status = "PARTIAL_FILLED" if float(pos.get("dca_filled_qty") or 0.0) > eps_qty else "CANCELLED_EXT"
                            self.journal.mark_dca_cancelled(pos["id"], rem_status, float(pos["margin_usd"]))
                        continue

                self.journal.mark_be_active(
                    pos["id"],
                    lock_px,
                    cancel_dca=dca_cancelled,
                    new_reserved_margin_usd=float(pos["margin_usd"]) if dca_cancelled else None,
                )
                moved_any = True
                badge = "🟢 LONG" if is_l else "🔴 SHORT"
                dca_note = (
                    f"\n🧹 <b>Лимитка 2-й ноги DCA снята</b> (освобождено <code>${freed_margin:,.1f}</code> маржи)"
                    if dca_cancelled
                    else ""
                )
                msg = (
                    f"🛡 <b>ПЕРЕВОД В BE (+0.30%) | {sym} {badge}</b>\n"
                    f"Закрытие 15m: <code>{closed_price}</code> (триггер <code>{trig}</code>)\n"
                    f"Новый серверный SL: <code>{lock_px}</code> (средняя <code>{pos['entry_price']}</code>)"
                    f"{dca_note}"
                )
                logger.info("Moved %s %s SL to BE +0.30%% (%s), dca_cancelled=%s", sym, pos["side"], lock_px, dca_cancelled)
                self.notify(msg)
            return moved_any

    # ------------------------------------------------------------------
    # Panic Close All
    # ------------------------------------------------------------------
    def panic_close_all(self) -> str:
        self.journal.set_paused(True)
        with self._lock:
            closed_count = 0
            errors: List[str] = []
            active_rows = self.journal.get_active_positions()
            ex_pos_before_map: Dict[str, Dict[str, Any]] = {}
            failed_syms: set[str] = set()

            if not self.dry_run and self.client.has_credentials:
                try:
                    for p in self.client.get_positions():
                        ex_pos_before_map[p["symbol"]] = p
                except Exception as exc:
                    errors.append(f"get_positions: {exc}")

                syms_to_cancel = set(self.coin_cfg_map) | {p["symbol"] for p in active_rows} | set(ex_pos_before_map)
                for sym in syms_to_cancel:
                    try:
                        self.client.cancel_all_orders(sym)
                    except Exception:
                        pass

                for sym, p in ex_pos_before_map.items():
                    sz = float(p.get("size") or 0.0)
                    if sz <= 0:
                        continue
                    close_side = "Sell" if p.get("side") == "Buy" else "Buy"
                    try:
                        self.client.close_position_market(sym, close_side, sz)
                        closed_count += 1
                    except Exception as exc:
                        failed_syms.add(sym)
                        errors.append(f"{sym}: {exc}")

            if not self.dry_run and self.client.has_credentials and closed_count > 0:
                time.sleep(0.5)

            for pos in self.journal.get_active_positions():
                sym = pos["symbol"]
                if sym in failed_syms:
                    continue
                if pos["status"] == "PENDING":
                    self.journal.mark_cancelled(pos["id"], "PANIC_CLOSE")
                    continue
                if pos["status"] == "PENDING_OPEN" and (not self.dry_run) and self.client.has_credentials and sym not in ex_pos_before_map:
                    self.journal.mark_cancelled(pos["id"], "PANIC_CLOSE")
                    continue

                found_closed, pnl_usd, exit_px = self._fetch_position_closed_pnl(pos)
                if (not found_closed) and sym in ex_pos_before_map:
                    ex_p = ex_pos_before_map[sym]
                    pnl_usd = float(ex_p.get("unrealisedPnl") or 0.0)
                    mp = float(ex_p.get("markPrice") or 0.0)
                    if mp > 0:
                        exit_px = mp
                margin = max(float(pos["margin_usd"]), 1.0)
                pnl_pct = pnl_usd / margin * 100.0
                self.journal.mark_closed(pos["id"], exit_px, "PANIC_CLOSE", round(pnl_usd, 2), round(pnl_pct, 2))

            if self.dry_run or not self.client.has_credentials:
                time.sleep(0.0)
            eq, _ = self.refresh_balance()
            summary = (
                f"🛑 <b>PANIC CLOSE ВЫПОЛНЕН | {self.strategy_name}</b>\n"
                f"Закрыто позиций: <b>{closed_count}</b> | Все DCA/TP ордера отменены\n"
                f"Режим: <b>⏸ ПАУЗА</b> | Баланс: <code>${eq:,.2f}</code>"
            )
            if errors:
                summary += "\n⚠️ Ошибки: " + "; ".join(errors)
            logger.warning("PANIC CLOSE executed: closed=%d errors=%s", closed_count, errors)
            return summary
