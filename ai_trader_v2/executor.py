"""
Dual-Mode Trade Executor (`ai_trader_v2/executor.py`):
  - Paper Mode (real-time live Bybit candles + exact fee/slippage/TP1+BE+TP2 simulation + Telegram alerts)
  - Testnet / Demo Mode (real Bybit v5 API orders with Isolated Margin, 10x leverage, atomic server-side SL,
    reduce-only TP1 55% + TP2 45%, automatic server-side SL move to BE +0.15% after TP1 fill, and optional trailing).
  - Hard safety rules from trading-core.md:
      * Daily -6.0% UTC drawdown hard stop (realized + unrealized)
      * Mandatory server-side SL inside [1.2%, 1.6%] (>= 1.1 * ATR)
      * Max 6 open positions, max 4 in same direction
      * 10% margin x 10x leverage, risk <= 2.5% per trade
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from ai_trader_v2.ai_judge import AIDecision, redact_secrets
from ai_trader_v2.journal import AITradeJournal, utc_now_iso, utc_today_str
from ai_trader_v2.ta_schools import CandidateSetup
from bot.bybit_client import BybitClient, InstrumentSpec, round_step_down, round_tick

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Lightweight isolated Telegram alert sender using .env credentials (never logs secrets)."""

    def __init__(self, enabled: bool = True) -> None:
        load_dotenv(override=True)
        self.enabled = enabled
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.token and self.chat_id and "your_" not in self.token.lower())

    def send(self, text: str) -> None:
        if not self.is_ready:
            return
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
                timeout=(5, 10),
            )
        except Exception as exc:
            logger.warning("Telegram notify error: %s", redact_secrets(exc))


class AITradeExecutor:
    def __init__(
        self,
        client: BybitClient,
        journal: AITradeJournal,
        config: Dict[str, Any],
        notifier: Optional[TelegramNotifier] = None,
        paper_mode: bool = True,
    ) -> None:
        self.client = client
        self.journal = journal
        self.config = config
        self.notifier = notifier or TelegramNotifier(enabled=bool(config.get("telegram_enabled", True)))
        self.paper_mode = paper_mode or bool(config.get("dry_run", True)) or (not client.has_credentials)
        self._specs_cache: Dict[str, InstrumentSpec] = {}

    def _mode_label(self) -> str:
        return "paper" if self.paper_mode else self.client.mode

    def get_specs(self, symbols: List[str]) -> Dict[str, InstrumentSpec]:
        if not self._specs_cache:
            try:
                self._specs_cache = self.client.get_instrument_specs_for_symbols(symbols)
            except Exception as exc:
                logger.warning("Instrument specs fetch error: %s", exc)
        return self._specs_cache

    def get_current_equity(self) -> float:
        if self.paper_mode:
            return self.journal.get_paper_equity(float(self.config.get("paper_start_equity_usdt", 2000.0)))
        try:
            bal = self.client.get_wallet_balance("USDT")
            return float(bal.get("equity", 0.0))
        except Exception as exc:
            logger.warning("Failed to fetch exchange wallet balance: %s", exc)
            return 0.0

    def compute_unrealized_pnl(self, active_trades: List[Dict[str, Any]], tickers: Dict[str, Dict[str, Any]]) -> float:
        unr = 0.0
        for t in active_trades:
            if t["status"] not in ("OPEN", "TP1_HIT"):
                continue
            sym = t["symbol"]
            cp = float(tickers.get(sym, {}).get("last_price", 0.0))
            if cp <= 0:
                continue
            ep = float(t["entry_price"])
            qty = float(t["remaining_qty"])
            unr += (cp - ep) * qty if t["direction"] == "LONG" else (ep - cp) * qty
        return unr

    def check_daily_loss_limit(self, tickers: Optional[Dict[str, Dict[str, Any]]] = None) -> tuple[bool, str]:
        """Enforce mandatory -6.0% UTC daily loss hard stop (realized + unrealized)."""
        eq = self.get_current_equity()
        if eq <= 0:
            return False, "equity_unavailable"
        active = self.journal.get_active_trades()
        tmap = tickers if tickers is not None else self._safe_get_tickers()
        unr = self.compute_unrealized_pnl(active, tmap)
        total_eq = eq + unr
        row = self.journal.ensure_daily_row(total_eq)
        if int(row.get("halted", 0)) == 1:
            return False, f"daily_halted:{row.get('halt_reason') or 'limit'}"
        start_eq = float(row["start_equity"])
        if start_eq > 0:
            dd_pct = ((total_eq - start_eq) / start_eq) * 100.0
            limit_pct = float(self.config.get("daily_loss_limit_pct", 6.0))
            if dd_pct <= -limit_pct:
                reason = f"Daily drawdown {dd_pct:.2f}% <= -{limit_pct:.1f}% UTC limit"
                self.journal.update_daily_stats(total_eq, halted=True, halt_reason=reason)
                self.notifier.send(f"🛑 [AI-TRADER v2] ДНЕВНОЙ СТОП!\n{reason}\nНовые входы заблокированы до 00:00 UTC.")
                return False, reason
        self.journal.update_daily_stats(total_eq)
        return True, "ok"

    def _safe_get_tickers(self) -> Dict[str, Dict[str, Any]]:
        try:
            return self.client.get_tickers()
        except Exception:
            return {}

    def get_portfolio_state(self) -> Dict[str, Any]:
        active = self.journal.get_active_trades()
        eq = self.get_current_equity()
        longs = [t["symbol"] for t in active if t["direction"] == "LONG"]
        shorts = [t["symbol"] for t in active if t["direction"] == "SHORT"]
        return {
            "mode": self._mode_label(),
            "equity_usdt": round(eq, 2),
            "open_count": len(active),
            "max_open_positions": int(self.config.get("max_open_positions", 6)),
            "long_count": len(longs),
            "short_count": len(shorts),
            "max_same_direction": int(self.config.get("max_same_direction", 4)),
            "open_longs": longs,
            "open_shorts": shorts,
        }

    def can_open_symbol(self, symbol: str, direction: str) -> tuple[bool, str]:
        active = self.journal.get_active_trades()
        if any(t["symbol"] == symbol for t in active):
            return False, "already_open_or_pending"
        max_open = int(self.config.get("max_open_positions", 6))
        if len(active) >= max_open:
            return False, f"max_open_reached ({len(active)}/{max_open})"
        max_same = int(self.config.get("max_same_direction", 4))
        same_cnt = sum(1 for t in active if t["direction"] == direction)
        if same_cnt >= max_same:
            return False, f"max_same_dir_reached ({same_cnt}/{max_same} {direction})"

        # Cooldown check after recent close
        cd_bars = int(self.config.get("cooldown_bars_15m", 4))
        last_exit = self.journal.get_recent_closed_time_for_symbol(symbol)
        if last_exit and cd_bars > 0:
            try:
                exit_dt = datetime.fromisoformat(last_exit.replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < exit_dt + timedelta(minutes=15 * cd_bars):
                    return False, "symbol_cooldown"
            except ValueError:
                pass
        return True, "ok"

    def open_trade(self, candidate: CandidateSetup, decision: AIDecision) -> Optional[int]:
        """Validate limits, calculate exact position size, and place Paper or Exchange order with atomic SL."""
        ok_dd, dd_reason = self.check_daily_loss_limit()
        if not ok_dd:
            logger.warning("Entry blocked for %s: %s", candidate.symbol, dd_reason)
            return None

        ok_slot, slot_reason = self.can_open_symbol(candidate.symbol, candidate.direction)
        if not ok_slot:
            logger.info("Entry skipped for %s: %s", candidate.symbol, slot_reason)
            return None

        eq = self.get_current_equity()
        if eq < 20.0:
            logger.warning("Insufficient equity %.2f USDT for %s", eq, candidate.symbol)
            return None

        leverage = int(self.config.get("leverage", 10))
        leverage = max(int(self.config.get("min_leverage", 3)), min(int(self.config.get("max_leverage", 10)), leverage))
        margin_pct = float(self.config.get("margin_pct_per_trade", 10.0))
        margin_usdt = eq * (margin_pct / 100.0)
        notional_usdt = margin_usdt * leverage

        # Orderbook liquidity cap (never exceed 25% of ±0.3% depth)
        ob_depth = float(candidate.orderbook.get("depth_03_usdt", 0.0))
        if ob_depth > 0:
            notional_usdt = min(notional_usdt, ob_depth * 0.25)
            margin_usdt = notional_usdt / leverage

        # Re-verify hard max risk <= 2.5%
        sl_frac = decision.sl_pct / 100.0
        est_risk_pct = (notional_usdt * (sl_frac + 0.00075) / eq) * 100.0
        if est_risk_pct > float(self.config.get("max_risk_pct_per_trade", 2.5)):
            logger.warning("Risk guardrail rejected %s: %.2f%% > 2.5%%", candidate.symbol, est_risk_pct)
            return None

        specs = self.get_specs(self.config.get("symbols", [candidate.symbol]))
        spec = specs.get(candidate.symbol)
        tick = spec.tick_size if spec else 0.0001
        qty_step = spec.qty_step if spec else 0.001
        min_qty = spec.min_order_qty if spec else 0.0

        is_long = candidate.direction == "LONG"
        entry_type = str(self.config.get("entry_order_type", "limit")).lower().strip()

        if entry_type == "limit":
            # Maker limit at closed price / best passive quote
            bb = float(candidate.orderbook.get("best_bid", 0.0))
            ba = float(candidate.orderbook.get("best_ask", 0.0))
            raw_ep = min(candidate.close_price, bb) if (is_long and bb > 0) else (
                max(candidate.close_price, ba) if ((not is_long) and ba > 0) else candidate.close_price
            )
        else:
            raw_ep = candidate.close_price

        ep = round_tick(raw_ep, tick)
        sl_dist = ep * sl_frac
        sl_price = round_tick((ep - sl_dist) if is_long else (ep + sl_dist), tick)
        tp1_price = round_tick((ep + decision.tp1_r * sl_dist) if is_long else (ep - decision.tp1_r * sl_dist), tick)
        tp2_price = round_tick((ep + decision.tp2_r * sl_dist) if is_long else (ep - decision.tp2_r * sl_dist), tick)
        be_off = float(self.config.get("breakeven_offset_pct", 0.15)) / 100.0
        be_price = round_tick(ep * (1.0 + be_off) if is_long else ep * (1.0 - be_off), tick)

        qty = round_step_down(notional_usdt / ep, qty_step)
        if qty <= 0 or (min_qty > 0 and qty < min_qty):
            logger.warning("Qty %.6f below min_order_qty %.6f for %s", qty, min_qty, candidate.symbol)
            return None
        tp1_frac = float(self.config.get("tp1_close_pct", 55.0)) / 100.0
        tp1_qty = round_step_down(qty * tp1_frac, qty_step)
        if tp1_qty <= 0 or tp1_qty >= qty:
            tp1_qty = round_step_down(qty * 0.5, qty_step)

        ttl_min = int(self.config.get("limit_entry_ttl_minutes", 45))
        expires_iso = (datetime.now(timezone.utc) + timedelta(minutes=ttl_min)).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry_order_id = ""
        status = "PENDING_LIMIT" if entry_type == "limit" else "OPEN"

        if not self.paper_mode:
            side = "Buy" if is_long else "Sell"
            try:
                self.client.ensure_isolated_margin_and_leverage(candidate.symbol, leverage)
                if entry_type == "limit":
                    res = self.client.place_limit_entry_with_sl(
                        symbol=candidate.symbol, side=side, qty=qty, price=ep, stop_loss=sl_price
                    )
                else:
                    res = self.client.place_market_order_with_sl(
                        symbol=candidate.symbol, side=side, qty=qty, stop_loss=sl_price
                    )
                entry_order_id = str(res.get("orderId", ""))
            except Exception as exc:
                logger.error("Exchange order placement failed for %s: %s", candidate.symbol, exc)
                self.notifier.send(f"⚠️ [AI-TRADER v2] Ошибка ордера {candidate.symbol}: {exc}")
                return None
        else:
            entry_order_id = f"PAPER-{candidate.symbol}-{int(datetime.now(timezone.utc).timestamp())}"

        trade_row = {
            "symbol": candidate.symbol,
            "direction": candidate.direction,
            "status": status,
            "mode": self._mode_label(),
            "entry_time": utc_now_iso(),
            "expires_at_utc": expires_iso if status == "PENDING_LIMIT" else None,
            "entry_price": ep,
            "sl_price": sl_price,
            "init_sl_price": sl_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "be_price": be_price,
            "use_trailing": 1 if decision.use_trailing_after_tp1 else 0,
            "peak_price": ep,
            "qty": qty,
            "tp1_qty": tp1_qty,
            "remaining_qty": qty,
            "leverage": leverage,
            "margin_usdt": round(qty * ep / leverage, 2),
            "notional_usdt": round(qty * ep, 2),
            "sl_pct": decision.sl_pct,
            "tp1_r": decision.tp1_r,
            "tp2_r": decision.tp2_r,
            "primary_school": decision.selected_school,
            "schools_json": json.dumps(candidate.fired_schools, ensure_ascii=False),
            "ai_confidence": decision.confidence,
            "ai_reason": decision.reason_ru,
            "entry_order_id": entry_order_id,
        }
        trade_id = self.journal.insert_trade(trade_row)
        self.journal.update_daily_stats(eq, opened_delta=1)

        # If market order filled immediately on exchange, place TP1 & TP2 right away
        if (not self.paper_mode) and status == "OPEN":
            self._place_exchange_tps(trade_id, trade_row, spec)

        icon = "🟢" if is_long else "🔴"
        msg = (
            f"{icon} [AI-TRADER v2 | {self._mode_label().upper()}] {candidate.direction} {candidate.symbol}\n"
            f"Школы ({len(candidate.fired_schools)}): {', '.join(candidate.fired_schools)}\n"
            f"AI Уверенность: {decision.confidence}/100 | Режим: {candidate.regime_label}\n"
            f"Вход ({entry_type.upper()}): {ep} | Плечо: {leverage}x | Маржа: ${trade_row['margin_usdt']:.1f}\n"
            f"SL: {sl_price} (-{decision.sl_pct:.2f}%) | TP1 (55%): {tp1_price} ({decision.tp1_r}R -> БУ {be_price})\n"
            f"TP2 (45%): {tp2_price} ({decision.tp2_r}R{' + Трейл' if decision.use_trailing_after_tp1 else ''})\n"
            f"AI Разбор: {decision.reason_ru}"
        )
        logger.info(msg)
        self.notifier.send(msg)
        return trade_id

    def _place_exchange_tps(self, trade_id: int, t: Dict[str, Any], spec: Optional[InstrumentSpec]) -> None:
        close_side = "Sell" if t["direction"] == "LONG" else "Buy"
        qty_step = spec.qty_step if spec else 0.001
        tp1_qty = float(t["tp1_qty"])
        tp2_qty = round_step_down(float(t["qty"]) - tp1_qty, qty_step)
        tp1_oid = ""
        tp2_oid = ""
        try:
            if tp1_qty > 0:
                r1 = self.client.place_limit_tp_order(t["symbol"], close_side, tp1_qty, float(t["tp1_price"]))
                tp1_oid = str(r1.get("orderId", ""))
            if tp2_qty > 0 and not bool(t.get("use_trailing")):
                r2 = self.client.place_limit_tp_order(t["symbol"], close_side, tp2_qty, float(t["tp2_price"]))
                tp2_oid = str(r2.get("orderId", ""))
        except Exception as exc:
            logger.warning("Failed placing TP orders for %s (#%d): %s", t["symbol"], trade_id, exc)
        self.journal.update_trade(trade_id, {"tp1_order_id": tp1_oid, "tp2_order_id": tp2_oid})

    # ------------------------------------------------------------------
    # Position Lifecycle Management (called every 30s)
    # ------------------------------------------------------------------
    def manage_active_positions(self) -> None:
        active = self.journal.get_active_trades()
        if not active:
            return
        tickers = self._safe_get_tickers()
        self.check_daily_loss_limit(tickers)
        specs = self.get_specs([t["symbol"] for t in active])

        for t in active:
            if t["mode"] == "paper":
                self._manage_paper_trade(t, tickers, specs.get(t["symbol"]))
            else:
                self._manage_exchange_trade(t, tickers, specs.get(t["symbol"]))

    def _manage_paper_trade(
        self, t: Dict[str, Any], tickers: Dict[str, Dict[str, Any]], spec: Optional[InstrumentSpec]
    ) -> None:
        sym = t["symbol"]
        cp = float(tickers.get(sym, {}).get("last_price", 0.0))
        if cp <= 0:
            return
        tid = int(t["id"])
        is_long = t["direction"] == "LONG"
        ep = float(t["entry_price"])
        sl_p = float(t["sl_price"])
        tp1_p = float(t["tp1_price"])
        tp2_p = float(t["tp2_price"])
        be_p = float(t["be_price"])
        maker_fee = float(self.config.get("maker_fee_rate", 0.0002))
        taker_fee = float(self.config.get("taker_fee_rate", 0.00055))

        # 1. PENDING_LIMIT fill or TTL expiry
        if t["status"] == "PENDING_LIMIT":
            filled = (cp <= ep) if is_long else (cp >= ep)
            if filled:
                self.journal.update_trade(tid, {"status": "OPEN", "peak_price": ep})
                self.notifier.send(f"✅ [PAPER] Лимитный ордер {sym} ({t['direction']}) ИСПОЛНЕН по {ep}")
                return
            exp = t.get("expires_at_utc")
            if exp:
                exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
                if datetime.now(timezone.utc) >= exp_dt:
                    self.journal.update_trade(tid, {"status": "CLOSED", "exit_time": utc_now_iso(), "exit_reason": "LIMIT_EXPIRED"})
                    logger.info("Paper limit expired for %s (#%d)", sym, tid)
            return

        # 2. Check Stop-Loss
        hit_sl = (cp <= sl_p) if is_long else (cp >= sl_p)
        if hit_sl:
            rem_qty = float(t["remaining_qty"])
            gross_rem = (sl_p - ep) * rem_qty if is_long else (ep - sl_p) * rem_qty
            entry_fee = float(t["qty"]) * ep * maker_fee
            exit_fee = rem_qty * sl_p * taker_fee
            tp1_realized = float(t.get("realized_tp1_pnl") or 0.0)
            net_pnl = tp1_realized + gross_rem - entry_fee - exit_fee
            net_pct = (net_pnl / max(float(t["margin_usdt"]), 1e-6)) * 100.0
            reason = "BE_OR_TRAIL_SL" if t["status"] == "TP1_HIT" else "SL_HIT"
            self.journal.update_trade(
                tid,
                {
                    "status": "CLOSED",
                    "exit_time": utc_now_iso(),
                    "remaining_qty": 0.0,
                    "exit_reason": reason,
                    "gross_pnl_usdt": round(tp1_realized + gross_rem, 2),
                    "fees_usdt": round(entry_fee + exit_fee, 2),
                    "net_pnl_usdt": round(net_pnl, 2),
                    "net_pnl_pct": round(net_pct, 2),
                },
            )
            eq = self.get_current_equity()
            self.journal.update_daily_stats(eq, realized_delta=net_pnl, closed_win=(net_pnl > 0))
            ico = "🟦" if net_pnl >= 0 else "❌"
            self.notifier.send(
                f"{ico} [PAPER] ЗАКРЫТА {sym} ({t['direction']}) — {reason}\n"
                f"Net PnL: ${net_pnl:+.2f} ({net_pct:+.1f}% к марже) | Баланс: ${eq:.2f}"
            )
            return

        # 3. Check TP1 (55%) -> Immediate SL move to BE +0.15%
        if t["status"] == "OPEN":
            hit_tp1 = (cp >= tp1_p) if is_long else (cp <= tp1_p)
            if hit_tp1:
                tp1_qty = float(t["tp1_qty"])
                rem_qty = max(0.0, float(t["qty"]) - tp1_qty)
                g1 = (tp1_p - ep) * tp1_qty if is_long else (ep - tp1_p) * tp1_qty
                f1 = tp1_qty * tp1_p * maker_fee
                part_net = g1 - f1
                self.journal.update_trade(
                    tid,
                    {
                        "status": "TP1_HIT",
                        "sl_price": be_p,
                        "remaining_qty": rem_qty,
                        "realized_tp1_pnl": round(part_net, 4),
                        "peak_price": cp,
                    },
                )
                self.notifier.send(
                    f"🎯 [PAPER] TP1 (55%) {sym} по {tp1_p} (+${part_net:.2f})!\n"
                    f"SL переведён в БЕЗУБЫТОК (+0.15%): {be_p}"
                )
                return

        # 4. Check TP2 (45%) & optional trailing after TP1
        if t["status"] == "TP1_HIT":
            hit_tp2 = (cp >= tp2_p) if is_long else (cp <= tp2_p)
            if hit_tp2:
                rem_qty = float(t["remaining_qty"])
                g2 = (tp2_p - ep) * rem_qty if is_long else (ep - tp2_p) * rem_qty
                entry_fee = float(t["qty"]) * ep * maker_fee
                f2 = rem_qty * tp2_p * maker_fee
                tp1_realized = float(t.get("realized_tp1_pnl") or 0.0)
                net_pnl = tp1_realized + g2 - entry_fee - f2
                net_pct = (net_pnl / max(float(t["margin_usdt"]), 1e-6)) * 100.0
                self.journal.update_trade(
                    tid,
                    {
                        "status": "CLOSED",
                        "exit_time": utc_now_iso(),
                        "remaining_qty": 0.0,
                        "exit_reason": "TP2_HIT",
                        "gross_pnl_usdt": round(tp1_realized + g2, 2),
                        "fees_usdt": round(entry_fee + f2, 2),
                        "net_pnl_usdt": round(net_pnl, 2),
                        "net_pnl_pct": round(net_pct, 2),
                    },
                )
                eq = self.get_current_equity()
                self.journal.update_daily_stats(eq, realized_delta=net_pnl, closed_win=True)
                self.notifier.send(
                    f"🏆 [PAPER] TP2 {sym} ({t['direction']}) по {tp2_p}!\n"
                    f"Итого Net PnL: +${net_pnl:.2f} (+{net_pct:.1f}% к марже) | Баланс: ${eq:.2f}"
                )
                return

            # Optional ATR trailing SL update on remaining 45%
            if int(t.get("use_trailing") or 0) == 1:
                peak = float(t["peak_price"])
                tick = spec.tick_size if spec else 0.0001
                trail_dist = ep * (float(t["sl_pct"]) / 100.0) * 1.4
                if is_long and cp > peak:
                    new_sl = round_tick(cp - trail_dist, tick)
                    updates: Dict[str, Any] = {"peak_price": cp}
                    if new_sl > sl_p:
                        updates["sl_price"] = new_sl
                    self.journal.update_trade(tid, updates)
                elif (not is_long) and cp < peak:
                    new_sl = round_tick(cp + trail_dist, tick)
                    updates = {"peak_price": cp}
                    if new_sl < sl_p:
                        updates["sl_price"] = new_sl
                    self.journal.update_trade(tid, updates)

    def _manage_exchange_trade(
        self, t: Dict[str, Any], tickers: Dict[str, Dict[str, Any]], spec: Optional[InstrumentSpec]
    ) -> None:
        sym = t["symbol"]
        tid = int(t["id"])
        is_long = t["direction"] == "LONG"

        try:
            positions = self.client.get_positions(sym)
        except Exception as exc:
            logger.warning("get_positions error for %s: %s", sym, exc)
            return

        pos = positions[0] if positions else None
        pos_size = float(pos.get("size", 0.0)) if pos else 0.0

        # 1. PENDING_LIMIT -> check fill or TTL expiry
        if t["status"] == "PENDING_LIMIT":
            if pos_size > 0:
                avg_p = float(pos.get("avgPrice") or t["entry_price"])
                self.journal.update_trade(tid, {"status": "OPEN", "entry_price": avg_p, "remaining_qty": pos_size})
                t["entry_price"] = avg_p
                t["qty"] = pos_size
                self._place_exchange_tps(tid, t, spec)
                self.notifier.send(f"✅ [{self.client.mode.upper()}] Вход {sym} исполнен по {avg_p}, TP1/TP2 выставлены.")
                return
            exp = t.get("expires_at_utc")
            if exp:
                exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
                if datetime.now(timezone.utc) >= exp_dt:
                    if t.get("entry_order_id"):
                        self.client.cancel_order(sym, str(t["entry_order_id"]))
                    self.journal.update_trade(tid, {"status": "CLOSED", "exit_time": utc_now_iso(), "exit_reason": "LIMIT_EXPIRED"})
            return

        # 2. Position fully closed on exchange (SL or TP2 hit)
        if pos_size <= 0:
            try:
                self.client.cancel_all_orders(sym)
            except Exception:
                pass
            entry_ms = int(datetime.fromisoformat(str(t["entry_time"]).replace("Z", "+00:00")).timestamp() * 1000) - 60_000
            net_pnl = 0.0
            try:
                cpnls = self.client.get_closed_pnl(sym, start_ms=entry_ms, limit=10)
                net_pnl = sum(float(c.get("closedPnl") or 0.0) for c in cpnls)
            except Exception:
                pass
            net_pct = (net_pnl / max(float(t["margin_usdt"]), 1e-6)) * 100.0
            self.journal.update_trade(
                tid,
                {
                    "status": "CLOSED",
                    "exit_time": utc_now_iso(),
                    "remaining_qty": 0.0,
                    "exit_reason": "EXCHANGE_CLOSED",
                    "net_pnl_usdt": round(net_pnl, 2),
                    "net_pnl_pct": round(net_pct, 2),
                },
            )
            eq = self.get_current_equity()
            self.journal.update_daily_stats(eq, realized_delta=net_pnl, closed_win=(net_pnl > 0))
            ico = "🏆" if net_pnl >= 0 else "❌"
            self.notifier.send(
                f"{ico} [{self.client.mode.upper()}] ЗАКРЫТА {sym} ({t['direction']})\n"
                f"Net PnL: ${net_pnl:+.2f} ({net_pct:+.1f}% к марже) | Баланс: ${eq:.2f}"
            )
            return

        # 3. Detect TP1 fill -> IMMEDIATELY move server-side SL to breakeven +0.15%
        if t["status"] == "OPEN" and pos_size < float(t["qty"]) * 0.70:
            be_p = float(t["be_price"])
            try:
                self.client.set_trading_stop(sym, stop_loss=be_p)
                self.journal.update_trade(tid, {"status": "TP1_HIT", "sl_price": be_p, "remaining_qty": pos_size})
                self.notifier.send(
                    f"🎯 [{self.client.mode.upper()}] TP1 (55%) {sym} исполнен!\n"
                    f"Серверный SL переведён в БЕЗУБЫТОК (+0.15%): {be_p}"
                )
            except Exception as exc:
                logger.error("Failed moving SL to BE for %s: %s", sym, exc)
            return

        # 4. Optional trailing stop update on exchange after TP1_HIT
        if t["status"] == "TP1_HIT" and int(t.get("use_trailing") or 0) == 1:
            cp = float(tickers.get(sym, {}).get("last_price", 0.0))
            if cp <= 0:
                return
            peak = float(t["peak_price"])
            sl_p = float(t["sl_price"])
            tick = spec.tick_size if spec else 0.0001
            trail_dist = float(t["entry_price"]) * (float(t["sl_pct"]) / 100.0) * 1.4
            if is_long and cp > peak:
                new_sl = round_tick(cp - trail_dist, tick)
                if new_sl > sl_p:
                    try:
                        self.client.set_trading_stop(sym, stop_loss=new_sl)
                        self.journal.update_trade(tid, {"sl_price": new_sl, "peak_price": cp})
                    except Exception:
                        pass
            elif (not is_long) and cp < peak:
                new_sl = round_tick(cp + trail_dist, tick)
                if new_sl < sl_p:
                    try:
                        self.client.set_trading_stop(sym, stop_loss=new_sl)
                        self.journal.update_trade(tid, {"sl_price": new_sl, "peak_price": cp})
                    except Exception:
                        pass
