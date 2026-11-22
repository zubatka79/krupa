"""Risk management, dynamic 3x-10x leverage sizing, portfolio guards, and Bybit fee math."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bot.bybit_client import InstrumentSpec, round_step_down, round_tick


@dataclass
class PositionPlan:
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    sl_price: float
    sl_pct: float
    tp1_price: float
    tp2_price: float
    be_price: float
    leverage: int
    margin_usdt: float
    notional_usdt: float
    qty: float
    tp1_qty: float
    tp2_qty: float
    estimated_risk_usdt: float
    estimated_risk_pct: float


class RiskManager:
    """Enforces Golden Middle risk parameters, leverage bounds, daily stop, and split feasibility."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def update_config(self, config: Dict[str, Any]) -> None:
        self.config = config

    @property
    def margin_pct(self) -> float:
        return float(self.config.get("margin_pct_per_trade", 15.0))

    @property
    def max_risk_pct(self) -> float:
        return float(self.config.get("max_risk_pct_per_trade", 2.5))

    @property
    def min_leverage(self) -> int:
        return int(self.config.get("min_leverage", 3))

    @property
    def max_leverage(self) -> int:
        return int(self.config.get("max_leverage", 10))

    @property
    def max_open_positions(self) -> int:
        return int(self.config.get("max_open_positions", 3))

    @property
    def max_same_direction(self) -> int:
        return int(self.config.get("max_same_direction", 2))

    @property
    def daily_loss_limit_pct(self) -> float:
        return float(self.config.get("daily_loss_limit_pct", 6.0))

    @property
    def maker_fee_rate(self) -> float:
        return float(self.config.get("maker_fee_pct", 0.02)) / 100.0

    @property
    def taker_fee_rate(self) -> float:
        return float(self.config.get("taker_fee_pct", 0.055)) / 100.0

    # ------------------------------------------------------------------
    # Portfolio & Daily Circuit-Breaker Checks
    # ------------------------------------------------------------------
    def is_daily_loss_halted(self, day_start_equity: float, current_equity: float) -> Tuple[bool, float]:
        """Return (is_halted, drawdown_pct_positive) for the current UTC day."""
        if day_start_equity <= 0:
            return True, 100.0
        dd_pct = ((day_start_equity - current_equity) / day_start_equity) * 100.0
        return dd_pct >= self.daily_loss_limit_pct, max(0.0, dd_pct)

    def can_open_position(
        self,
        symbol: str,
        direction: str,
        open_positions: List[Dict[str, Any]],
        day_start_equity: float,
        current_equity: float,
    ) -> Tuple[bool, str]:
        halted, dd_pct = self.is_daily_loss_halted(day_start_equity, current_equity)
        if halted:
            return False, f"Daily loss limit reached ({dd_pct:.2f}% >= {self.daily_loss_limit_pct:.2f}%)"

        if len(open_positions) >= self.max_open_positions:
            return False, f"Max open positions reached ({len(open_positions)}/{self.max_open_positions})"

        for pos in open_positions:
            if pos.get("symbol") == symbol:
                return False, f"Position already open for {symbol}"

        same_dir = sum(1 for p in open_positions if p.get("direction") == direction)
        if same_dir >= self.max_same_direction:
            return False, f"Max {direction} positions reached ({same_dir}/{self.max_same_direction})"

        return True, "OK"

    # ------------------------------------------------------------------
    # Minimum Order & TP1/TP2 Split Feasibility Check
    # ------------------------------------------------------------------
    def check_split_feasibility(
        self,
        notional_usdt: float,
        price: float,
        spec: Optional[InstrumentSpec],
    ) -> Tuple[bool, float, float, float, str]:
        """
        Verify total qty can be cleanly split into TP1 (55%) and TP2 (45%)
        while satisfying Bybit min_order_qty and min_notional rules.
        Returns (ok, total_qty, tp1_qty, tp2_qty, reason).
        """
        if price <= 0 or notional_usdt <= 0:
            return False, 0.0, 0.0, 0.0, "Invalid price or notional"

        tp1_ratio = float(self.config.get("tp1_close_ratio", 0.55))
        if spec is None:
            qty = notional_usdt / price
            tp1_qty = qty * tp1_ratio
            tp2_qty = qty - tp1_qty
            return True, qty, tp1_qty, tp2_qty, "OK"

        raw_qty = notional_usdt / price
        qty = round_step_down(raw_qty, spec.qty_step)
        if qty < spec.min_order_qty:
            return False, 0.0, 0.0, 0.0, f"Qty {qty} < minOrderQty {spec.min_order_qty}"

        tp1_qty = round_step_down(qty * tp1_ratio, spec.qty_step)
        tp2_qty = round_step_down(qty - tp1_qty, spec.qty_step)
        # Adjust total qty to exact sum of tp1_qty + tp2_qty
        qty = round_step_down(tp1_qty + tp2_qty, spec.qty_step)

        if tp1_qty < spec.min_order_qty or tp2_qty < spec.min_order_qty:
            return (
                False,
                0.0,
                0.0,
                0.0,
                f"Cannot split safely into TP1 ({tp1_qty}) / TP2 ({tp2_qty}) >= minOrderQty ({spec.min_order_qty})",
            )

        min_notional_leg = max(spec.min_notional, 5.0)
        if tp1_qty * price < min_notional_leg or tp2_qty * price < min_notional_leg:
            # Reduce-only TP legs on Bybit allow closing below minNotional in some cases,
            # but we require total order >= 2 * min_order_usdt and each leg >= min_order_qty.
            if qty * price < max(spec.min_order_usdt * 2.0, 10.0):
                return (
                    False,
                    0.0,
                    0.0,
                    0.0,
                    f"Notional {qty * price:.2f} USDT too small for safe 55/45 split",
                )

        return True, qty, tp1_qty, tp2_qty, "OK"

    # ------------------------------------------------------------------
    # Dynamic Leverage (3x-10x) & Position Sizing
    # ------------------------------------------------------------------
    def build_position_plan(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_pct: float,
        equity: float,
        spec: Optional[InstrumentSpec] = None,
    ) -> Tuple[Optional[PositionPlan], str]:
        """Compute exact leverage (3x-10x), margin, notional, SL, TP1, TP2, BE prices."""
        if equity <= 0 or entry_price <= 0 or sl_pct <= 0:
            return None, "Invalid equity, entry_price, or sl_pct"

        target_margin = equity * (self.margin_pct / 100.0)
        max_risk_usdt = equity * (self.max_risk_pct / 100.0)

        # Effective loss fraction on full SL hit includes entry Taker fee + SL exit Taker fee
        sl_frac = sl_pct / 100.0
        eff_loss_frac = sl_frac + 2.0 * self.taker_fee_rate
        max_notional_by_risk = max_risk_usdt / eff_loss_frac

        raw_leverage = max_notional_by_risk / max(target_margin, 1e-9)
        sym_max_lev = int(spec.max_leverage) if spec is not None else self.max_leverage
        effective_max_lev = min(self.max_leverage, max(self.min_leverage, sym_max_lev))
        leverage = int(max(self.min_leverage, min(effective_max_lev, math.floor(raw_leverage))))

        # Cap notional so actual risk NEVER exceeds max_risk_pct_per_trade
        notional = min(target_margin * leverage, max_notional_by_risk)
        margin_used = notional / leverage

        ok, qty, tp1_qty, tp2_qty, reason = self.check_split_feasibility(notional, entry_price, spec)
        if not ok:
            return None, reason

        actual_notional = qty * entry_price
        actual_margin = actual_notional / leverage
        est_risk_usdt = actual_notional * eff_loss_frac
        est_risk_pct = (est_risk_usdt / equity) * 100.0

        tp1_pct = float(self.config.get("tp1_pct", 0.8)) / 100.0
        tp2_pct = float(self.config.get("tp2_pct", 1.6)) / 100.0
        be_pct = float(self.config.get("breakeven_offset_pct", 0.15)) / 100.0

        if direction == "LONG":
            sl_price = entry_price * (1.0 - sl_frac)
            tp1_price = entry_price * (1.0 + tp1_pct)
            tp2_price = entry_price * (1.0 + tp2_pct)
            be_price = entry_price * (1.0 + be_pct)
        else:
            sl_price = entry_price * (1.0 + sl_frac)
            tp1_price = entry_price * (1.0 - tp1_pct)
            tp2_price = entry_price * (1.0 - tp2_pct)
            be_price = entry_price * (1.0 - be_pct)

        if spec is not None and spec.tick_size > 0:
            sl_price = round_tick(sl_price, spec.tick_size)
            tp1_price = round_tick(tp1_price, spec.tick_size)
            tp2_price = round_tick(tp2_price, spec.tick_size)
            be_price = round_tick(be_price, spec.tick_size)

        return (
            PositionPlan(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                sl_price=sl_price,
                sl_pct=sl_pct,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                be_price=be_price,
                leverage=leverage,
                margin_usdt=actual_margin,
                notional_usdt=actual_notional,
                qty=qty,
                tp1_qty=tp1_qty,
                tp2_qty=tp2_qty,
                estimated_risk_usdt=est_risk_usdt,
                estimated_risk_pct=est_risk_pct,
            ),
            "OK",
        )

    # ------------------------------------------------------------------
    # Exact Fee & Net PnL Accounting
    # ------------------------------------------------------------------
    def fee_usdt(self, notional: float, is_maker: bool) -> float:
        rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        return abs(notional) * rate

    @staticmethod
    def leg_gross_pnl(direction: str, entry_price: float, exit_price: float, qty: float) -> float:
        if direction == "LONG":
            return (exit_price - entry_price) * qty
        return (entry_price - exit_price) * qty
