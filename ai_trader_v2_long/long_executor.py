"""
L2_NEWS_LONG Executor: inherits the full certified L2_news VikaExecutor
(15x Isolated, 5x12%, Cap $5k/$2.5k storm, 2-Step DCA 35/65, BE +0.30% with Leg-2 DCA cancel,
Hard-SL-only AC 6h & Probation, strict <=85% margin reserve, US Macro Blackout).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from ai_trader_v2.executor import VikaExecutor
from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime
from ai_trader_v2.macro_calendar import MacroCalendarGuard
from ai_trader_v2_long.long_engine import LongCoinEvaluation
from bot.bybit_client import BybitClient


class VikaLongExecutor(VikaExecutor):
    def __init__(
        self,
        cfg: dict[str, Any],
        client: BybitClient,
        journal: VikaJournal,
        notifier: Optional[Callable[[str], None]] = None,
        macro_guard: Optional[MacroCalendarGuard] = None,
    ) -> None:
        super().__init__(cfg=cfg, client=client, journal=journal, notifier=notifier, macro_guard=macro_guard)
        self.max_entries_per_bar = self.max_entries_per_bar_l

    def open_signal(  # type: ignore[override]
        self,
        ev: LongCoinEvaluation,
        *,
        btc_regime: Optional[BtcL2Regime] = None,
    ) -> Optional[int]:
        if ev.final_signal != "LONG":
            return None
        return super().open_signal(ev, btc_regime=btc_regime)

    def check_breakeven_on_closed_bar_with_wick(
        self,
        sym: str,
        closed_bar_ts: int,
        closed_close: float,
        closed_high: float = 0.0,
    ) -> bool:
        """L2_news standardized closed-candle BE (+1.00% -> +0.30%) with synchronous Leg-2 DCA cancel."""
        return self.check_breakeven_on_closed_bar(sym, closed_bar_ts, closed_close)
