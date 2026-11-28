"""
Dedicated Telegram Interface for standalone L2_NEWS_LONG runner (if launched separately).
Reads LONG_TELEGRAM_BOT_TOKEN so it never conflicts with the main L2_news hybrid bot.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from ai_trader_v2.journal import VikaJournal
from ai_trader_v2.l2_market_data import BtcL2Regime
from ai_trader_v2.macro_calendar import MacroCalendarStatus
from ai_trader_v2.telegram_ui import VikaTelegramUI
from ai_trader_v2_long.long_engine import LongCoinEvaluation
from ai_trader_v2_long.long_executor import VikaLongExecutor

logger = logging.getLogger("vika_long.telegram")


class VikaLongTelegramUI(VikaTelegramUI):
    def __init__(
        self,
        cfg: Dict[str, Any],
        executor: VikaLongExecutor,
        journal: VikaJournal,
        get_latest_evals: Callable[[], Dict[str, LongCoinEvaluation]],
        get_btc_regime: Callable[[], Optional[BtcL2Regime]],
        get_macro_status: Optional[Callable[[], MacroCalendarStatus]] = None,
    ) -> None:
        cfg_view = dict(cfg)
        cfg_view["long_coins"] = list(cfg.get("coins", []))
        cfg_view["short_coins"] = []
        super().__init__(
            cfg=cfg_view,
            executor=executor,
            journal=journal,
            get_latest_evals=lambda: {},
            get_latest_long_evals=get_latest_evals,
            get_btc_regime=get_btc_regime,
            get_macro_status=get_macro_status,
        )
        # Override token/chat with dedicated LONG env vars so polling never collides
        self.token = os.getenv("LONG_TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = (
            os.getenv("LONG_TELEGRAM_CHAT_ID", "").strip()
            or os.getenv("TELEGRAM_CHAT_ID", "").strip()
        )
        raw_users = (
            os.getenv("LONG_TELEGRAM_ALLOWED_USER_IDS", "").strip()
            or os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").strip()
        )
        self.allowed_users = {u.strip() for u in raw_users.split(",") if u.strip()}
        self.enabled = bool(
            self.token
            and "your_telegram" not in self.token.lower()
            and (self.chat_id or self.allowed_users)
        )
        self.api_base = f"https://api.telegram.org/bot{self.token}" if self.enabled else ""
