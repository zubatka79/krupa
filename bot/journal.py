"""Signal journal: every setup the bot saw on a closed bar and what happened to it.

Rows live in the bot's trades DB (table `signals`), so one file per bot holds both
the trades and the decisions behind them. Outcomes:
  SCAN_BLOCKED - the whole scan was skipped (macro shield, pause, BTC shock)
  FILTERED     - the strategy saw a setup but a filter rejected it
  OPENED       - an entry order was placed (trade_id points to trades.id)
  REJECTED     - the executor refused the entry (reason holds the rule)
  ERROR        - placing the entry raised
Journal failures are logged and never stop trading.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    bar_time_utc TEXT NOT NULL,
    strategy TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    entry_price REAL,
    score REAL,
    trade_id INTEGER,
    details_json TEXT
)
"""


def _iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SignalJournal:
    def __init__(self, db_path: str | Path, strategy: str) -> None:
        self.db_path = Path(db_path)
        self.strategy = strategy
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=15) as conn:
            conn.execute(SCHEMA)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_bar ON signals(bar_time_utc)")

    def record(
        self,
        *,
        bar_ts_ms: int,
        symbol: str,
        outcome: str,
        direction: str = "",
        reason: str = "",
        entry_price: Optional[float] = None,
        score: Optional[float] = None,
        trade_id: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            bar_time = _iso(datetime.fromtimestamp(int(bar_ts_ms) / 1000, tz=timezone.utc))
            with sqlite3.connect(self.db_path, timeout=15) as conn:
                conn.execute(
                    """
                    INSERT INTO signals (logged_at, bar_time_utc, strategy, symbol, direction, outcome, reason,
                                         entry_price, score, trade_id, details_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _iso(datetime.now(timezone.utc)),
                        bar_time,
                        self.strategy,
                        symbol,
                        direction,
                        outcome,
                        reason,
                        entry_price,
                        score,
                        trade_id,
                        json.dumps(details or {}, ensure_ascii=False, default=str),
                    ),
                )
        except Exception as exc:
            logger.warning("Signal journal write failed (%s %s %s): %s", symbol, outcome, reason, exc)
