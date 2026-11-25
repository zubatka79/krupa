"""Thread-safe SQLite trade & state journal for VIKA_N2_N7 (`data/ai_trader_v2/vika_trades.db`)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class VikaJournal:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    key_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    order_id TEXT,
                    entry_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    margin_usd REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    tp_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    be_trigger_price REAL NOT NULL,
                    be_lock_price REAL NOT NULL,
                    be_active INTEGER NOT NULL DEFAULT 0,
                    entry_bar_ts INTEGER NOT NULL,
                    expire_ts INTEGER NOT NULL DEFAULT 0,
                    opened_at INTEGER NOT NULL,
                    closed_at INTEGER,
                    exit_price REAL,
                    exit_reason TEXT,
                    pnl_usd REAL,
                    pnl_pct REAL,
                    filters_json TEXT
                );
                CREATE TABLE IF NOT EXISTS daily_state (
                    day_utc TEXT PRIMARY KEY,
                    start_equity REAL NOT NULL,
                    halted INTEGER NOT NULL DEFAULT 0,
                    halt_notified INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS kv_state (
                    k TEXT PRIMARY KEY,
                    v TEXT NOT NULL
                );
                """
            )

    # --- KV state (pause, cooldowns, last processed bar) ---
    def get_kv(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT v FROM kv_state WHERE k = ?", (key,)).fetchone()
            return row["v"] if row else default

    def set_kv(self, key: str, val: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO kv_state (k, v) VALUES (?, ?)", (key, str(val)))

    def is_paused(self) -> bool:
        return self.get_kv("paused", "0") == "1"

    def set_paused(self, paused: bool) -> None:
        self.set_kv("paused", "1" if paused else "0")

    def get_cooldown_until_ms(self, symbol: str) -> int:
        v = self.get_kv(f"cd_{symbol}", "0")
        return int(v or "0")

    def set_cooldown_until_ms(self, symbol: str, ts_ms: int) -> None:
        self.set_kv(f"cd_{symbol}", str(int(ts_ms)))

    def get_last_signal_bar_ts(self, symbol: str) -> int:
        v = self.get_kv(f"sigbar_{symbol}", "0")
        return int(v or "0")

    def set_last_signal_bar_ts(self, symbol: str, bar_ts: int) -> None:
        self.set_kv(f"sigbar_{symbol}", str(int(bar_ts)))

    # --- Daily UTC Risk State (-6.0%) ---
    @staticmethod
    def today_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def ensure_daily_state(self, current_equity: float) -> Dict[str, Any]:
        day = self.today_utc()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM daily_state WHERE day_utc = ?", (day,)).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO daily_state (day_utc, start_equity, halted, halt_notified) VALUES (?, ?, 0, 0)",
                    (day, float(current_equity)),
                )
                return {"day_utc": day, "start_equity": float(current_equity), "halted": False, "halt_notified": False}
            return {
                "day_utc": row["day_utc"],
                "start_equity": float(row["start_equity"]),
                "halted": bool(row["halted"]),
                "halt_notified": bool(row["halt_notified"]),
            }

    def mark_daily_halted(self, notified: bool = True) -> None:
        day = self.today_utc()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE daily_state SET halted = 1, halt_notified = ? WHERE day_utc = ?",
                (1 if notified else 0, day),
            )

    # --- Positions lifecycle ---
    def add_position(
        self,
        *,
        symbol: str,
        side: str,
        key_name: str,
        status: str,
        order_id: str,
        entry_price: float,
        qty: float,
        margin_usd: float,
        leverage: int,
        tp_price: float,
        sl_price: float,
        be_trigger_price: float,
        be_lock_price: float,
        entry_bar_ts: int,
        expire_ts: int,
        filters: Dict[str, Any],
    ) -> int:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO positions (
                    symbol, side, key_name, status, order_id,
                    entry_price, qty, margin_usd, leverage,
                    tp_price, sl_price, be_trigger_price, be_lock_price, be_active,
                    entry_bar_ts, expire_ts, opened_at, filters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    symbol, side, key_name, status, order_id,
                    float(entry_price), float(qty), float(margin_usd), int(leverage),
                    float(tp_price), float(sl_price), float(be_trigger_price), float(be_lock_price),
                    int(entry_bar_ts), int(expire_ts), now_ms, json.dumps(filters, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Return all PENDING or OPEN positions."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status IN ('PENDING', 'OPEN') ORDER BY id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_open_positions(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM positions WHERE status = 'OPEN' ORDER BY id ASC").fetchall()
            return [dict(r) for r in rows]

    def mark_open(self, pos_id: int, fill_price: float, fill_qty: float, margin_usd: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE positions SET status = 'OPEN', entry_price = ?, qty = ?, margin_usd = ? WHERE id = ?",
                (float(fill_price), float(fill_qty), float(margin_usd), int(pos_id)),
            )

    def mark_be_active(self, pos_id: int, new_sl: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE positions SET be_active = 1, sl_price = ? WHERE id = ?",
                (float(new_sl), int(pos_id)),
            )

    def mark_cancelled(self, pos_id: int, reason: str = "TTL_EXPIRED") -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE positions SET status = 'CANCELLED', closed_at = ?, exit_reason = ? WHERE id = ?",
                (now_ms, reason, int(pos_id)),
            )

    def mark_closed(
        self, pos_id: int, exit_price: float, exit_reason: str, pnl_usd: float, pnl_pct: float
    ) -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET status = 'CLOSED', closed_at = ?, exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?
                WHERE id = ?
                """,
                (now_ms, float(exit_price), exit_reason, float(pnl_usd), float(pnl_pct), int(pos_id)),
            )

    def get_recent_closed(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_total_realized_stats(self) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT pnl_usd, exit_reason FROM positions WHERE status = 'CLOSED'").fetchall()
            n = len(rows)
            pnl = sum(float(r["pnl_usd"] or 0.0) for r in rows)
            wins = sum(1 for r in rows if float(r["pnl_usd"] or 0.0) > 0)
            sl_n = sum(1 for r in rows if r["exit_reason"] == "SL")
            return {
                "trades": n,
                "pnl_usd": round(pnl, 2),
                "wr": round(wins / n * 100.0, 1) if n else 0.0,
                "sl": sl_n,
            }
