"""Thread-safe SQLite trade & state journal for L2_news (`data/ai_trader_v2/vika_trades.db`)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


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
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
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
                    reserved_margin_usd REAL NOT NULL DEFAULT 0.0,
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
                    filters_json TEXT,
                    dca_order_id TEXT DEFAULT '',
                    dca_price REAL NOT NULL DEFAULT 0.0,
                    dca_qty REAL NOT NULL DEFAULT 0.0,
                    dca_filled_qty REAL NOT NULL DEFAULT 0.0,
                    dca_margin_usd REAL NOT NULL DEFAULT 0.0,
                    dca_status TEXT NOT NULL DEFAULT 'NONE',
                    in_storm INTEGER NOT NULL DEFAULT 0,
                    tp_pct REAL NOT NULL DEFAULT 1.5
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
            # Safe forward-compatible migration for existing pre-L2_news SQLite databases
            existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(positions)").fetchall()}
            migrations: List[Tuple[str, str]] = [
                ("reserved_margin_usd", "REAL NOT NULL DEFAULT 0.0"),
                ("dca_order_id", "TEXT DEFAULT ''"),
                ("dca_price", "REAL NOT NULL DEFAULT 0.0"),
                ("dca_qty", "REAL NOT NULL DEFAULT 0.0"),
                ("dca_filled_qty", "REAL NOT NULL DEFAULT 0.0"),
                ("dca_margin_usd", "REAL NOT NULL DEFAULT 0.0"),
                ("dca_status", "TEXT NOT NULL DEFAULT 'NONE'"),
                ("in_storm", "INTEGER NOT NULL DEFAULT 0"),
                ("tp_pct", "REAL NOT NULL DEFAULT 1.5"),
            ]
            for col_name, col_def in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE positions ADD COLUMN {col_name} {col_def}")
            # Backfill reserved_margin_usd for legacy active rows
            conn.execute(
                "UPDATE positions SET reserved_margin_usd = margin_usd WHERE reserved_margin_usd <= 0 AND margin_usd > 0"
            )

    # --- KV state (pause, cooldowns, AC 6h, Coin Probation, last processed bar) ---
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

    def get_cooldown_until_ms(self, symbol: str, side: Optional[str] = None) -> int:
        if side:
            v = self.get_kv(f"cd_{symbol}_{side.upper()}", "")
            if v:
                return int(v)
        v = self.get_kv(f"cd_{symbol}", "0")
        return int(v or "0")

    def set_cooldown_until_ms(self, symbol: str, ts_ms: int, side: Optional[str] = None) -> None:
        if side:
            self.set_kv(f"cd_{symbol}_{side.upper()}", str(int(ts_ms)))
        else:
            self.set_kv(f"cd_{symbol}", str(int(ts_ms)))

    def get_last_signal_bar_ts(self, symbol: str, side: Optional[str] = None) -> int:
        if side:
            v = self.get_kv(f"sigbar_{symbol}_{side.upper()}", "0")
            return int(v or "0")
        v = self.get_kv(f"sigbar_{symbol}", "0")
        return int(v or "0")

    def set_last_signal_bar_ts(self, symbol: str, bar_ts: int, side: Optional[str] = None) -> None:
        if side:
            self.set_kv(f"sigbar_{symbol}_{side.upper()}", str(int(bar_ts)))
        else:
            self.set_kv(f"sigbar_{symbol}", str(int(bar_ts)))

    # --- Anti-Cascade 6h Pause (Hard SL ONLY, separated by LONG / SHORT) ---
    def get_ac_pause_until_ms(self, side: str) -> int:
        v = self.get_kv(f"ac_pause_{side.upper()}", "0")
        return int(v or "0")

    def set_ac_pause_until_ms(self, side: str, ts_ms: int) -> None:
        cur = self.get_ac_pause_until_ms(side)
        self.set_kv(f"ac_pause_{side.upper()}", str(max(cur, int(ts_ms))))

    # --- Coin Probation (2 Hard SLs within 21d -> 14d quarantine; Hard SL ONLY) ---
    def get_coin_probation_until_ms(self, symbol: str) -> int:
        v = self.get_kv(f"prob_until_{symbol}", "0")
        return int(v or "0")

    def record_coin_hard_sl(
        self,
        symbol: str,
        now_ms: int,
        *,
        window_days: int = 21,
        threshold: int = 2,
        pause_days: int = 14,
    ) -> Tuple[bool, int, int]:
        """
        Record a Hard SL for `symbol`. Returns (triggered_probation, sl_count_in_window, pause_until_ms).
        NEVER called on BE exits!
        """
        window_ms = int(window_days * 86_400_000)
        raw = self.get_kv(f"sl_hist_{symbol}", "[]") or "[]"
        try:
            hist = [int(x) for x in json.loads(raw) if (now_ms - int(x)) <= window_ms]
        except Exception:
            hist = []
        hist.append(int(now_ms))
        if len(hist) >= threshold:
            until_ms = max(self.get_coin_probation_until_ms(symbol), int(now_ms + pause_days * 86_400_000))
            self.set_kv(f"prob_until_{symbol}", str(until_ms))
            self.set_kv(f"sl_hist_{symbol}", "[]")
            return True, len(hist), until_ms
        self.set_kv(f"sl_hist_{symbol}", json.dumps(hist))
        return False, len(hist), 0

    def get_all_active_probations(self, now_ms: Optional[int] = None) -> List[Tuple[str, int]]:
        cur = int(now_ms if now_ms is not None else time.time() * 1000)
        out: List[Tuple[str, int]] = []
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT k, v FROM kv_state WHERE k LIKE 'prob_until_%'").fetchall()
            for r in rows:
                try:
                    until_ms = int(r["v"])
                    if until_ms > cur:
                        out.append((r["k"].replace("prob_until_", "", 1), until_ms))
                except Exception:
                    continue
        out.sort(key=lambda x: x[1])
        return out

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
        reserved_margin_usd: Optional[float] = None,
        dca_order_id: str = "",
        dca_price: float = 0.0,
        dca_qty: float = 0.0,
        dca_filled_qty: float = 0.0,
        dca_margin_usd: float = 0.0,
        dca_status: str = "NONE",
        in_storm: bool = False,
        tp_pct: float = 1.5,
    ) -> int:
        now_ms = int(time.time() * 1000)
        res_m = float(reserved_margin_usd if reserved_margin_usd is not None else margin_usd)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO positions (
                    symbol, side, key_name, status, order_id,
                    entry_price, qty, margin_usd, reserved_margin_usd, leverage,
                    tp_price, sl_price, be_trigger_price, be_lock_price, be_active,
                    entry_bar_ts, expire_ts, opened_at, filters_json,
                    dca_order_id, dca_price, dca_qty, dca_filled_qty, dca_margin_usd, dca_status, in_storm, tp_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol, side, key_name, status, order_id,
                    float(entry_price), float(qty), float(margin_usd), res_m, int(leverage),
                    float(tp_price), float(sl_price), float(be_trigger_price), float(be_lock_price),
                    int(entry_bar_ts), int(expire_ts), now_ms, json.dumps(filters, ensure_ascii=False),
                    str(dca_order_id), float(dca_price), float(dca_qty), float(dca_filled_qty), float(dca_margin_usd),
                    str(dca_status), 1 if in_storm else 0, float(tp_pct),
                ),
            )
            return int(cur.lastrowid)

    def get_active_positions(self) -> List[Dict[str, Any]]:
        """Return all PENDING_OPEN, PENDING, or OPEN positions."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status IN ('PENDING_OPEN', 'PENDING', 'OPEN') ORDER BY id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_pending_open_positions(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status = 'PENDING_OPEN' ORDER BY id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def finalize_pending_open(
        self,
        pos_id: int,
        *,
        order_id: str,
        entry_price: float,
        qty: float,
        margin_usd: float,
        reserved_margin_usd: float,
        tp_price: float,
        sl_price: float,
        be_trigger_price: float,
        be_lock_price: float,
        dca_order_id: str,
        dca_price: float,
        dca_qty: float,
        dca_margin_usd: float,
        dca_status: str,
        dca_filled_qty: float = 0.0,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET status = 'OPEN', order_id = ?, entry_price = ?, qty = ?, margin_usd = ?, reserved_margin_usd = ?,
                    tp_price = ?, sl_price = ?, be_trigger_price = ?, be_lock_price = ?,
                    dca_order_id = ?, dca_price = ?, dca_qty = ?, dca_margin_usd = ?, dca_status = ?, dca_filled_qty = ?
                WHERE id = ?
                """,
                (
                    str(order_id), float(entry_price), float(qty), float(margin_usd), float(reserved_margin_usd),
                    float(tp_price), float(sl_price), float(be_trigger_price), float(be_lock_price),
                    str(dca_order_id), float(dca_price), float(dca_qty), float(dca_margin_usd), str(dca_status),
                    float(dca_filled_qty), int(pos_id),
                ),
            )

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

    def update_open_fill(
        self,
        pos_id: int,
        fill_price: float,
        fill_qty: float,
        margin_usd: float,
        clear_order_id: bool = False,
    ) -> None:
        with self._lock, self._connect() as conn:
            if clear_order_id:
                conn.execute(
                    "UPDATE positions SET status = 'OPEN', entry_price = ?, qty = ?, margin_usd = ?, order_id = '' WHERE id = ?",
                    (float(fill_price), float(fill_qty), float(margin_usd), int(pos_id)),
                )
            else:
                conn.execute(
                    "UPDATE positions SET status = 'OPEN', entry_price = ?, qty = ?, margin_usd = ? WHERE id = ?",
                    (float(fill_price), float(fill_qty), float(margin_usd), int(pos_id)),
                )

    def update_dca_partial(
        self,
        pos_id: int,
        *,
        new_entry_price: float,
        new_qty: float,
        new_margin_usd: float,
        new_reserved_margin_usd: float,
        new_tp_price: float,
        new_be_trigger_price: float,
        new_be_lock_price: float,
        dca_filled_qty: float,
    ) -> None:
        """Record a partial Leg-2 DCA fill while the remaining DCA limit is still resting in the book."""
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET entry_price = ?, qty = ?, margin_usd = ?, reserved_margin_usd = ?,
                    tp_price = ?, be_trigger_price = ?, be_lock_price = ?,
                    dca_filled_qty = ?
                WHERE id = ?
                """,
                (
                    float(new_entry_price), float(new_qty), float(new_margin_usd), float(new_reserved_margin_usd),
                    float(new_tp_price), float(new_be_trigger_price), float(new_be_lock_price),
                    float(dca_filled_qty), int(pos_id),
                ),
            )

    def update_dca_filled(
        self,
        pos_id: int,
        *,
        new_entry_price: float,
        new_qty: float,
        new_margin_usd: float,
        new_tp_price: float,
        new_be_trigger_price: float,
        new_be_lock_price: float,
        dca_status: str = "FILLED",
        dca_filled_qty: Optional[float] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            if dca_filled_qty is not None:
                conn.execute(
                    """
                    UPDATE positions
                    SET entry_price = ?, qty = ?, margin_usd = ?, reserved_margin_usd = ?,
                        tp_price = ?, be_trigger_price = ?, be_lock_price = ?,
                        dca_order_id = '', dca_status = ?, dca_filled_qty = ?
                    WHERE id = ?
                    """,
                    (
                        float(new_entry_price), float(new_qty), float(new_margin_usd), float(new_margin_usd),
                        float(new_tp_price), float(new_be_trigger_price), float(new_be_lock_price),
                        str(dca_status), float(dca_filled_qty), int(pos_id),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE positions
                    SET entry_price = ?, qty = ?, margin_usd = ?, reserved_margin_usd = ?,
                        tp_price = ?, be_trigger_price = ?, be_lock_price = ?,
                        dca_order_id = '', dca_status = ?
                    WHERE id = ?
                    """,
                    (
                        float(new_entry_price), float(new_qty), float(new_margin_usd), float(new_margin_usd),
                        float(new_tp_price), float(new_be_trigger_price), float(new_be_lock_price),
                        str(dca_status), int(pos_id),
                    ),
                )

    def update_partial_tp_reduce(
        self,
        pos_id: int,
        *,
        new_qty: float,
        new_margin_usd: float,
        cancel_dca: bool = True,
    ) -> None:
        """Update remaining size/margin after a partial Limit TP fill and optionally retire pending DCA."""
        with self._lock, self._connect() as conn:
            if cancel_dca:
                conn.execute(
                    """
                    UPDATE positions
                    SET qty = ?, margin_usd = ?, reserved_margin_usd = ?,
                        dca_order_id = '',
                        dca_status = CASE WHEN dca_status = 'PENDING' THEN 'CANCELLED_TP_PARTIAL' ELSE dca_status END
                    WHERE id = ?
                    """,
                    (float(new_qty), float(new_margin_usd), float(new_margin_usd), int(pos_id)),
                )
            else:
                conn.execute(
                    "UPDATE positions SET qty = ?, margin_usd = ?, reserved_margin_usd = ? WHERE id = ?",
                    (float(new_qty), float(new_margin_usd), float(new_margin_usd), int(pos_id)),
                )

    def get_last_closed_at_ms(self, symbol: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(closed_at) AS max_ts FROM positions WHERE symbol = ? AND status = 'CLOSED'",
                (symbol,),
            ).fetchone()
            return int(row["max_ts"]) if (row and row["max_ts"] is not None) else 0

    def mark_dca_cancelled(self, pos_id: int, status: str, new_reserved_margin_usd: float) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE positions SET dca_order_id = '', dca_status = ?, reserved_margin_usd = ? WHERE id = ?",
                (str(status), float(new_reserved_margin_usd), int(pos_id)),
            )

    def mark_be_active(
        self,
        pos_id: int,
        new_sl: float,
        *,
        cancel_dca: bool = False,
        new_reserved_margin_usd: Optional[float] = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            if cancel_dca:
                if new_reserved_margin_usd is not None:
                    conn.execute(
                        """
                        UPDATE positions
                        SET be_active = 1, sl_price = ?, dca_order_id = '', dca_status = 'CANCELLED_BE', reserved_margin_usd = ?
                        WHERE id = ?
                        """,
                        (float(new_sl), float(new_reserved_margin_usd), int(pos_id)),
                    )
                else:
                    conn.execute(
                        "UPDATE positions SET be_active = 1, sl_price = ?, dca_order_id = '', dca_status = 'CANCELLED_BE' WHERE id = ?",
                        (float(new_sl), int(pos_id)),
                    )
            else:
                conn.execute(
                    "UPDATE positions SET be_active = 1, sl_price = ? WHERE id = ?",
                    (float(new_sl), int(pos_id)),
                )

    def mark_cancelled(self, pos_id: int, reason: str = "TTL_EXPIRED") -> None:
        now_ms = int(time.time() * 1000)
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE positions SET status = 'CANCELLED', closed_at = ?, exit_reason = ?, dca_order_id = '' WHERE id = ?",
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
                SET status = 'CLOSED', closed_at = ?, exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?, dca_order_id = ''
                WHERE id = ?
                """,
                (now_ms, float(exit_price), exit_reason, float(pnl_usd), float(pnl_pct), int(pos_id)),
            )

    def record_position_exit_atomic(
        self,
        *,
        pos_id: int,
        symbol: str,
        side: str,
        now_ms: int,
        exit_price: float,
        exit_reason: str,
        pnl_usd: float,
        pnl_pct: float,
        cooldown_until_ms: int,
        is_hard_sl: bool = False,
        ac_pause_until_ms: int = 0,
        probation_window_days: int = 21,
        probation_threshold: int = 2,
        probation_pause_days: int = 14,
    ) -> Tuple[bool, int, int]:
        """Atomically persist position close + cooldown + AC 6h + Coin Probation in ONE SQLite transaction."""
        side_u = side.upper()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE positions
                SET status = 'CLOSED', closed_at = ?, exit_price = ?, exit_reason = ?, pnl_usd = ?, pnl_pct = ?, dca_order_id = ''
                WHERE id = ?
                """,
                (int(now_ms), float(exit_price), str(exit_reason), float(pnl_usd), float(pnl_pct), int(pos_id)),
            )
            conn.execute(
                "INSERT OR REPLACE INTO kv_state (k, v) VALUES (?, ?)",
                (f"cd_{symbol}_{side_u}", str(int(cooldown_until_ms))),
            )
            if not is_hard_sl:
                return False, 0, 0

            ac_key = f"ac_pause_{side_u}"
            ac_row = conn.execute("SELECT v FROM kv_state WHERE k = ?", (ac_key,)).fetchone()
            cur_ac = int(ac_row["v"] or "0") if ac_row else 0
            conn.execute(
                "INSERT OR REPLACE INTO kv_state (k, v) VALUES (?, ?)",
                (ac_key, str(max(cur_ac, int(ac_pause_until_ms)))),
            )

            window_ms = int(probation_window_days * 86_400_000)
            sl_key = f"sl_hist_{symbol}"
            prob_key = f"prob_until_{symbol}"
            sl_row = conn.execute("SELECT v FROM kv_state WHERE k = ?", (sl_key,)).fetchone()
            raw_hist = sl_row["v"] if sl_row else "[]"
            try:
                hist = [int(x) for x in json.loads(raw_hist or "[]") if (now_ms - int(x)) <= window_ms]
            except Exception:
                hist = []
            hist.append(int(now_ms))

            if len(hist) >= probation_threshold:
                prob_row = conn.execute("SELECT v FROM kv_state WHERE k = ?", (prob_key,)).fetchone()
                cur_prob = int(prob_row["v"] or "0") if prob_row else 0
                until_ms = max(cur_prob, int(now_ms + probation_pause_days * 86_400_000))
                conn.execute("INSERT OR REPLACE INTO kv_state (k, v) VALUES (?, ?)", (prob_key, str(until_ms)))
                conn.execute("INSERT OR REPLACE INTO kv_state (k, v) VALUES (?, ?)", (sl_key, "[]"))
                return True, len(hist), until_ms

            conn.execute("INSERT OR REPLACE INTO kv_state (k, v) VALUES (?, ?)", (sl_key, json.dumps(hist)))
            return False, len(hist), 0

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
            sl_n = sum(1 for r in rows if r["exit_reason"] in ("SL", "HARD_SL"))
            be_n = sum(1 for r in rows if r["exit_reason"] == "BE")
            return {
                "trades": n,
                "total": n,
                "pnl_usd": round(pnl, 2),
                "wr": round(wins / n * 100.0, 1) if n else 0.0,
                "sl": sl_n,
                "be": be_n,
            }
