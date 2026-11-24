"""
Isolated SQLite State Persistence for AI Trader v2 (`ai_trader_v2/data/ai_trades.db`).
Tracks:
  - `trades`: full lifecycle (PENDING_LIMIT -> OPEN -> TP1_HIT -> CLOSED), SL/TP1/TP2/BE, PnL, AI rationale
  - `ai_decisions`: every candidate evaluated by AI Judge (ENTER and SKIP) for transparent auditing
  - `daily_stats`: UTC daily equity snapshots, realized PnL, and -6.0% daily halt flag
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(os.getenv("AI_TRADER_DATA_DIR", str(ROOT_DIR / "data" / "ai_trader_v2"))) / "ai_trades.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class AITradeJournal:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                    init_sl_price REAL NOT NULL,
                    tp1_price REAL NOT NULL,
                    tp2_price REAL NOT NULL,
                    be_price REAL NOT NULL,
                    use_trailing INTEGER DEFAULT 0,
                    peak_price REAL NOT NULL,
                    qty REAL NOT NULL,
                    tp1_qty REAL NOT NULL,
                    remaining_qty REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    margin_usdt REAL NOT NULL,
                    notional_usdt REAL NOT NULL,
                    sl_pct REAL NOT NULL,
                    tp1_r REAL NOT NULL,
                    tp2_r REAL NOT NULL,
                    primary_school TEXT NOT NULL,
                    schools_json TEXT NOT NULL,
                    ai_confidence INTEGER NOT NULL,
                    ai_reason TEXT,
                    entry_order_id TEXT,
                    tp1_order_id TEXT,
                    tp2_order_id TEXT,
                    exit_reason TEXT,
                    realized_tp1_pnl REAL DEFAULT 0.0,
                    gross_pnl_usdt REAL DEFAULT 0.0,
                    fees_usdt REAL DEFAULT 0.0,
                    net_pnl_usdt REAL DEFAULT 0.0,
                    net_pnl_pct REAL DEFAULT 0.0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decided_at TEXT NOT NULL,
                    bar_ts INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence INTEGER NOT NULL,
                    primary_school TEXT NOT NULL,
                    schools_json TEXT NOT NULL,
                    price REAL NOT NULL,
                    sl_pct REAL NOT NULL,
                    tp1_r REAL NOT NULL,
                    tp2_r REAL NOT NULL,
                    regime TEXT,
                    provider TEXT,
                    reason_ru TEXT,
                    warnings_json TEXT,
                    guardrail_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date_utc TEXT PRIMARY KEY,
                    start_equity REAL NOT NULL,
                    current_equity REAL NOT NULL,
                    realized_pnl_usdt REAL DEFAULT 0.0,
                    trades_opened INTEGER DEFAULT 0,
                    trades_closed INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    halted INTEGER DEFAULT 0,
                    halt_reason TEXT
                )
                """
            )

    # ------------------------------------------------------------------
    # AI Decisions Log
    # ------------------------------------------------------------------
    def log_ai_decision(
        self,
        bar_ts: int,
        symbol: str,
        direction: str,
        decision: str,
        confidence: int,
        primary_school: str,
        fired_schools: List[str],
        price: float,
        sl_pct: float,
        tp1_r: float,
        tp2_r: float,
        regime: str,
        provider: str,
        reason_ru: str,
        warnings: List[str],
        guardrail_notes: List[str],
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO ai_decisions (
                    decided_at, bar_ts, symbol, direction, decision, confidence,
                    primary_school, schools_json, price, sl_pct, tp1_r, tp2_r,
                    regime, provider, reason_ru, warnings_json, guardrail_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now_iso(),
                    int(bar_ts),
                    symbol,
                    direction,
                    decision,
                    int(confidence),
                    primary_school,
                    json.dumps(fired_schools, ensure_ascii=False),
                    float(price),
                    float(sl_pct),
                    float(tp1_r),
                    float(tp2_r),
                    regime,
                    provider,
                    reason_ru,
                    json.dumps(warnings, ensure_ascii=False),
                    json.dumps(guardrail_notes, ensure_ascii=False),
                ),
            )
            return int(cur.lastrowid)

    def has_decided_bar(self, symbol: str, bar_ts: int) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ai_decisions WHERE symbol = ? AND bar_ts = ? LIMIT 1",
                (symbol, int(bar_ts)),
            ).fetchone()
            return row is not None

    # ------------------------------------------------------------------
    # Trades CRUD
    # ------------------------------------------------------------------
    def insert_trade(self, trade: Dict[str, Any]) -> int:
        cols = list(trade.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_sql = ", ".join(cols)
        with self._connect() as conn:
            cur = conn.execute(
                f"INSERT INTO trades ({col_sql}) VALUES ({placeholders})",
                [trade[c] for c in cols],
            )
            return int(cur.lastrowid)

    def update_trade(self, trade_id: int, updates: Dict[str, Any]) -> None:
        if not updates:
            return
        set_sql = ", ".join(f"{k} = ?" for k in updates.keys())
        vals = list(updates.values()) + [int(trade_id)]
        with self._connect() as conn:
            conn.execute(f"UPDATE trades SET {set_sql} WHERE id = ?", vals)

    def get_active_trades(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE status IN ('PENDING_LIMIT', 'OPEN', 'TP1_HIT') ORDER BY id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_recent_closed_time_for_symbol(self, symbol: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT exit_time FROM trades WHERE symbol = ? AND status = 'CLOSED' ORDER BY id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            return str(row["exit_time"]) if row and row["exit_time"] else None

    # ------------------------------------------------------------------
    # Daily Stats & Equity Tracking
    # ------------------------------------------------------------------
    def ensure_daily_row(self, equity: float, date_utc: Optional[str] = None) -> Dict[str, Any]:
        d = date_utc or utc_today_str()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM daily_stats WHERE date_utc = ?", (d,)).fetchone()
            if row is not None:
                return dict(row)
            conn.execute(
                """
                INSERT INTO daily_stats (date_utc, start_equity, current_equity)
                VALUES (?, ?, ?)
                """,
                (d, float(equity), float(equity)),
            )
            row = conn.execute("SELECT * FROM daily_stats WHERE date_utc = ?", (d,)).fetchone()
            return dict(row)

    def update_daily_stats(
        self,
        current_equity: float,
        realized_delta: float = 0.0,
        opened_delta: int = 0,
        closed_win: Optional[bool] = None,
        halted: Optional[bool] = None,
        halt_reason: Optional[str] = None,
        date_utc: Optional[str] = None,
    ) -> Dict[str, Any]:
        d = date_utc or utc_today_str()
        row = self.ensure_daily_row(current_equity, d)
        new_realized = float(row["realized_pnl_usdt"]) + realized_delta
        new_opened = int(row["trades_opened"]) + opened_delta
        new_closed = int(row["trades_closed"]) + (1 if closed_win is not None else 0)
        new_wins = int(row["wins"]) + (1 if closed_win is True else 0)
        new_losses = int(row["losses"]) + (1 if closed_win is False else 0)
        new_halted = int(row["halted"]) if halted is None else (1 if halted else 0)
        new_reason = row["halt_reason"] if halt_reason is None else halt_reason

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE daily_stats
                SET current_equity = ?, realized_pnl_usdt = ?, trades_opened = ?,
                    trades_closed = ?, wins = ?, losses = ?, halted = ?, halt_reason = ?
                WHERE date_utc = ?
                """,
                (
                    float(current_equity),
                    round(new_realized, 4),
                    new_opened,
                    new_closed,
                    new_wins,
                    new_losses,
                    new_halted,
                    new_reason,
                    d,
                ),
            )
            updated = conn.execute("SELECT * FROM daily_stats WHERE date_utc = ?", (d,)).fetchone()
            return dict(updated)

    def get_paper_equity(self, start_equity: float = 2000.0) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(net_pnl_usdt), 0.0) AS total_pnl FROM trades WHERE mode = 'paper' AND status = 'CLOSED'"
            ).fetchone()
            total_closed = float(row["total_pnl"]) if row else 0.0
            row2 = conn.execute(
                "SELECT COALESCE(SUM(realized_tp1_pnl), 0.0) AS part_pnl FROM trades WHERE mode = 'paper' AND status = 'TP1_HIT'"
            ).fetchone()
            total_part = float(row2["part_pnl"]) if row2 else 0.0
            return round(start_equity + total_closed + total_part, 2)

    def get_summary_report(self, paper_start_eq: float = 2000.0) -> Dict[str, Any]:
        with self._connect() as conn:
            closed = conn.execute("SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY id ASC").fetchall()
            active = conn.execute(
                "SELECT * FROM trades WHERE status IN ('PENDING_LIMIT', 'OPEN', 'TP1_HIT') ORDER BY id ASC"
            ).fetchall()
            dec_tot = conn.execute(
                "SELECT decision, COUNT(*) AS cnt FROM ai_decisions GROUP BY decision"
            ).fetchall()

        pnls = [float(r["net_pnl_usdt"]) for r in closed]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        gp = sum(wins)
        gl = -sum(losses)
        pf = round(gp / gl, 2) if gl > 1e-9 else (99.0 if gp > 0 else 0.0)
        dec_map = {str(r["decision"]): int(r["cnt"]) for r in dec_tot}

        return {
            "paper_equity_usdt": self.get_paper_equity(paper_start_eq),
            "closed_trades": len(closed),
            "active_trades": len(active),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100.0, 1) if closed else 0.0,
            "net_pnl_usdt": round(sum(pnls), 2),
            "profit_factor": pf,
            "ai_enter_count": dec_map.get("ENTER", 0),
            "ai_skip_count": dec_map.get("SKIP", 0),
            "active_positions": [dict(r) for r in active],
        }

    def get_recent_ai_decisions(self, limit: int = 8) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ai_decisions ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_period_report(self, days: int = 1) -> Dict[str, Any]:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._connect() as conn:
            closed = conn.execute(
                "SELECT * FROM trades WHERE status = 'CLOSED' AND exit_time >= ? ORDER BY id ASC",
                (cutoff,),
            ).fetchall()
            decs = conn.execute(
                "SELECT decision, COUNT(*) AS cnt FROM ai_decisions WHERE decided_at >= ? GROUP BY decision",
                (cutoff,),
            ).fetchall()

        closed_list = [dict(r) for r in closed]
        real_closed = [r for r in closed_list if r.get("exit_reason") != "LIMIT_EXPIRED"]
        expired = len(closed_list) - len(real_closed)
        pnls = [float(r["net_pnl_usdt"]) for r in real_closed]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x < 0]
        gp = sum(wins)
        gl = -sum(losses)
        pf = round(gp / gl, 2) if gl > 1e-9 else (99.0 if gp > 0 else 0.0)
        tp2_cnt = sum(1 for r in real_closed if r.get("exit_reason") == "TP2_HIT")
        be_cnt = sum(1 for r in real_closed if r.get("exit_reason") == "BE_OR_TRAIL_SL")
        sl_cnt = sum(1 for r in real_closed if r.get("exit_reason") == "SL_HIT")
        ex_cnt = sum(1 for r in real_closed if r.get("exit_reason") == "EXCHANGE_CLOSED")
        dec_map = {str(r["decision"]): int(r["cnt"]) for r in decs}
        return {
            "days": days,
            "closed_trades": len(real_closed),
            "expired_limits": expired,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(len(wins) / len(real_closed) * 100.0, 1) if real_closed else 0.0,
            "net_pnl_usdt": round(sum(pnls), 2),
            "fees_usdt": round(sum(float(r.get("fees_usdt") or 0.0) for r in real_closed), 2),
            "profit_factor": pf,
            "tp2_hits": tp2_cnt,
            "be_trail_hits": be_cnt,
            "sl_hits": sl_cnt,
            "exchange_closed": ex_cnt,
            "ai_enter": dec_map.get("ENTER", 0),
            "ai_skip": dec_map.get("SKIP", 0),
            "recent_closed": real_closed[-5:],
        }

    def get_school_stats(self) -> Dict[str, Dict[str, Any]]:
        stats: Dict[str, Dict[str, Any]] = {}
        with self._connect() as conn:
            decs = conn.execute("SELECT primary_school, decision FROM ai_decisions").fetchall()
            closed = conn.execute(
                "SELECT primary_school, net_pnl_usdt FROM trades WHERE status = 'CLOSED' AND exit_reason != 'LIMIT_EXPIRED'"
            ).fetchall()
        for r in decs:
            sc = str(r["primary_school"])
            s = stats.setdefault(sc, {"signals": 0, "enter": 0, "skip": 0, "closed": 0, "wins": 0, "pnl": 0.0})
            s["signals"] += 1
            if r["decision"] == "ENTER":
                s["enter"] += 1
            else:
                s["skip"] += 1
        for r in closed:
            sc = str(r["primary_school"])
            s = stats.setdefault(sc, {"signals": 0, "enter": 0, "skip": 0, "closed": 0, "wins": 0, "pnl": 0.0})
            pnl = float(r["net_pnl_usdt"] or 0.0)
            s["closed"] += 1
            if pnl > 0:
                s["wins"] += 1
            s["pnl"] = round(s["pnl"] + pnl, 2)
        return stats

