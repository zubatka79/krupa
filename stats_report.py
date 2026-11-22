"""Forward-test report across all bot databases.

Usage (inside a bot container or locally next to data/):
    python stats_report.py                  # whole history
    python stats_report.py --since 2026-11-22
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return []


def _max_drawdown(pnls: List[float]) -> float:
    peak = equity = dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dd


def trade_stats(trades: List[sqlite3.Row]) -> Dict[str, float]:
    closed = [t for t in trades if t["status"] == "CLOSED"]
    pnls = [float(t["net_pnl_usdt"] or 0.0) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)
    filled = sum(1 for t in trades if t["status"] in ("OPEN", "CLOSED"))
    cancelled = sum(1 for t in trades if t["status"] == "CANCELLED")
    return {
        "orders": len(trades),
        "filled": filled,
        "cancelled": cancelled,
        "fill_rate": filled / (filled + cancelled) * 100 if filled + cancelled else 0.0,
        "closed": len(closed),
        "win_rate": len(wins) / len(closed) * 100 if closed else 0.0,
        "net": sum(pnls),
        "fees": sum(float(t["fees_usdt"] or 0.0) for t in closed),
        "avg_win": gross_win / len(wins) if wins else 0.0,
        "avg_loss": -gross_loss / len(losses) if losses else 0.0,
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        "max_dd": _max_drawdown(pnls),
        "best": max(pnls) if pnls else 0.0,
        "worst": min(pnls) if pnls else 0.0,
    }


def format_stats(s: Dict[str, float]) -> List[str]:
    pf = "inf" if s["profit_factor"] == float("inf") else f"{s['profit_factor']:.2f}"
    return [
        f"  Ордеров на вход: {s['orders']:.0f} | исполнено {s['filled']:.0f} | отменено {s['cancelled']:.0f}"
        f" | fill rate {s['fill_rate']:.0f}%",
        f"  Закрыто сделок: {s['closed']:.0f} | win rate {s['win_rate']:.1f}% | profit factor {pf}",
        f"  Net PnL: {s['net']:+.2f} USDT (комиссии {s['fees']:.2f}) | макс. просадка {s['max_dd']:.2f} USDT",
        f"  Средний плюс {s['avg_win']:+.2f} | средний минус {s['avg_loss']:+.2f}"
        f" | лучшая {s['best']:+.2f} | худшая {s['worst']:+.2f}",
    ]


def report_db(path: Path, since: Optional[str]) -> tuple[List[str], List[sqlite3.Row]]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    where, params = ("WHERE entry_time >= ?", (since,)) if since else ("", ())
    trades = _rows(conn, f"SELECT * FROM trades {where} ORDER BY entry_time", params)
    swhere, sparams = ("WHERE bar_time_utc >= ?", (since,)) if since else ("", ())
    signals = _rows(conn, f"SELECT outcome, reason, direction FROM signals {swhere}", sparams)
    conn.close()

    lines = [f"== {path.stem.replace('trades_', '').upper()} ({path.name})"]
    if trades:
        lines.append(f"  Период: {trades[0]['entry_time'][:16]} .. {trades[-1]['entry_time'][:16]}")
    lines += format_stats(trade_stats(trades))

    closed = [t for t in trades if t["status"] == "CLOSED"]
    if closed:
        by_reason = Counter(t["exit_reason"] or "?" for t in closed)
        lines.append("  Выходы: " + ", ".join(f"{k} {v}" for k, v in by_reason.most_common()))
        for d in ("LONG", "SHORT"):
            side = [float(t["net_pnl_usdt"] or 0.0) for t in closed if t["direction"] == d]
            if side:
                lines.append(f"  {d}: {len(side)} сделок, net {sum(side):+.2f} USDT")

    if signals:
        outcomes = Counter(r["outcome"] for r in signals)
        lines.append("  Сигналы: " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
        reasons = Counter(
            f"{r['outcome']}:{r['reason']}" for r in signals if r["outcome"] in ("FILTERED", "REJECTED", "SCAN_BLOCKED")
        )
        for key, n in reasons.most_common(8):
            lines.append(f"    {key} - {n}")
    elif trades:
        lines.append("  Журнала сигналов нет (база старше журнала)")
    return lines, trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-test report for all Krupa bots")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--since", help="UTC date, e.g. 2026-11-22")
    args = parser.parse_args()

    dbs = sorted(p for p in Path(args.data_dir).glob("trades_*.db") if p.stem != "trades_default")
    if not dbs:
        print(f"Нет баз trades_*.db в {args.data_dir}")
        return

    all_trades: List[sqlite3.Row] = []
    for db in dbs:
        lines, trades = report_db(db, args.since)
        all_trades += trades
        print("\n".join(lines))
        print()

    print("== ВСЕГО ПО ПОРТФЕЛЮ")
    all_trades.sort(key=lambda t: t["exit_time"] or t["entry_time"])
    print("\n".join(format_stats(trade_stats(all_trades))))


if __name__ == "__main__":
    main()
