"""Shared read-only analytics, Bybit hidden-cost auditing, blacklist pause manager,
and subprocess R&D Coin Scout for watchdog.py and ai_supervisor.py.

Safety guarantees:
- Opens all data/trades_*.db strictly in read-only URI mode (?mode=ro) with short timeout (3.0s)
  and retry on OperationalError so live trading bots are never blocked.
- Never modifies config_*.json or restarts trading bots.
- Pause management writes atomically to data/blacklist.json ONLY when explicitly called by user action.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from bot.bybit_client import BybitClient, INTERVAL_TO_MS
from bot.indicators import compute_adx, compute_atr, compute_rsi, ema

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent

BOT_CONFIGS: List[Dict[str, Any]] = [
    {
        "key": "ada",
        "symbol": "ADAUSDT",
        "strategy": "ada_fibo50_15x",
        "config_file": "config_ada.json",
        "db_path": "data/trades_ada.db",
        "log_path": "logs/bot_ada.log",
        "max_notional_usdt": 40000.0,
        "bench_wr": 86.5,
        "bench_pf": 2.27,
    },
    {
        "key": "wlfi",
        "symbol": "WLFIUSDT",
        "strategy": "wlfi_meanrev_15x_v2a",
        "config_file": "config_wlfi.json",
        "db_path": "data/trades_wlfi.db",
        "log_path": "logs/bot_wlfi.log",
        "max_notional_usdt": 8000.0,
        "bench_wr": 84.9,
        "bench_pf": 2.28,
    },
    {
        "key": "ltc",
        "symbol": "LTCUSDT",
        "strategy": "ltc_trend_bb_10x",
        "config_file": "config_ltc.json",
        "db_path": "data/trades_ltc.db",
        "log_path": "logs/bot_ltc.log",
        "max_notional_usdt": 50000.0,
        "bench_wr": 78.9,
        "bench_pf": 1.84,
    },
    {
        "key": "mnt",
        "symbol": "MNTUSDT",
        "strategy": "mnt_breakout_don_10x",
        "config_file": "config_mnt.json",
        "db_path": "data/trades_mnt.db",
        "log_path": "logs/bot_mnt.log",
        "max_notional_usdt": 20000.0,
        "bench_wr": 85.5,
        "bench_pf": 2.21,
    },
]

PORTFOLIO_BENCHMARK = {
    "package": "NEW",
    "win_rate_pct": 84.0,
    "oos_win_rate_pct": 80.7,
    "profit_factor": 1.92,
    "max_dd_pct": 11.0,
    "mc_p95_dd_pct": 17.0,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso(dt: Optional[datetime] = None) -> str:
    target = dt or utc_now()
    return target.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_bybit_mode(default: str = "demo") -> str:
    """Resolve Bybit execution mode ('demo', 'mainnet', 'testnet') from BYBIT_MODE in .env."""
    load_dotenv(ROOT_DIR / ".env")
    raw = os.getenv("BYBIT_MODE", "").strip().lower()
    if raw in ("live", "mainnet", "real"):
        return "mainnet"
    if raw in ("demo", "paper"):
        return "demo"
    if raw == "testnet":
        return "testnet"
    try:
        cfg = json.loads((ROOT_DIR / "config_ada.json").read_text(encoding="utf-8"))
        m = str(cfg.get("mode", default)).strip().lower()
        return "mainnet" if m in ("live", "mainnet") else m
    except Exception:
        return default


def create_bybit_client(mode: Optional[str] = None, timeout: int = 12) -> BybitClient:
    resolved = mode or resolve_bybit_mode()
    return BybitClient(mode=resolved, cache_dir=ROOT_DIR / "data" / "cache", timeout=timeout, public_mainnet=True)


# ----------------------------------------------------------------------
# Safe Read-Only SQLite Helper with Lock Retry & Short Timeout
# ----------------------------------------------------------------------
def query_sqlite_ro(db_path: str | Path, sql: str, params: Tuple[Any, ...] = (), retries: int = 3) -> List[Dict[str, Any]]:
    """Read SQLite DB strictly in read-only mode (?mode=ro) with 3.0s timeout and lock backoff."""
    p = Path(db_path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    if not p.exists():
        return []

    uri = f"file:{p.resolve().as_posix()}?mode=ro"
    for attempt in range(retries):
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=3.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError as exc:
            if attempt < retries - 1:
                time.sleep(0.25 * (attempt + 1))
                continue
            logger.warning("SQLite read-only query busy/failed on %s: %s", p.name, exc)
            return []
        except Exception as exc:
            logger.warning("SQLite read-only error on %s: %s", p.name, exc)
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    return []


# ----------------------------------------------------------------------
# Atomic Blacklist / Pause Manager (Advisor Mode -> User-Triggered Only)
# ----------------------------------------------------------------------
def get_active_pauses(blacklist_path: str | Path = "data/blacklist.json") -> Dict[str, Any]:
    p = ROOT_DIR / blacklist_path if not Path(blacklist_path).is_absolute() else Path(blacklist_path)
    if not p.exists():
        return {"macro_pause_reason": None, "active_entries": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"macro_pause_reason": None, "active_entries": []}

    now = utc_now()
    active_entries: List[Dict[str, Any]] = []
    for entry in data.get("entries", []):
        sym = str(entry.get("symbol", "")).upper().strip()
        if not sym:
            continue
        exp = entry.get("expires_at")
        if exp:
            try:
                exp_dt = datetime.fromisoformat(str(exp).replace("Z", "+00:00"))
                if exp_dt <= now:
                    continue
            except ValueError:
                pass
        active_entries.append(entry)
    return {
        "macro_pause_reason": data.get("macro_pause_reason"),
        "active_entries": active_entries,
    }


def set_coin_pause(
    symbol: str,
    hours: float,
    reason: str = "Paused by user via Telegram",
    blacklist_path: str | Path = "data/blacklist.json",
) -> Dict[str, Any]:
    """Atomically pause a coin in data/blacklist.json until now + hours (or remove pause if hours <= 0)."""
    sym = symbol.upper().strip()
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"

    p = ROOT_DIR / blacklist_path if not Path(blacklist_path).is_absolute() else Path(blacklist_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    now = utc_now()

    data: Dict[str, Any] = {"updated_at": utc_now_iso(now), "macro_pause_reason": None, "symbols": [], "entries": []}
    if p.exists():
        try:
            loaded = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass

    kept_entries: List[Dict[str, Any]] = []
    for entry in data.get("entries", []):
        esym = str(entry.get("symbol", "")).upper().strip()
        if not esym or esym == sym:
            continue
        exp = entry.get("expires_at")
        if exp:
            try:
                if datetime.fromisoformat(str(exp).replace("Z", "+00:00")) <= now:
                    continue
            except ValueError:
                pass
        kept_entries.append(entry)

    expires_iso: Optional[str] = None
    if hours > 0:
        expires_iso = utc_now_iso(now + timedelta(hours=float(hours)))
        kept_entries.append(
            {
                "symbol": sym,
                "reason": reason,
                "added_at": utc_now_iso(now),
                "expires_at": expires_iso,
            }
        )

    data["updated_at"] = utc_now_iso(now)
    data["entries"] = kept_entries
    data["symbols"] = sorted({e["symbol"] for e in kept_entries})

    tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(3):
        try:
            os.replace(tmp, p)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
            else:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

    return {"symbol": sym, "paused": hours > 0, "expires_at": expires_iso, "active_symbols": data["symbols"]}


# ----------------------------------------------------------------------
# 1. News Scout (Crypto News & Exchange Announcements for ADA/WLFI/LTC/MNT)
# ----------------------------------------------------------------------
def fetch_coin_news_digest(timeout: int = 10) -> Dict[str, Any]:
    """Fetch latest crypto headlines and filter for ADA, WLFI, LTC, MNT, Bybit, unlocks, hacks, SEC."""
    feeds = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ]
    keywords = {
        "ADAUSDT": ("ada", "cardano"),
        "WLFIUSDT": ("wlfi", "world liberty"),
        "LTCUSDT": ("ltc", "litecoin"),
        "MNTUSDT": ("mnt", "mantle"),
        "RISK": ("hack", "exploit", "delist", "unlock", "sec ", "lawsuit", "bybit", "halt", "insolvenc"),
    }
    matched: List[Dict[str, str]] = []
    general: List[Dict[str, str]] = []
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; KrupaNewsScout/1.0)"})

    for src, url in feeds:
        try:
            resp = sess.get(url, timeout=(8, timeout))
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:25]:
                title = (item.findtext("title") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                if not title:
                    continue
                low = title.lower()
                tags = [k for k, words in keywords.items() if any(w in low for w in words)]
                rec = {"source": src, "title": title, "published": pub, "tags": ",".join(tags)}
                if tags:
                    matched.append(rec)
                elif len(general) < 8:
                    general.append(rec)
        except Exception as exc:
            logger.debug("RSS fetch note (%s): %s", src, exc)
    sess.close()
    return {
        "checked_at_utc": utc_now_iso(),
        "coin_and_risk_headlines": matched[:15],
        "general_headlines": general[:6],
    }


# ----------------------------------------------------------------------
# 2. Market Strategist (BTC & ETH 4H/1D + Funding Rates + Flush Detection)
# ----------------------------------------------------------------------
def compute_market_regime(client: Optional[BybitClient] = None) -> Dict[str, Any]:
    """Analyze BTCUSDT and ETHUSDT on 4H and 1D plus Bybit Funding Rates."""
    cl = client or create_bybit_client()
    tickers = cl.get_tickers()
    assets: Dict[str, Any] = {}

    for sym in ("BTCUSDT", "ETHUSDT", "ADAUSDT", "WLFIUSDT", "LTCUSDT", "MNTUSDT"):
        tk = tickers.get(sym, {})
        assets[sym] = {
            "last_price": tk.get("last_price", 0.0),
            "funding_rate_pct": round(float(tk.get("funding_rate", 0.0)) * 100.0, 4),
            "price_24h_pcnt": round(float(tk.get("price_24h_pcnt", 0.0)) * 100.0, 2),
            "turnover_24h_usdt": round(float(tk.get("turnover_24h", 0.0)), 0),
        }

    for sym in ("BTCUSDT", "ETHUSDT"):
        df_15m = cl.get_klines(sym, interval="15", days=35, use_cache=True, closed_only=True)
        if len(df_15m) < 200:
            continue
        r1h = (float(df_15m["close"].iloc[-1]) / float(df_15m["close"].iloc[-5]) - 1.0) * 100.0 if len(df_15m) >= 5 else 0.0
        r4h = (float(df_15m["close"].iloc[-1]) / float(df_15m["close"].iloc[-17]) - 1.0) * 100.0 if len(df_15m) >= 17 else 0.0

        df_4h = BybitClient.resample_klines(df_15m, "240")
        e20_4h = float(ema(df_4h["close"], 20).iloc[-1])
        e50_4h = float(ema(df_4h["close"], 50).iloc[-1])
        adx_4h_s, _, _ = compute_adx(df_4h["high"], df_4h["low"], df_4h["close"], 14)
        adx_4h = float(adx_4h_s.fillna(0.0).iloc[-1])
        rsi_4h = float(compute_rsi(df_4h["close"], 14).fillna(50.0).iloc[-1])
        c_4h = float(df_4h["close"].iloc[-1])
        trend_4h = "BULL" if (c_4h > e50_4h and e20_4h > e50_4h) else ("BEAR" if (c_4h < e50_4h and e20_4h < e50_4h) else "RANGE")

        df_1d = BybitClient.resample_klines(df_15m, "D")
        e20_1d = float(ema(df_1d["close"], 20).iloc[-1])
        c_1d = float(df_1d["close"].iloc[-1])
        trend_1d = "ABOVE_EMA20" if c_1d >= e20_1d else "BELOW_EMA20"

        assets[sym].update(
            {
                "ret_1h_pct": round(r1h, 2),
                "ret_4h_pct": round(r4h, 2),
                "trend_4h": trend_4h,
                "adx_4h": round(adx_4h, 1),
                "rsi_4h": round(rsi_4h, 1),
                "trend_1d": trend_1d,
            }
        )

    btc = assets.get("BTCUSDT", {})
    btc_r4h = float(btc.get("ret_4h_pct", 0.0))
    btc_24h = float(btc.get("price_24h_pcnt", 0.0))
    btc_adx = float(btc.get("adx_4h", 0.0))
    btc_trend = btc.get("trend_4h", "RANGE")

    if btc_r4h <= -3.0 or btc_24h <= -6.0:
        regime = "PANIC_FLUSH"
        wlfi_advice = "PAUSE_RECOMMENDED (безоткатный пролив BTC — риск для контртрендовых лонгов WLFI)"
    elif btc_adx < 18.0:
        regime = "CHOP_RANGE"
        wlfi_advice = "FAVORABLE (боковик благоприятен для возврата к среднему WLFI)"
    else:
        regime = f"DIRECTIONAL_{btc_trend}"
        wlfi_advice = "NORMAL (работают штатные фильтры BTC 1H Veto)"

    return {
        "checked_at_utc": utc_now_iso(),
        "regime": regime,
        "wlfi_advisor_note": wlfi_advice,
        "assets": assets,
        "pauses": get_active_pauses(),
    }


# ----------------------------------------------------------------------
# 3. Trade & Hidden Cost Auditor (SQLite + Bybit Execution / Transaction Log)
# ----------------------------------------------------------------------
def audit_trades_and_hidden_costs(days: int = 7, client: Optional[BybitClient] = None) -> Dict[str, Any]:
    """Compare closed trades across all 4 SQLite DBs against NEW benchmark & fetch Bybit funding/fee logs."""
    cl = client or create_bybit_client()
    cutoff_dt = utc_now() - timedelta(days=days)
    cutoff_iso = utc_now_iso(cutoff_dt)

    per_bot: Dict[str, Any] = {}
    all_closed: List[Dict[str, Any]] = []
    active_trades: List[Dict[str, Any]] = []

    for b in BOT_CONFIGS:
        sym = b["symbol"]
        closed = query_sqlite_ro(
            b["db_path"],
            "SELECT * FROM trades WHERE status = 'CLOSED' AND (exit_time IS NULL OR exit_time >= ?) ORDER BY id DESC",
            (cutoff_iso,),
        )
        active = query_sqlite_ro(
            b["db_path"],
            "SELECT id, symbol, direction, status, entry_time, expires_at_utc, entry_price, sl_price, tp1_price, "
            "be_trigger_price, be_price, be2_trigger_price, be2_price, qty, margin_usdt, entry_order_id "
            "FROM trades WHERE status IN ('PENDING', 'OPEN', 'TP1_HIT', 'BE_ACTIVE', 'BE2_ACTIVE') ORDER BY id ASC",
        )
        active_trades.extend(active)
        all_closed.extend(closed)

        n = len(closed)
        nets = [float(t.get("net_pnl_usdt") or 0.0) for t in closed]
        fees = [float(t.get("fees_usdt") or 0.0) for t in closed]
        wins = [x for x in nets if x > 0]
        losses = [-x for x in nets if x < 0]
        sls = sum(1 for t in closed if str(t.get("exit_reason", "")).upper() == "SL")
        flips = sum(1 for t in closed if "FLIP" in str(t.get("exit_reason", "")).upper())
        wr = round(len(wins) / n * 100.0, 1) if n > 0 else 0.0
        pf = round(sum(wins) / sum(losses), 2) if sum(losses) > 0 else (99.0 if wins else 0.0)

        per_bot[sym] = {
            "trades": n,
            "win_rate_pct": wr,
            "benchmark_wr_pct": b["bench_wr"],
            "profit_factor": pf,
            "benchmark_pf": b["bench_pf"],
            "sl_count": sls,
            "h1_flip_exits": flips,
            "net_pnl_usdt": round(sum(nets), 2),
            "db_fees_usdt": round(sum(fees), 2),
            "active_count": len(active),
        }

    # Fetch live wallet balance + Bybit exchange execution fees & funding logs if credentials exist
    wallet = {"equity": 0.0, "available": 0.0}
    exchange_costs = {"funding_usdt": 0.0, "exchange_exec_fees_usdt": 0.0, "maker_fills": 0, "taker_fills": 0}
    if cl.has_credentials:
        try:
            wallet = cl.get_wallet_balance("USDT")
        except Exception as exc:
            logger.warning("Wallet balance fetch warning: %s", exc)

        try:
            priv = cl._require_private()
            start_ms = int((utc_now() - timedelta(days=min(days, 7))).timestamp() * 1000)
            for b in BOT_CONFIGS:
                resp = priv.get_executions(category="linear", symbol=b["symbol"], startTime=start_ms, limit=100)
                if resp.get("retCode") == 0:
                    for ex in resp.get("result", {}).get("list", []):
                        fee = float(ex.get("execFee") or 0.0)
                        exec_type = str(ex.get("execType", ""))
                        is_maker = bool(ex.get("isMaker", False))
                        if exec_type == "Funding":
                            exchange_costs["funding_usdt"] -= fee
                        else:
                            exchange_costs["exchange_exec_fees_usdt"] += fee
                            if is_maker:
                                exchange_costs["maker_fills"] += 1
                            else:
                                exchange_costs["taker_fills"] += 1
        except Exception as exc:
            logger.debug("Exchange executions cost query note: %s", exc)

    total_n = len(all_closed)
    tot_nets = [float(t.get("net_pnl_usdt") or 0.0) for t in all_closed]
    tot_wins = [x for x in tot_nets if x > 0]
    tot_loss = [-x for x in tot_nets if x < 0]

    return {
        "checked_at_utc": utc_now_iso(),
        "mode": cl.mode,
        "window_days": days,
        "wallet": wallet,
        "portfolio_summary": {
            "trades": total_n,
            "win_rate_pct": round(len(tot_wins) / total_n * 100.0, 1) if total_n > 0 else 0.0,
            "profit_factor": round(sum(tot_wins) / sum(tot_loss), 2) if sum(tot_loss) > 0 else 0.0,
            "net_pnl_db_usdt": round(sum(tot_nets), 2),
            "net_after_funding_usdt": round(sum(tot_nets) + exchange_costs["funding_usdt"], 2),
            "benchmark": PORTFOLIO_BENCHMARK,
        },
        "exchange_hidden_costs": {
            "funding_net_usdt": round(exchange_costs["funding_usdt"], 3),
            "exchange_exec_fees_usdt": round(exchange_costs["exchange_exec_fees_usdt"], 3),
            "maker_fills": exchange_costs["maker_fills"],
            "taker_fills": exchange_costs["taker_fills"],
        },
        "per_symbol": per_bot,
        "active_trades": active_trades,
        "pauses": get_active_pauses(),
    }


# ----------------------------------------------------------------------
# 4. Missed Fill Auditor (Cancelled TTL Limit Orders vs Subsequent Price Path)
# ----------------------------------------------------------------------
def audit_missed_fills(days: int = 14, client: Optional[BybitClient] = None) -> Dict[str, Any]:
    """Analyze CANCELLED Maker limit orders to see how often price missed entry by <=0.10% and hit TP."""
    cl = client or create_bybit_client()
    cutoff_iso = utc_now_iso(utc_now() - timedelta(days=days))
    e_ms = INTERVAL_TO_MS["15"]
    report: Dict[str, Any] = {}

    for b in BOT_CONFIGS:
        sym = b["symbol"]
        cancelled = query_sqlite_ro(
            b["db_path"],
            "SELECT id, symbol, direction, entry_time, expires_at_utc, entry_price, sl_price, tp1_price, exit_reason "
            "FROM trades WHERE status = 'CANCELLED' AND entry_time >= ? ORDER BY id DESC",
            (cutoff_iso,),
        )
        if not cancelled:
            report[sym] = {"cancelled_ttl": 0, "near_miss_010_pct": 0, "would_hit_tp": 0, "would_hit_sl": 0}
            continue

        df = cl.get_klines(sym, interval="15", days=min(days + 2, 30), use_cache=True, closed_only=True)
        near_miss = 0
        would_tp = 0
        would_sl = 0

        for tr in cancelled:
            try:
                t0 = int(datetime.fromisoformat(str(tr["entry_time"]).replace("Z", "+00:00")).timestamp() * 1000)
            except Exception:
                continue
            lp = float(tr.get("entry_price") or 0.0)
            tp = float(tr.get("tp1_price") or 0.0)
            sl = float(tr.get("sl_price") or 0.0)
            is_l = str(tr.get("direction", "")).upper() == "LONG"
            if lp <= 0:
                continue

            sub = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t0 + 16 * e_ms)]
            if sub.empty:
                continue

            ttl_sub = sub.iloc[:4]
            if is_l:
                closest_gap_pct = (float(ttl_sub["low"].min()) - lp) / lp * 100.0
            else:
                closest_gap_pct = (lp - float(ttl_sub["high"].max())) / lp * 100.0

            if 0.0 <= closest_gap_pct <= 0.12:
                near_miss += 1
                for _, row in sub.iterrows():
                    hi, lo = float(row["high"]), float(row["low"])
                    if (lo <= sl) if is_l else (hi >= sl):
                        would_sl += 1
                        break
                    if (hi >= tp) if is_l else (lo <= tp):
                        would_tp += 1
                        break

        report[sym] = {
            "cancelled_ttl": len(cancelled),
            "near_miss_010_pct": near_miss,
            "would_hit_tp": would_tp,
            "would_hit_sl": would_sl,
        }

    return {"checked_at_utc": utc_now_iso(), "window_days": days, "per_symbol": report}


# ----------------------------------------------------------------------
# 5. Liquidity & Notional Cap Monitor (Orderbook Depth vs Max Notional Caps)
# ----------------------------------------------------------------------
def check_liquidity_and_caps(equity: float = 2000.0, client: Optional[BybitClient] = None) -> Dict[str, Any]:
    """Inspect real Bybit L50 orderbook depth and 24h turnover vs current max_notional_usdt caps."""
    cl = client or create_bybit_client()
    tickers = cl.get_tickers()
    target_notional = equity * 0.15 * 10.0  # 15% margin * 10x leverage
    out: Dict[str, Any] = {}

    for b in BOT_CONFIGS:
        sym = b["symbol"]
        cap = float(b["max_notional_usdt"])
        tk = tickers.get(sym, {})
        turnover_24h = float(tk.get("turnover_24h") or 0.0)
        spread_bps = 0.0
        bid_depth_03pct = 0.0
        ask_depth_03pct = 0.0

        try:
            ob = cl._public_get("/v5/market/orderbook", params={"category": "linear", "symbol": sym, "limit": 50})
            bids = [(float(p), float(q)) for p, q in ob.get("b", [])]
            asks = [(float(p), float(q)) for p, q in ob.get("a", [])]
            if bids and asks:
                best_b, best_a = bids[0][0], asks[0][0]
                mid = 0.5 * (best_b + best_a)
                spread_bps = (best_a - best_b) / mid * 10000.0 if mid > 0 else 0.0
                bid_depth_03pct = sum(p * q for p, q in bids if p >= best_b * 0.997)
                ask_depth_03pct = sum(p * q for p, q in asks if p <= best_a * 1.003)
        except Exception as exc:
            logger.debug("Orderbook fetch note for %s: %s", sym, exc)

        min_side_depth = min(bid_depth_03pct, ask_depth_03pct)
        safe_cap_est = max(5000.0, round(min(turnover_24h * 0.001, max(min_side_depth * 1.5, cap)) / 1000.0) * 1000.0)
        if target_notional >= cap * 0.85 and min_side_depth > cap * 0.8:
            rec = f"Депозит вырос: текущий лот ${target_notional:,.0f} близок к потолку ${cap:,.0f}, стакан позволяет поднять потолок."
        elif min_side_depth > 0 and min_side_depth < cap * 0.25:
            rec = f"Ликвидность стакана сжалась (${min_side_depth:,.0f} в ±0.3%): рекомендуется временно снизить потолок с ${cap:,.0f}."
        else:
            rec = "OK (потолок соответствует глубине стакана)"

        out[sym] = {
            "current_cap_usdt": cap,
            "current_order_notional_usdt": round(min(target_notional, cap), 0),
            "turnover_24h_musdt": round(turnover_24h / 1e6, 2),
            "spread_bps": round(spread_bps, 2),
            "depth_within_03pct_usdt": round(min_side_depth, 0),
            "recommendation": rec,
        }

    return {"checked_at_utc": utc_now_iso(), "equity_usdt": equity, "per_symbol": out}


# ----------------------------------------------------------------------
# 6. R&D Coin Scout (Subprocess 3-Fold + OOS Screening on Heavy Liquid Coins)
# ----------------------------------------------------------------------
RND_LOCK_PATH = ROOT_DIR / "data" / "rnd_scout.lock"
RND_RESULT_PATH = ROOT_DIR / "data" / "rnd_scout_latest.json"


def run_rnd_coin_scout_blocking(candidates: Tuple[str, ...] = ("SOLUSDT", "XRPUSDT", "SUIUSDT", "AVAXUSDT", "DOGEUSDT")) -> Dict[str, Any]:
    """Evaluate our strategies across liquid candidates over 180d (3 folds + OOS + correlation vs ADAUSDT)."""
    cl = create_bybit_client()
    e_ms, h1_ms = INTERVAL_TO_MS["15"], INTERVAL_TO_MS["60"]
    day_ms = 86_400_000

    df_btc = cl.get_klines("BTCUSDT", "15", days=185, use_cache=True, closed_only=True)
    df_ada = cl.get_klines("ADAUSDT", "15", days=185, use_cache=True, closed_only=True)
    sim_end = int(df_ada["timestamp"].iloc[-1])
    sim_start = sim_end - 180 * day_ms
    split_oos = sim_start + 120 * day_ms
    folds = [sim_start + k * 60 * day_ms for k in range(4)]

    ada_1h = BybitClient.resample_klines(df_ada[df_ada["timestamp"] >= sim_start], "60")["close"].pct_change().dropna()
    btc = df_btc[["timestamp", "close"]].copy().sort_values("timestamp")
    btc["btc_r1h"] = btc["close"].pct_change(4) * 100.0

    results: List[Dict[str, Any]] = []
    mk, tk = 0.0002, 0.0010

    for sym in candidates:
        try:
            df_c = cl.get_klines(sym, "15", days=185, use_cache=True, closed_only=True)
            df = df_c[(df_c["timestamp"] >= sim_start - 5 * day_ms) & (df_c["timestamp"] <= sim_end)].copy().reset_index(drop=True)
            if len(df) < 5000:
                continue
            sym_1h = BybitClient.resample_klines(df[df["timestamp"] >= sim_start], "60")["close"].pct_change().dropna()
            m_len = min(len(ada_1h), len(sym_1h))
            corr_ada = float(np.corrcoef(ada_1h.iloc[-m_len:], sym_1h.iloc[-m_len:])[0, 1]) if m_len > 50 else 0.0

            df["close_ts"] = (df["timestamp"] + e_ms).astype("int64")
            df["atr"] = compute_atr(df["high"], df["low"], df["close"], 14)
            df["atr_pct"] = df["atr"] / df["close"].replace(0.0, np.nan) * 100.0
            df["e20"] = ema(df["close"], 20)
            df["e200"] = ema(df["close"], 200)
            df["sd20"] = df["close"].rolling(20, min_periods=20).std().fillna(0.0)
            df["rsi14"] = compute_rsi(df["close"], 14)
            df["hi_10"] = df["high"].shift(1).rolling(10, min_periods=10).max()
            df["lo_10"] = df["low"].shift(1).rolling(10, min_periods=10).min()
            is_abn = ((df["high"] - df["low"]) > 3.2 * df["atr"]).fillna(False)
            df["vol_cd"] = is_abn | is_abn.shift(1, fill_value=False)
            df = pd.merge_asof(df, btc[["timestamp", "btc_r1h"]], on="timestamp", direction="backward")

            d1 = BybitClient.resample_klines(df_c, "60")
            e20h, e50h = ema(d1["close"], 20), ema(d1["close"], 50)
            adxh, _, _ = compute_adx(d1["high"], d1["low"], d1["close"], 14)
            h1 = pd.DataFrame({
                "close_ts": (d1["timestamp"] + h1_ms).astype("int64"),
                "h1_bull": (d1["close"] > e50h) & (e20h > e50h),
                "h1_bear": (d1["close"] < e50h) & (e20h < e50h),
                "h1_adx": adxh.fillna(0.0).values,
                "h1_slope": (e50h.pct_change(3) * 100.0).fillna(0.0).values,
            }).sort_values("close_ts")
            df = pd.merge_asof(df, h1, on="close_ts", direction="backward")
            df = df[df["timestamp"] >= sim_start].reset_index(drop=True)

            ts_arr = df["timestamp"].to_numpy(np.int64)
            hi_a, lo_a, cl_a = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
            atr_a, atrp_a = df["atr"].to_numpy(float), df["atr_pct"].to_numpy(float)
            e20_a, sd20_a, rsi_a = df["e20"].to_numpy(float), df["sd20"].to_numpy(float), df["rsi14"].to_numpy(float)
            hi10_a, lo10_a = df["hi_10"].to_numpy(float), df["lo_10"].to_numpy(float)
            vcd_a = df["vol_cd"].astype("boolean").fillna(False).to_numpy(bool)
            hbull_a = df["h1_bull"].astype("boolean").fillna(False).to_numpy(bool)
            hbear_a = df["h1_bear"].astype("boolean").fillna(False).to_numpy(bool)
            hadx_a = df["h1_adx"].fillna(0.0).to_numpy(float)
            hslp_a = df["h1_slope"].fillna(0.0).to_numpy(float)
            br1_a = df["btc_r1h"].fillna(0.0).to_numpy(float)

            for tpl in ("trend_bb_10x", "breakout_don_10x", "keltner_mr_10x"):
                trades = []
                pos, pend, cd = None, None, -1
                notional = 3000.0
                for i in range(25, len(df)):
                    hi, lo, cl = hi_a[i], lo_a[i], cl_a[i]
                    if pos is not None:
                        is_l = pos["dir"] == "LONG"
                        csl = pos["be2_p"] if pos["be2"] else (pos["be1_p"] if pos["be1"] else pos["sl_p"])
                        cr = "BE2" if pos["be2"] else ("BE" if pos["be1"] else "SL")
                        closed = False
                        if (lo <= csl) if is_l else (hi >= csl):
                            px, r, im, closed = csl, cr, False, True
                        elif (hi >= pos["tp_p"]) if is_l else (lo <= pos["tp_p"]):
                            px, r, im, closed = pos["tp_p"], "TP", True, True
                        else:
                            if not pos["be1"] and ((hi >= pos["be1_tr"]) if is_l else (lo <= pos["be1_tr"])):
                                pos["be1"] = True
                            if not pos["be2"] and ((hi >= pos["be2_tr"]) if is_l else (lo <= pos["be2_tr"])):
                                pos["be2"] = True
                            if pos["flip"] and (hbear_a[i] if is_l else hbull_a[i]):
                                px, r, im, closed = cl, "FLIP", False, True
                        if closed:
                            qty = notional / pos["ep"]
                            gross = (px - pos["ep"]) * qty if is_l else (pos["ep"] - px) * qty
                            net = gross - notional * mk - abs(qty * px) * (mk if im else tk)
                            trades.append({"ts": pos["ts"], "net": float(net), "sl": r == "SL"})
                            pos, cd = None, i + 1
                    if pend is not None:
                        if i > pend["exp"] or i < cd:
                            pend = None
                        else:
                            is_l = pend["dir"] == "LONG"
                            lp = pend["lp"]
                            if (lo < lp) if is_l else (hi > lp):
                                eff = pend["eff"]
                                sl_f = pend["sl"] / 100.0
                                b1t, b1p, b2t, b2p = pend["bef"]
                                pos = {
                                    "dir": pend["dir"], "ep": lp, "ts": int(ts_arr[i]), "be1": False, "be2": False,
                                    "sl_p": lp * (1 - sl_f) if is_l else lp * (1 + sl_f),
                                    "tp_p": lp * (1 + eff) if is_l else lp * (1 - eff),
                                    "be1_tr": lp * (1 + eff * b1t) if is_l else lp * (1 - eff * b1t),
                                    "be1_p": lp * (1 + eff * b1p) if is_l else lp * (1 - eff * b1p),
                                    "be2_tr": lp * (1 + eff * b2t) if is_l else lp * (1 - eff * b2t),
                                    "be2_p": lp * (1 + eff * b2p) if is_l else lp * (1 - eff * b2p),
                                    "flip": pend["flip"],
                                }
                                pend = None
                            elif i >= pend["exp"]:
                                pend = None
                    if pos is None and pend is None and i >= cd:
                        atr, atrp = atr_a[i], atrp_a[i]
                        if atr <= 0 or atrp < 0.10 or atrp > 1.90:
                            continue
                        if tpl == "trend_bb_10x":
                            if vcd_a[i] or hadx_a[i] < 14.0 or sd20_a[i] <= 0:
                                continue
                            if hbull_a[i] and cl < e20_a[i] and br1_a[i] > -0.7:
                                lp = e20_a[i] - 1.4 * sd20_a[i]
                                if 0 < lp < cl * 0.9992:
                                    pend = {"dir": "LONG", "lp": lp, "eff": 0.0135, "sl": 2.5, "exp": i + 2, "flip": True, "bef": (0.65, 0.22, 0.82, 0.42)}
                            elif hbear_a[i] and cl > e20_a[i] and br1_a[i] < 0.7:
                                lp = e20_a[i] + 1.4 * sd20_a[i]
                                if lp > cl * 1.0008:
                                    pend = {"dir": "SHORT", "lp": lp, "eff": 0.0135, "sl": 2.5, "exp": i + 2, "flip": True, "bef": (0.65, 0.22, 0.82, 0.42)}
                        elif tpl == "breakout_don_10x":
                            if vcd_a[i] or hadx_a[i] < 18.0:
                                continue
                            if hbull_a[i] and cl > hi10_a[i] and hslp_a[i] > 0 and br1_a[i] > -0.5:
                                lp = cl - 0.30 * atr
                                if 0 < lp < cl * 0.9992:
                                    pend = {"dir": "LONG", "lp": lp, "eff": 0.014, "sl": 2.6, "exp": i + 3, "flip": True, "bef": (0.50, 0.15, 0.75, 0.35)}
                            elif hbear_a[i] and cl < lo10_a[i] and hslp_a[i] < 0 and br1_a[i] < 0.5:
                                lp = cl + 0.30 * atr
                                if lp > cl * 1.0008:
                                    pend = {"dir": "SHORT", "lp": lp, "eff": 0.014, "sl": 2.6, "exp": i + 3, "flip": True, "bef": (0.50, 0.15, 0.75, 0.35)}
                        elif tpl == "keltner_mr_10x":
                            if atrp > 1.40 or hadx_a[i] > 40.0:
                                continue
                            if cl < e20_a[i] and rsi_a[i] <= 40.0 and br1_a[i] > -1.0:
                                lp = e20_a[i] - 3.2 * atr
                                if 0 < lp < cl * 0.9992:
                                    pend = {"dir": "LONG", "lp": lp, "eff": 0.012, "sl": 2.5, "exp": i + 3, "flip": False, "bef": (0.65, 0.22, 0.82, 0.42)}
                            elif cl > e20_a[i] and rsi_a[i] >= 60.0 and br1_a[i] < 1.0:
                                lp = e20_a[i] + 3.2 * atr
                                if lp > cl * 1.0008:
                                    pend = {"dir": "SHORT", "lp": lp, "eff": 0.012, "sl": 2.5, "exp": i + 3, "flip": False, "bef": (0.65, 0.22, 0.82, 0.42)}

                if len(trades) < 30:
                    continue
                nets = [t["net"] for t in trades]
                wins = [x for x in nets if x > 0]
                losses = [-x for x in nets if x < 0]
                oos_nets = [t["net"] for t in trades if t["ts"] >= split_oos]
                fold_nets = [
                    round(sum(t["net"] for t in trades if folds[k] <= t["ts"] < folds[k + 1]), 0)
                    for k in range(3)
                ]
                results.append(
                    {
                        "symbol": sym,
                        "template": tpl,
                        "trades": len(trades),
                        "win_rate_pct": round(len(wins) / len(trades) * 100.0, 1),
                        "sl_count": sum(1 for t in trades if t["sl"]),
                        "net_usdt_fixed3k": round(sum(nets), 0),
                        "oos_net_usdt": round(sum(oos_nets), 0),
                        "profit_factor": round(sum(wins) / sum(losses), 2) if sum(losses) > 0 else 99.0,
                        "folds_net": fold_nets,
                        "positive_folds": sum(1 for fn in fold_nets if fn > 0),
                        "corr_vs_ada": round(corr_ada, 2),
                    }
                )
        except Exception as exc:
            logger.warning("R&D Coin Scout warning on %s: %s", sym, exc)

    results.sort(key=lambda r: (r["positive_folds"], r["oos_net_usdt"], r["net_usdt_fixed3k"]), reverse=True)
    payload = {"completed_at_utc": utc_now_iso(), "top_candidates": results[:8]}
    RND_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RND_RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="Supervisor tools CLI worker")
    parser.add_argument("--rnd-scout", action="store_true", help="Run heavy R&D Coin Scout in background subprocess")
    parser.add_argument("--notify-telegram", action="store_true", help="Send R&D result to Telegram when finished")
    args = parser.parse_args()

    if args.rnd_scout:
        load_dotenv(ROOT_DIR / ".env")
        try:
            RND_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
            RND_LOCK_PATH.write_text(json.dumps({"pid": os.getpid(), "started_at": utc_now_iso()}), encoding="utf-8")
            res = run_rnd_coin_scout_blocking()
            if args.notify_telegram:
                token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
                chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
                if token and chat_id and "your_telegram" not in token.lower():
                    lines = ["🔬 *R&D Coin Scout (180д, 3 фолда + OOS, фикс. лот $3,000):*"]
                    for c in res.get("top_candidates", [])[:5]:
                        lines.append(
                            f"• `{c['symbol']}` ({c['template']}): Net *+${c['net_usdt_fixed3k']:.0f}* "
                            f"(OOS *+${c['oos_net_usdt']:.0f}*), WR `{c['win_rate_pct']}%`, PF `{c['profit_factor']}`, "
                            f"SL `{c['sl_count']}`, Фолды `{c['positive_folds']}/3`, Corr(ADA)=`{c['corr_vs_ada']}`"
                        )
                    requests.post(
                        f"https://api.telegram.org/bot{token}/sendMessage",
                        json={"chat_id": chat_id, "text": "\n".join(lines), "parse_mode": "Markdown"},
                        timeout=10,
                    )
        finally:
            try:
                RND_LOCK_PATH.unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    _cli_main()
