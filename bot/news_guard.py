"""Live Fail-Closed US Macro Shield (`bot/news_guard.py`).

Enforces strict protection around high-impact US macro releases (FOMC, CPI, NFP, PPI, Fed Chair):
  1. Live Calendar + Local UTC Cache (`data/macro_events.json`):
     Downloads the ForexFactory free JSON calendar (`thisweek`), normalizes all
     timestamps strictly to UTC, and refreshes automatically every `macro_refresh_hours` (12h).
  2. Hard Entry & Limit Block (`-30m .. +15m` UTC):
     Blocks all new limit entries and instructs executor to cancel any unfilled pending limits.
  3. Open Position Pre-News Protection (`-30m .. 0m` UTC):
     Instructs executor to force-close open positions at market if `uPnL < +0.25%`.
  4. Hard Fail-Safe (Fail-Closed):
     If calendar refresh fails AND local cache is missing, corrupt, or older than
     `macro_max_stale_hours` (36h), blocks all new entries (`fail_safe_active=True`) and logs ERROR.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from bot.bybit_client import BybitClient

logger = logging.getLogger(__name__)

FF_CALENDAR_URLS = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
)

# Core US Macro Shield keywords (FOMC, CPI, NFP, PPI, Fed Chair/Chairman / Rates)
CORE_US_MACRO_PATTERNS = (
    re.compile(r"\bfomc\b", re.I),
    re.compile(r"\bfederal\s+funds\s+rate\b", re.I),
    re.compile(r"\b(core\s+)?cpi\b", re.I),
    re.compile(r"\bconsumer\s+price\s+index\b", re.I),
    re.compile(r"\bnon[-\s]?farm\b|\bnfp\b|\bemployment\s+change\b", re.I),
    re.compile(r"\b(core\s+)?ppi\b", re.I),
    re.compile(r"\bproducer\s+price\s+index\b", re.I),
    re.compile(r"\bfed\s+chair(man|person)?\b|\bpowell\b", re.I),
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def is_core_us_macro_title(title: str, extra_patterns: Optional[List[str]] = None) -> bool:
    if any(pat.search(title) for pat in CORE_US_MACRO_PATTERNS):
        return True
    for raw in extra_patterns or []:
        try:
            if re.search(str(raw), title, re.I):
                return True
        except re.error:
            logger.warning("Invalid macro_extra_patterns entry ignored: %r", raw)
    return False


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.error("Could not write %s: %s", path, exc)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def read_news_state(path: str | Path = "data/news_state.json") -> Dict[str, Any]:
    """Fail-CLOSED reader for `data/news_state.json`."""
    p = Path(path)
    if not p.exists():
        return {
            "trading_allowed": False,
            "entry_allowed": False,
            "cancel_pending_limits": False,
            "protect_open_positions": False,
            "fail_safe_active": True,
            "reason": "US Macro Shield state missing (fail-safe block until calendar refreshes)",
            "layer": "MACRO_SHIELD_FAIL_SAFE",
            "pause_until_utc": None,
            "updated_at": None,
            "upcoming_core_events": [],
        }
    try:
        state = json.loads(p.read_text(encoding="utf-8"))
        # Auto-expire timed macro window if NOT in fail-safe mode
        if not state.get("fail_safe_active", False) and not state.get("trading_allowed", True):
            until = parse_iso_utc(state.get("pause_until_utc"))
            if until is not None and utc_now() >= until:
                state["trading_allowed"] = True
                state["entry_allowed"] = True
                state["cancel_pending_limits"] = False
                state["protect_open_positions"] = False
                state["pause_until_utc"] = None
                state["reason"] = None
                state["layer"] = None
        return state
    except Exception as exc:
        logger.error("US Macro Shield: unreadable state %s (%s) -> FAIL-SAFE BLOCK.", p, exc)
        return {
            "trading_allowed": False,
            "entry_allowed": False,
            "cancel_pending_limits": False,
            "protect_open_positions": False,
            "fail_safe_active": True,
            "reason": f"US Macro Shield state unreadable: {exc}",
            "layer": "MACRO_SHIELD_FAIL_SAFE",
            "pause_until_utc": None,
            "updated_at": None,
            "upcoming_core_events": [],
        }


class NewsGuard:
    """Live US Macro Shield (FOMC/CPI/NFP/PPI) with UTC cache (`data/macro_events.json`) & fail-closed safety."""

    def __init__(
        self,
        config: Dict[str, Any],
        bybit_client: Optional[BybitClient] = None,
        state_path: str | Path = "data/news_state.json",
        macro_events_path: Optional[str | Path] = None,
        on_pause_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.config = config
        self.client = bybit_client
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_macro = macro_events_path or config.get("macro_events_path", "data/macro_events.json")
        self.macro_events_path = Path(resolved_macro)
        self.macro_events_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_pause_callback = on_pause_callback

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; BybitMacroShield/2.0)"})
        self._last_logged_key: Optional[str] = None
        self._last_fetch_attempt_mono: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("macro_shield_enabled", True))

    @property
    def pre_window_min(self) -> int:
        return int(self.config.get("macro_pre_window_min", 30))

    @property
    def post_window_min(self) -> int:
        return int(self.config.get("macro_post_window_min", 15))

    @property
    def protect_upnl_pct(self) -> float:
        return float(self.config.get("macro_protect_upnl_pct", 0.25))

    @property
    def refresh_hours(self) -> float:
        return float(self.config.get("macro_refresh_hours", 12.0))

    @property
    def max_stale_hours(self) -> float:
        return float(self.config.get("macro_max_stale_hours", 36.0))

    @property
    def extra_patterns(self) -> List[str]:
        raw = self.config.get("macro_extra_patterns", [])
        return [str(x) for x in raw] if isinstance(raw, list) else []

    @staticmethod
    def _covers(payload: Dict[str, Any], now: datetime) -> bool:
        """The weekly calendar must actually span `now`, not just be recently downloaded."""
        start = parse_iso_utc(payload.get("coverage_from_utc"))
        end = parse_iso_utc(payload.get("coverage_to_utc"))
        if start is None or end is None:
            return False
        return start - timedelta(days=1) <= now <= end + timedelta(days=2)

    # ------------------------------------------------------------------
    # Live Economic Calendar Fetch + UTC Normalization + Atomic Cache
    # ------------------------------------------------------------------
    def _read_cached_calendar(self) -> Optional[Dict[str, Any]]:
        if not self.macro_events_path.exists():
            return None
        try:
            data = json.loads(self.macro_events_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("events"), list) and data.get("fetched_at_utc"):
                data["events"] = [e for e in data["events"] if isinstance(e, dict)]
                return data
        except Exception as exc:
            logger.error("US Macro Shield: corrupted calendar cache %s (%s).", self.macro_events_path, exc)
        return None

    def _cache_age_hours(self, payload: Dict[str, Any], now: datetime) -> float:
        fetched = parse_iso_utc(payload.get("fetched_at_utc"))
        if fetched is None:
            return 1e9
        return max(0.0, (now - fetched).total_seconds() / 3600.0)

    def refresh_calendar(self, force: bool = False, now: Optional[datetime] = None) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str]]:
        """
        Ensure `data/macro_events.json` is fresh.
        Returns `(payload, is_fresh_ok, error_msg)`.
        Never silently ignores errors: if refresh fails and cache is missing or older than
        `max_stale_hours`, returns `(payload_or_none, False, error_msg)` -> triggers FAIL-CLOSED block.
        """
        cur_now = now or utc_now()
        cached = self._read_cached_calendar()

        if not force and cached is not None:
            age_h = self._cache_age_hours(cached, cur_now)
            if age_h < self.refresh_hours and self._covers(cached, cur_now):
                return cached, True, None

        # Throttle retry spam to once per 5 min when network is down (unless force=True)
        now_mono = time.time()
        if not force and (now_mono - self._last_fetch_attempt_mono) < 300.0:
            if (
                cached is not None
                and self._cache_age_hours(cached, cur_now) <= self.max_stale_hours
                and self._covers(cached, cur_now)
            ):
                return cached, True, None
            return cached, False, "Macro calendar stale/missing (awaiting retry cooldown)"

        self._last_fetch_attempt_mono = now_mono
        raw_all: List[Dict[str, Any]] = []
        fetched_sources: List[str] = []
        errors: List[str] = []

        for url in FF_CALENDAR_URLS:
            try:
                resp = self._session.get(url, timeout=(10, 12))
                resp.raise_for_status()
                items = resp.json()
                if isinstance(items, list):
                    raw_all.extend(i for i in items if isinstance(i, dict))
                    fetched_sources.append(url)
                else:
                    errors.append(f"{url}: unexpected JSON root")
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                try:
                    self._session.close()
                except Exception:
                    pass
                self._session = requests.Session()
                self._session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; BybitMacroShield/2.0)"})

        all_dates = [d for d in (parse_iso_utc(ev.get("date")) for ev in raw_all) if d is not None]
        if fetched_sources and not all_dates:
            errors.append("calendar has no dated events")
            fetched_sources = []
        elif fetched_sources:
            cov = {"coverage_from_utc": to_iso_z(min(all_dates)), "coverage_to_utc": to_iso_z(max(all_dates))}
            if not self._covers(cov, cur_now):
                errors.append(f"calendar covers {cov['coverage_from_utc']}..{cov['coverage_to_utc']}, not the current date")
                fetched_sources = []

        if fetched_sources:
            seen_keys = set()
            core_events: List[Dict[str, Any]] = []
            all_usd_high: List[Dict[str, Any]] = []

            for ev in raw_all:
                country = str(ev.get("country", "")).upper().strip()
                impact = str(ev.get("impact", "")).strip().lower()
                if country != "USD" or impact != "high":
                    continue
                dt_utc = parse_iso_utc(ev.get("date"))
                if dt_utc is None:
                    continue
                title = str(ev.get("title", "USD High Impact")).strip()
                time_utc = to_iso_z(dt_utc)
                dedup_key = (time_utc, title.lower())
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                is_core = is_core_us_macro_title(title, self.extra_patterns)
                rec = {
                    "title": title,
                    "country": "USD",
                    "impact": "High",
                    "time_utc": time_utc,
                    "is_core_us_macro": is_core,
                    "forecast": str(ev.get("forecast", "") or ""),
                    "previous": str(ev.get("previous", "") or ""),
                }
                all_usd_high.append(rec)
                if is_core:
                    core_events.append(rec)

            core_events.sort(key=lambda x: x["time_utc"])
            all_usd_high.sort(key=lambda x: x["time_utc"])

            payload: Dict[str, Any] = {
                "fetched_at_utc": to_iso_z(cur_now),
                "source_urls": fetched_sources,
                "status": "OK",
                "coverage_from_utc": cov["coverage_from_utc"],
                "coverage_to_utc": cov["coverage_to_utc"],
                "core_count": len(core_events),
                "usd_high_count": len(all_usd_high),
                "events": core_events,
                "all_usd_high_events": all_usd_high,
                "last_error": "; ".join(errors) if errors else None,
            }
            _atomic_write_json(self.macro_events_path, payload)
            logger.info(
                "US Macro Shield: refreshed %s (%d core FOMC/CPI/NFP/PPI events, %d total USD High).",
                self.macro_events_path,
                len(core_events),
                len(all_usd_high),
            )
            return payload, True, None

        err_summary = "; ".join(errors) if errors else "unknown fetch failure"
        if cached is not None:
            age_h = self._cache_age_hours(cached, cur_now)
            if not self._covers(cached, cur_now):
                msg = f"Cached macro calendar does not cover the current date and refresh failed: {err_summary}"
                logger.error("US Macro Shield FAIL-SAFE: %s", msg)
                return cached, False, msg
            if age_h <= self.max_stale_hours:
                logger.warning(
                    "US Macro Shield: live refresh failed (%s); using cached calendar (age %.1fh <= %.1fh).",
                    err_summary, age_h, self.max_stale_hours,
                )
                return cached, True, err_summary
            msg = f"Macro calendar stale (age {age_h:.1f}h > {self.max_stale_hours:.1f}h) and refresh failed: {err_summary}"
            logger.error("US Macro Shield FAIL-SAFE: %s", msg)
            return cached, False, msg

        msg = f"Macro calendar unavailable (no local cache) and refresh failed: {err_summary}"
        logger.error("US Macro Shield FAIL-SAFE: %s", msg)
        return None, False, msg

    # ------------------------------------------------------------------
    # Hard Core Shield Evaluation (-30m..+15m block, -30m protect, fail-closed)
    # ------------------------------------------------------------------
    def evaluate_macro_shield(
        self,
        now: Optional[datetime] = None,
        allow_network_refresh: bool = True,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate hard US Macro Shield rules at `now` (UTC):
          - `-30m .. +15m` UTC around any core event (FOMC/CPI/NFP/PPI/Fed Chair):
            `entry_allowed=False`, `cancel_pending_limits=True`.
          - `-30m .. 0m` UTC before release:
            `protect_open_positions=True` (close open position if `uPnL < +0.25%`).
          - Missing / corrupt / stale (>36h) calendar:
            `entry_allowed=False`, `fail_safe_active=True` (fail-closed!).
        """
        cur_now = now or utc_now()
        if not self.enabled:
            return {
                "enabled": False,
                "trading_allowed": True,
                "entry_allowed": True,
                "cancel_pending_limits": False,
                "protect_open_positions": False,
                "fail_safe_active": False,
                "pause_until_utc": None,
                "reason": None,
                "layer": None,
                "active_events": [],
                "upcoming_core_events": [],
                "next_event": None,
                "checked_at_utc": to_iso_z(cur_now),
            }

        if allow_network_refresh:
            payload, is_fresh_ok, err_msg = self.refresh_calendar(force=force_refresh, now=cur_now)
        else:
            payload = self._read_cached_calendar()
            if payload is None:
                is_fresh_ok, err_msg = False, "Local macro_events.json missing or corrupted"
            else:
                age_h = self._cache_age_hours(payload, cur_now)
                is_fresh_ok = age_h <= self.max_stale_hours and self._covers(payload, cur_now)
                err_msg = None if is_fresh_ok else f"Local macro_events.json stale or not covering now (age {age_h:.1f}h)"

        if not is_fresh_ok or payload is None:
            reason = f"FAIL-SAFE BLOCK: {err_msg or 'macro calendar unavailable'}"
            return {
                "enabled": True,
                "trading_allowed": False,
                "entry_allowed": False,
                "cancel_pending_limits": False,
                "protect_open_positions": False,
                "fail_safe_active": True,
                "pause_until_utc": None,
                "reason": reason,
                "layer": "MACRO_SHIELD_FAIL_SAFE",
                "active_events": [],
                "upcoming_core_events": [],
                "next_event": None,
                "checked_at_utc": to_iso_z(cur_now),
            }

        events: List[Dict[str, Any]] = payload.get("events", [])
        pre_td = timedelta(minutes=self.pre_window_min)
        post_td = timedelta(minutes=self.post_window_min)

        active_events: List[Dict[str, Any]] = []
        protect_open = False
        pause_until: Optional[datetime] = None
        upcoming: List[Dict[str, Any]] = []

        for ev in events:
            if not isinstance(ev, dict):
                continue
            ev_dt = parse_iso_utc(ev.get("time_utc"))
            if ev_dt is None:
                continue
            win_start = ev_dt - pre_td
            win_end = ev_dt + post_td

            if ev_dt >= cur_now - timedelta(hours=2):
                upcoming.append(ev)

            if win_start <= cur_now <= win_end:
                active_events.append(ev)
                if pause_until is None or win_end > pause_until:
                    pause_until = win_end
                # Pre-release window (-30m .. 0m): protect open positions with uPnL < +0.25%
                if win_start <= cur_now <= ev_dt:
                    protect_open = True

        in_window = len(active_events) > 0
        reason = None
        if in_window:
            ev_desc = "; ".join(f"{e.get('title')} ({e.get('time_utc')})" for e in active_events)
            reason = f"US Macro Shield (-{self.pre_window_min}m..+{self.post_window_min}m): {ev_desc}"

        return {
            "enabled": True,
            "trading_allowed": not in_window,
            "entry_allowed": not in_window,
            "cancel_pending_limits": in_window,
            "protect_open_positions": protect_open,
            "fail_safe_active": False,
            "pause_until_utc": to_iso_z(pause_until) if (in_window and pause_until is not None) else None,
            "reason": reason,
            "layer": "US_MACRO_SHIELD" if in_window else None,
            "active_events": active_events,
            "upcoming_core_events": upcoming,
            "all_usd_high_count": int(payload.get("usd_high_count", 0)),
            "calendar_fetched_at_utc": payload.get("fetched_at_utc"),
            "next_event": upcoming[0] if upcoming else None,
            "checked_at_utc": to_iso_z(cur_now),
        }

    def update(self, force_refresh: bool = False, now: Optional[datetime] = None, **_kwargs: Any) -> Dict[str, Any]:
        """Refresh calendar if due, evaluate shield, write `data/news_state.json`, and fire alert once per block."""
        state = self.evaluate_macro_shield(now=now, allow_network_refresh=True, force_refresh=force_refresh)
        state["updated_at"] = state["checked_at_utc"]

        _atomic_write_json(self.state_path, state)

        if not state["entry_allowed"]:
            key = f"{state['layer']}:{state['pause_until_utc']}:{state['reason']}"
            if key != self._last_logged_key:
                self._last_logged_key = key
                level = logging.ERROR if state["fail_safe_active"] else logging.WARNING
                logger.log(
                    level,
                    "US Macro Shield ACTIVE | entry_allowed=False | cancel_limits=%s | protect_open=%s | %s",
                    state["cancel_pending_limits"],
                    state["protect_open_positions"],
                    state["reason"],
                )
                if self.on_pause_callback is not None:
                    try:
                        self.on_pause_callback(
                            str(state["layer"] or "US_MACRO_SHIELD"),
                            str(state["pause_until_utc"] or "UNTIL_CALENDAR_REFRESH"),
                            str(state["reason"] or "Macro Shield active"),
                        )
                    except Exception as exc:
                        logger.warning("US Macro Shield pause callback error: %s", exc)
        else:
            self._last_logged_key = None

        return state


def minutes_to_next_event(state: Dict[str, Any], now: Optional[datetime] = None) -> Optional[float]:
    """Minutes until the nearest core event that has not started yet, from an evaluated shield state."""
    cur_now = now or utc_now()
    best: Optional[float] = None
    for ev in state.get("upcoming_core_events") or []:
        if not isinstance(ev, dict):
            continue
        ev_dt = parse_iso_utc(ev.get("time_utc"))
        if ev_dt is None or ev_dt < cur_now:
            continue
        mins = (ev_dt - cur_now).total_seconds() / 60.0
        if best is None or mins < best:
            best = mins
    return best
