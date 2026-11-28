"""
Thread-safe US Macro News Calendar Guard (`config/macro_events_utc.json`) for L2_news.

Rules (verified in Step 14/16/17):
  - Blocks ONLY new trade entries in [T - 60m .. T + 45m] UTC around FOMC, CPI, and NFP.
  - Never touches open positions, BE transitions, or server-side TP/SL.
  - Auto-reloads JSON when file mtime changes (no bot restart needed after weekly cron update).
  - Emits a horizon warning when the farthest scheduled event is < 3 days ahead (or missing).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("vika.macro_calendar")


@dataclass(frozen=True)
class MacroEvent:
    event_type: str
    title: str
    dt_utc: datetime
    iso_utc: str


@dataclass(frozen=True)
class MacroCalendarStatus:
    in_blackout: bool
    active_event: Optional[MacroEvent]
    blackout_end_utc: Optional[datetime]
    next_event: Optional[MacroEvent]
    upcoming_events: Tuple[MacroEvent, ...]
    horizon_days: float
    horizon_warning: bool
    pre_minutes: int
    post_minutes: int
    calendar_error: Optional[str] = None

    @property
    def short_label(self) -> str:
        if self.calendar_error:
            return f"⚠️ Ошибка календаря ({self.calendar_error})"
        if self.in_blackout and self.active_event is not None:
            end_s = self.blackout_end_utc.strftime("%H:%M UTC") if self.blackout_end_utc else "?"
            return f"🚫 ПАУЗА [{self.active_event.event_type}] до {end_s}"
        if self.next_event is not None:
            dt_s = self.next_event.dt_utc.strftime("%d.%m %H:%M UTC")
            warn = " ⚠️ горизонт <3д!" if self.horizon_warning else ""
            return f"🟢 Открыто | След.: {self.next_event.event_type} {dt_s}{warn}"
        return "⚠️ Нет будущих событий (обнови календарь!)"


def _parse_iso_utc(raw: str) -> datetime:
    s = str(raw or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MacroCalendarGuard:
    def __init__(
        self,
        calendar_path: str | Path,
        *,
        pre_minutes: int = 60,
        post_minutes: int = 45,
        horizon_warn_days: float = 3.0,
    ) -> None:
        self.path = Path(calendar_path)
        self.default_pre = int(pre_minutes)
        self.default_post = int(post_minutes)
        self.horizon_warn_days = float(horizon_warn_days)

        self._lock = threading.RLock()
        self._mtime: float = -1.0
        self._events: List[MacroEvent] = []
        self._pre_min: int = self.default_pre
        self._post_min: int = self.default_post
        self._last_error: Optional[str] = None
        self._reload_if_needed(force=True)

    def _reload_if_needed(self, force: bool = False) -> None:
        with self._lock:
            if not self.path.exists():
                self._last_error = f"Файл не найден: {self.path.name}"
                return
            try:
                st = self.path.stat()
                if (not force) and st.st_mtime == self._mtime:
                    return
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                bw = raw.get("blackout_window") or {}
                pre_m = int(bw.get("pre_minutes", self.default_pre))
                post_m = int(bw.get("post_minutes", self.default_post))
                parsed: List[MacroEvent] = []
                for item in raw.get("events", []):
                    ev_type = str(item.get("type", "")).strip().upper()
                    title = str(item.get("title", "")).strip() or ev_type
                    iso = str(item.get("utc", "")).strip()
                    dt_utc = _parse_iso_utc(iso)
                    parsed.append(MacroEvent(event_type=ev_type, title=title, dt_utc=dt_utc, iso_utc=dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")))
                parsed.sort(key=lambda e: e.dt_utc)
                self._events = parsed
                self._pre_min = pre_m
                self._post_min = post_m
                self._mtime = st.st_mtime
                self._last_error = None
                logger.info(
                    "Loaded macro calendar %s: %d events (window [-%dm..+%dm]).",
                    self.path.name, len(parsed), pre_m, post_m,
                )
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("Failed loading macro calendar %s (keeping previous %d events): %s", self.path, len(self._events), exc)

    def evaluate(self, now_utc: Optional[datetime] = None) -> MacroCalendarStatus:
        self._reload_if_needed()
        now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._lock:
            events = list(self._events)
            pre_m = self._pre_min
            post_m = self._post_min
            err = self._last_error

        pre_td = timedelta(minutes=pre_m)
        post_td = timedelta(minutes=post_m)

        active_ev: Optional[MacroEvent] = None
        blackout_end: Optional[datetime] = None
        upcoming: List[MacroEvent] = []

        for ev in events:
            win_start = ev.dt_utc - pre_td
            win_end = ev.dt_utc + post_td
            if win_start <= now <= win_end:
                if active_ev is None or win_end > (blackout_end or win_end):
                    active_ev = ev
                    blackout_end = win_end
            if ev.dt_utc >= now:
                upcoming.append(ev)

        if upcoming:
            farthest_dt = upcoming[-1].dt_utc
            horizon_days = round((farthest_dt - now).total_seconds() / 86400.0, 2)
        else:
            horizon_days = 0.0

        horizon_warning = (horizon_days < self.horizon_warn_days) or (len(upcoming) == 0)
        return MacroCalendarStatus(
            in_blackout=active_ev is not None,
            active_event=active_ev,
            blackout_end_utc=blackout_end,
            next_event=upcoming[0] if upcoming else None,
            upcoming_events=tuple(upcoming[:6]),
            horizon_days=horizon_days,
            horizon_warning=horizon_warning,
            pre_minutes=pre_m,
            post_minutes=post_m,
            calendar_error=err,
        )
