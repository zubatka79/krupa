"""Keeps the Docker deployment honest about the bot process.

Docker restarts a container only when its process exits, so a bot stuck on a hung
request would sit there silently. StallGuard notices that the main loop stopped
ticking and exits the process so the restart policy brings it back. RunState
remembers whether the previous run ended cleanly, which lets the startup message
say "the last run crashed" and lets crash loops reach Telegram.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger("vika.liveness")

UNCLEAN_WINDOW_SEC = 30 * 60
FATAL_ALERT_COOLDOWN_SEC = 10 * 60


def git_version(root: Path) -> str:
    """Short commit hash from a .git directory (mounted read-only by docker-compose), or 'unknown'."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head[:7] or "unknown"
        ref = head.split(":", 1)[1].strip()
        ref_file = git / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip()[:7]
        packed = git / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0][:7]
    except OSError:
        pass
    return "unknown"


def telegram_chat_ids() -> List[str]:
    raw = os.getenv("TELEGRAM_CHAT_ID", "") or os.getenv("TELEGRAM_ALLOWED_USER_IDS", "")
    return [x.strip() for x in raw.split(",") if x.strip() and "your_" not in x.lower()]


def send_telegram_plain(text: str, timeout: float = 10.0) -> bool:
    """Minimal sender that works even when the Telegram UI object was never built."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chats = telegram_chat_ids()
    if not token or "your_" in token.lower() or not chats:
        return False
    sent = False
    for chat in chats:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat, "text": text, "disable_web_page_preview": True},
                timeout=timeout,
            )
            sent = sent or r.ok
        except requests.RequestException as exc:
            logger.warning("Telegram alert failed: %s", type(exc).__name__)
    return sent


class RunState:
    """Small JSON file next to the trades DB: was the previous run stopped cleanly?"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.state: Dict[str, Any] = {}
        self.previous_unclean = False
        self.recent_unclean = 0

    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            logger.warning("Run state not saved: %s", exc)

    def start(self, version: str, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        prev = self._load()
        self.previous_unclean = bool(prev) and not prev.get("clean_exit", False)
        starts = [t for t in prev.get("unclean_starts", []) if now - float(t) < UNCLEAN_WINDOW_SEC]
        if self.previous_unclean:
            starts.append(now)
        self.recent_unclean = len(starts)
        self.state = {
            "clean_exit": False,
            "started_at": now,
            "version": version,
            "unclean_starts": starts,
            "last_fatal_alert": float(prev.get("last_fatal_alert", 0.0)),
        }
        self._save()

    def mark_clean_exit(self) -> None:
        self.state["clean_exit"] = True
        self._save()

    def fatal_alert_allowed(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if now - float(self.state.get("last_fatal_alert", 0.0)) < FATAL_ALERT_COOLDOWN_SEC:
            return False
        self.state["last_fatal_alert"] = now
        self._save()
        return True

    def startup_note(self) -> str:
        if not self.previous_unclean:
            return ""
        if self.recent_unclean >= 3:
            return f"⚠️ Бот перезапускается по кругу: {self.recent_unclean} аварийных запусков за 30 мин. Проверьте логи."
        return "⚠️ Прошлый запуск завершился аварийно (падение или зависание), бот перезапущен."


class StallGuard:
    """Exits the process when the main loop has not called beat() for stall_sec seconds."""

    def __init__(
        self,
        stall_sec: float,
        on_stall: Callable[[float], None],
        exit_fn: Callable[[int], None] = os._exit,
        check_every: float = 15.0,
    ) -> None:
        self.stall_sec = stall_sec
        self.on_stall = on_stall
        self.exit_fn = exit_fn
        self.check_every = check_every
        self._last = time.monotonic()
        self._stop = threading.Event()

    def beat(self) -> None:
        self._last = time.monotonic()

    def start(self) -> None:
        threading.Thread(target=self._run, name="stall-guard", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.check_every):
            age = time.monotonic() - self._last
            if age < self.stall_sec:
                continue
            logger.critical("Main loop stalled for %.0fs, exiting so Docker restarts the bot.", age)
            alert = threading.Thread(target=self.on_stall, args=(age,), daemon=True)
            alert.start()
            alert.join(timeout=20)
            self.exit_fn(3)
            return
