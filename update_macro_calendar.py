"""
Weekly Auto-Updater for `config/macro_events_utc.json` (US FOMC / CPI / NFP in exact UTC).

Intended to run once per week via cron or systemd timer:
    0 3 * * 1 /path/to/.venv/bin/python /path/to/update_macro_calendar.py

Safety guarantees:
  - Uses ANTHROPIC_API_KEY from environment / .env (NEVER hardcoded).
  - Requests the next 60 days of official US FOMC, CPI, and NFP release timestamps in UTC.
  - Strictly validates JSON structure, event types, ISO-8601 UTC format, and non-empty horizon.
  - Writes atomically via temp file + os.replace(); on ANY API/network/validation failure,
    the existing `config/macro_events_utc.json` is NEVER overwritten.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DEFAULT_CALENDAR_PATH = ROOT / "config" / "macro_events_utc.json"
ALLOWED_TYPES = {"FOMC", "CPI", "NFP"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | macro_updater | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("macro_updater")


def parse_utc_iso(raw: str) -> datetime:
    """Strictly parse an ISO-8601 UTC timestamp ending in 'Z' or '+00:00'."""
    s = str(raw or "").strip()
    if not s:
        raise ValueError("Empty UTC timestamp")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        raise ValueError(f"Timestamp missing timezone offset: {raw}")
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc


def validate_calendar_payload(
    payload: Dict[str, Any],
    *,
    now_utc: datetime | None = None,
    min_future_events: int = 1,
) -> Dict[str, Any]:
    """Validate and normalize a macro calendar JSON payload. Raises ValueError on any issue."""
    if not isinstance(payload, dict):
        raise ValueError("Calendar payload must be a JSON object")

    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("Calendar 'events' must be a non-empty list")

    now = now_utc or datetime.now(timezone.utc)
    seen: set[Tuple[str, str]] = set()
    cleaned: List[Dict[str, str]] = []
    future_count = 0

    for idx, ev in enumerate(raw_events):
        if not isinstance(ev, dict):
            raise ValueError(f"Event #{idx} is not an object")
        ev_type = str(ev.get("type", "")).strip().upper()
        if ev_type not in ALLOWED_TYPES:
            raise ValueError(f"Event #{idx} has invalid type '{ev_type}' (allowed: {sorted(ALLOWED_TYPES)})")
        title = str(ev.get("title", "")).strip() or f"US {ev_type}"
        dt_utc = parse_utc_iso(str(ev.get("utc", "")))
        if dt_utc.year < 2025 or dt_utc.year > 2035:
            raise ValueError(f"Event #{idx} year out of plausible range: {dt_utc.isoformat()}")
        iso_z = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        dedup_key = (ev_type, iso_z)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        if dt_utc > now:
            future_count += 1
        cleaned.append({"type": ev_type, "title": title, "utc": iso_z})

    if future_count < min_future_events:
        raise ValueError(f"Expected at least {min_future_events} future event(s) after {now.isoformat()}, got {future_count}")

    cleaned.sort(key=lambda x: x["utc"])
    bw = payload.get("blackout_window") or {}
    pre_m = int(bw.get("pre_minutes", 60))
    post_m = int(bw.get("post_minutes", 45))
    if not (5 <= pre_m <= 240 and 5 <= post_m <= 240):
        raise ValueError(f"blackout_window out of bounds: pre={pre_m}, post={post_m}")

    return {
        "schema_version": 1,
        "updated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(payload.get("source") or "Claude API macro schedule updater"),
        "blackout_window": {
            "pre_minutes": pre_m,
            "post_minutes": post_m,
        },
        "events": cleaned,
    }


def _extract_json_block(text: str) -> Dict[str, Any]:
    """Extract first valid JSON object from Claude response text (supports fenced code blocks)."""
    s = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    else:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end > start:
            s = s[start : end + 1]
    return json.loads(s)


def fetch_events_from_claude(
    *,
    api_key: str,
    model: str,
    now_utc: datetime,
    horizon_days: int = 60,
    timeout_sec: int = 45,
) -> Dict[str, Any]:
    """Call Anthropic Messages API to fetch official FOMC, CPI, and NFP timestamps in UTC."""
    end_utc = now_utc + timedelta(days=horizon_days)
    prompt = (
        f"Provide the official scheduled release timestamps in exact UTC (ISO-8601 ending with Z) "
        f"for ALL major US macro events between {now_utc.strftime('%Y-%m-%d')} and {end_utc.strftime('%Y-%m-%d')} "
        f"(inclusive, {horizon_days} days ahead) for ONLY these three event types:\n"
        f"1. FOMC — Federal Reserve interest rate decision & statement (14:00 America/New_York -> 19:00Z EST or 18:00Z EDT)\n"
        f"2. CPI — US BLS Consumer Price Index release (08:30 America/New_York -> 13:30Z EST or 12:30Z EDT)\n"
        f"3. NFP — US BLS Employment Situation / Non-Farm Payrolls (08:30 America/New_York -> 13:30Z EST or 12:30Z EDT)\n\n"
        f"Account carefully for US Daylight Saving Time (EST=UTC-5 vs EDT=UTC-4).\n"
        f"Return STRICTLY a valid JSON object with NO commentary:\n"
        f'{{"source": "Claude API (Federal Reserve / BLS)", '
        f'"blackout_window": {{"pre_minutes": 60, "post_minutes": 45}}, '
        f'"events": [{{"type": "NFP", "title": "US Non-Farm Payrolls (...)", "utc": "YYYY-MM-DDTHH:MM:SSZ"}}]}}'
    )

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 1500,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    data = resp.json()
    content_blocks = data.get("content", [])
    text_parts = [b.get("text", "") for b in content_blocks if isinstance(b, dict) and b.get("type") == "text"]
    raw_text = "\n".join(text_parts).strip()
    if not raw_text:
        raise ValueError(f"Empty text response from Claude API: {data}")
    return _extract_json_block(raw_text)


def merge_and_prune_events(
    existing: Dict[str, Any] | None,
    fresh: Dict[str, Any],
    now_utc: datetime,
) -> Dict[str, Any]:
    """Keep recent past events (last 7 days for auditability) + all validated upcoming events."""
    cutoff = now_utc - timedelta(days=7)
    combined: List[Dict[str, str]] = []
    if existing and isinstance(existing.get("events"), list):
        for ev in existing["events"]:
            try:
                dt = parse_utc_iso(str(ev.get("utc", "")))
                if cutoff <= dt <= now_utc:
                    combined.append(ev)
            except Exception:
                continue
    combined.extend(fresh["events"])
    merged_payload = dict(fresh)
    merged_payload["events"] = combined
    return validate_calendar_payload(merged_payload, now_utc=now_utc, min_future_events=1)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    try:
        tmp_path.write_text(serialized, encoding="utf-8")
        # Double-check written temp file parses cleanly before replacing target
        json.loads(tmp_path.read_text(encoding="utf-8"))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main() -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(description="Update config/macro_events_utc.json via Claude API (60d horizon)")
    p.add_argument("--output", default=str(DEFAULT_CALENDAR_PATH), help="Target macro calendar JSON path")
    p.add_argument("--horizon-days", type=int, default=60, help="Forward horizon in days (default: 60)")
    p.add_argument("--model", default="", help="Override ANTHROPIC_MODEL")
    p.add_argument("--validate-only", action="store_true", help="Validate existing calendar file and exit")
    p.add_argument("--dry-run", action="store_true", help="Fetch and validate via Claude without writing file")
    args = p.parse_args()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    now_utc = datetime.now(timezone.utc)

    if args.validate_only:
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            validated = validate_calendar_payload(existing, now_utc=now_utc, min_future_events=1)
            logger.info("Calendar %s is VALID (%d events).", out_path, len(validated["events"]))
            return 0
        except Exception as exc:
            logger.error("Calendar validation FAILED for %s: %s", out_path, exc)
            return 1

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key or "your_anthropic" in api_key.lower():
        logger.error("ANTHROPIC_API_KEY is missing/placeholder in .env — keeping existing %s untouched.", out_path)
        return 1

    model = (args.model or os.getenv("ANTHROPIC_MODEL", "") or "claude-sonnet-4-5").strip()
    existing_data: Dict[str, Any] | None = None
    if out_path.exists():
        try:
            existing_data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Existing calendar unreadable (%s); will require valid fresh payload.", exc)

    try:
        logger.info("Requesting %dd US macro calendar (FOMC/CPI/NFP) from Claude model=%s ...", args.horizon_days, model)
        raw_fresh = fetch_events_from_claude(
            api_key=api_key,
            model=model,
            now_utc=now_utc,
            horizon_days=args.horizon_days,
        )
        fresh_valid = validate_calendar_payload(raw_fresh, now_utc=now_utc, min_future_events=1)
        final_payload = merge_and_prune_events(existing_data, fresh_valid, now_utc)
    except Exception as exc:
        logger.error("Macro calendar update FAILED (%s) — existing %s preserved unchanged.", exc, out_path)
        return 1

    if args.dry_run:
        logger.info("[DRY-RUN] Validated %d events (not written):\n%s", len(final_payload["events"]), json.dumps(final_payload, indent=2))
        return 0

    try:
        atomic_write_json(out_path, final_payload)
        logger.info(
            "Updated %s successfully: %d events (latest=%s).",
            out_path,
            len(final_payload["events"]),
            final_payload["events"][-1]["utc"],
        )
        return 0
    except Exception as exc:
        logger.error("Atomic write failed (%s) — existing file preserved.", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
