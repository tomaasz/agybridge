"""Remembering a spent AGY quota until it resets.

Detecting a spent quota still costs an AGY launch and one failed attempt
(about 5 s), and a Hermes gateway sends many requests (auxiliary tasks,
kanban workers) that would each pay it. AGY's message says when the quota
resets, so the bridge records that per model and answers at once until then.

The record lives in a small JSON file shared by every process on the host
(gateway, serve, workers, the HTTP server). One request per probe interval is
still let through, so logging into another account or buying more quota is
noticed without waiting for the reset.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .backends.agy import AGY_EFFORTS
from .protocol import AGYQuotaError

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows CI
    fcntl = None  # type: ignore[assignment]

QUOTA_STATE_ENV = "HERMES_AGY_QUOTA_STATE"
PROBE_SECONDS_ENV = "HERMES_AGY_QUOTA_PROBE_SECONDS"
DEFAULT_PROBE_SECONDS = 900.0
MIN_COOLDOWN_SECONDS = 60.0
MAX_COOLDOWN_SECONDS = 7 * 24 * 3600.0

_RESET_RE = re.compile(
    r"Resets in\s+(?P<span>(?:\d+d)?(?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)",
    re.IGNORECASE,
)
_UNIT_SECONDS = {"d": 86400, "h": 3600, "m": 60, "s": 1}

logger = logging.getLogger(__name__)


def reset_seconds(message: str) -> float | None:
    """Seconds until reset from AGY's "Resets in 93h56m25s", or None."""
    match = _RESET_RE.search(message)
    if not match or not match.group("span"):
        return None
    total = 0.0
    for amount, unit in re.findall(r"(\d+(?:\.\d+)?)([dhms])", match.group("span")):
        total += float(amount) * _UNIT_SECONDS[unit.lower()]
    return total or None


def quota_key(model: str) -> str:
    """Quota bucket for a model: the effort variants of one model share it."""
    base, _, suffix = model.strip().rpartition("-")
    return base if base and suffix in AGY_EFFORTS else model.strip()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m" if minutes else f"{seconds}s"


def state_path() -> Path | None:
    """Where spent quotas are recorded, or None when remembering is off."""
    raw = os.environ.get(QUOTA_STATE_ENV)
    if raw is not None:
        raw = raw.strip()
        if raw.lower() in {"", "0", "off", "false", "no"}:
            return None
        return Path(raw).expanduser()
    home = os.environ.get("HERMES_HOME") or "~/.hermes"
    return Path(home).expanduser() / "state" / "agy-quota.json"


def probe_seconds() -> float:
    try:
        value = float(os.environ.get(PROBE_SECONDS_ENV, ""))
    except ValueError:
        return DEFAULT_PROBE_SECONDS
    return value if value > 0 else DEFAULT_PROBE_SECONDS


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class QuotaBook:
    """Per-model record of a spent quota, shared through a locked JSON file."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _change(self, change: Callable[[dict[str, Any]], Any]) -> Any:
        """Run ``change`` on the current records under an exclusive lock.

        A failure to read or write the file never blocks a request: the bridge
        then behaves as if nothing were remembered.
        """
        path = state_path()
        if path is None:
            return None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with (
                self._lock,
                open(f"{path}.lock", "a+", encoding="utf-8") as lock_file,
            ):
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                data = _load(path)
                before = json.dumps(data, sort_keys=True)
                result = change(data)
                now = time.time()
                data = {
                    key: entry
                    for key, entry in data.items()
                    if isinstance(entry, dict) and float(entry.get("until", 0)) > now
                }
                if json.dumps(data, sort_keys=True) != before:
                    temporary = path.with_name(
                        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                    )
                    descriptor = os.open(
                        temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                    )
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        json.dump(data, handle)
                    os.replace(temporary, path)
                return result
        except (OSError, TypeError, ValueError):
            logger.debug("AGY quota state update failed", exc_info=True)
            return None

    def check(self, model: str) -> bool:
        """Raise AGYQuotaError while ``model``'s quota is remembered as spent.

        Returns True when the request goes ahead as a probe of a remembered
        quota, so a success should call :meth:`clear`.
        """
        key = quota_key(model)

        def change(data: dict[str, Any]) -> AGYQuotaError | bool:
            entry = data.get(key)
            if not isinstance(entry, dict):
                return False
            now = time.time()
            until = float(entry.get("until", 0))
            if until <= now:
                return True
            if now >= float(entry.get("probe_at", 0)):
                # Push the next probe out first, so parallel requests are
                # still answered from the record while this one runs.
                entry["probe_at"] = now + probe_seconds()
                logger.info(
                    "AGY quota for %s remembered as spent for %s more; probing once",
                    key,
                    format_duration(until - now),
                )
                return True
            message = str(entry.get("message") or "quota reached")
            return AGYQuotaError(
                f"AGY quota exhausted (remembered, resets in "
                f"{format_duration(until - now)}): {message}",
                quota_message=message,
                retry_after=until - now,
            )

        result = self._change(change)
        if isinstance(result, AGYQuotaError):
            raise result
        return bool(result)

    def record(self, model: str, message: str) -> float:
        """Remember ``model``'s quota as spent; returns seconds until reset."""
        key = quota_key(model)
        seconds = reset_seconds(message)
        cooldown = min(
            MAX_COOLDOWN_SECONDS,
            max(MIN_COOLDOWN_SECONDS, seconds if seconds else probe_seconds()),
        )
        now = time.time()

        def change(data: dict[str, Any]) -> None:
            data[key] = {
                "until": now + cooldown,
                "probe_at": now + min(cooldown, probe_seconds()),
                "message": message[:500],
                "updated": now,
            }

        self._change(change)
        logger.warning(
            "AGY quota for %s remembered as spent for %s; requests fail at once until then",
            key,
            format_duration(cooldown),
        )
        return cooldown

    def clear(self, model: str | None = None) -> None:
        """Forget the record for ``model``, or every record when None."""

        def change(data: dict[str, Any]) -> None:
            if model is None:
                data.clear()
            else:
                data.pop(quota_key(model), None)

        self._change(change)

    def entries(self) -> dict[str, dict[str, Any]]:
        """Current records (expired ones are dropped)."""
        now = time.time()
        return (
            self._change(
                lambda data: {
                    key: dict(entry)
                    for key, entry in data.items()
                    if isinstance(entry, dict) and float(entry.get("until", 0)) > now
                }
            )
            or {}
        )


QUOTA = QuotaBook()
