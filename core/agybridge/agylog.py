"""Early detection of a spent AGY quota from the log file AGY writes.

AGY retries a quota 429 with growing backoff for minutes (about 3 in one-shot
mode) before it reports the failure on stdout, and its stream only says that
an error step happened, not which one. The text is written to AGY's log right
away, so each process gets its own ``--log-file`` that the bridge tails.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import time
from pathlib import Path
from typing import BinaryIO

from .protocol import quota_message

LOG_DIR_ENV = "HERMES_AGY_LOG_DIR"
LOG_PREFIX = "agybridge-"
# Logs are kept for diagnostics, like AGY's own cli-*.log, but bounded.
KEEP_LOGS = 50
CHECK_INTERVAL = 0.25
_MAX_PENDING = 64 * 1024

_counter = itertools.count(1)


def log_dir() -> Path:
    """Where per-process AGY logs go: next to AGY's own logs unless overridden."""
    override = os.environ.get(LOG_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".gemini" / "antigravity-cli" / "log"


def _prune(directory: Path) -> None:
    with contextlib.suppress(OSError):
        logs = sorted(
            directory.glob(f"{LOG_PREFIX}*.log"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in logs[KEEP_LOGS:]:
            with contextlib.suppress(OSError):
                stale.unlink()


def new_log_path() -> Path | None:
    """A fresh log path for one AGY process, or None if logs cannot be written.

    Without a path AGY simply runs as before, without early quota detection.
    """
    directory = log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        return None
    if not os.access(directory, os.W_OK):
        return None
    _prune(directory)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return directory / f"{LOG_PREFIX}{stamp}-{os.getpid()}-{next(_counter)}.log"


class AGYLogWatcher:
    """Tails one AGY log and reports AGY's quota message once it appears."""

    def __init__(self, path: Path, interval: float = CHECK_INTERVAL) -> None:
        self.path = path
        self.interval = interval
        self._handle: BinaryIO | None = None
        self._pending = b""
        self._next_check = 0.0
        self._found: str | None = None

    def check(self) -> str | None:
        """Read what AGY logged since the last call; cheap to call in a loop."""
        if self._found is not None:
            return self._found
        now = time.monotonic()
        if now < self._next_check:
            return None
        self._next_check = now + self.interval
        if self._handle is None:
            try:
                self._handle = self.path.open("rb")
            except OSError:
                return None  # AGY has not created it yet
        try:
            chunk = self._handle.read()
        except OSError:
            return None
        if not chunk:
            return None
        lines = (self._pending + chunk).split(b"\n")
        self._pending = lines.pop()[-_MAX_PENDING:]
        for line in lines:
            if b"RESOURCE_EXHAUSTED" not in line:
                continue
            message = quota_message(line.decode("utf-8", errors="replace"))
            if message:
                self._found = message
                return message
        return None

    def close(self) -> None:
        if self._handle is not None:
            with contextlib.suppress(OSError):
                self._handle.close()
            self._handle = None
