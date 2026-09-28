"""Process-wide admission control for AGY requests."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterator

from .protocol import AGYBusyError, AGYTimeoutError


class Admission:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._sessions: set[tuple[str, ...]] = set()

    @contextmanager
    def enter(self, key: tuple[str, ...] | None, deadline: float) -> Iterator[None]:
        try:
            limit = max(1, int(os.environ.get("HERMES_AGY_MAX_ACTIVE", "4")))
        except ValueError:
            limit = 4
        queue_deadline = min(deadline, time.monotonic() + 5.0)
        with self._condition:
            # Serializing stale parallel histories still overwrites state: reject
            # a second turn so its caller can retry with the completed history.
            if key is not None and key in self._sessions:
                raise AGYBusyError("AGY conversation already has a request in flight")
            if key is not None:
                self._sessions.add(key)
            try:
                while self._active >= limit:
                    remaining = queue_deadline - time.monotonic()
                    if remaining <= 0:
                        if time.monotonic() < deadline:
                            raise AGYBusyError("AGY capacity unavailable; retry later")
                        raise AGYTimeoutError("AGY request expired waiting for capacity")
                    self._condition.wait(remaining)
                self._active += 1
            except BaseException:
                self._sessions.discard(key)
                raise
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._sessions.discard(key)
                self._condition.notify_all()


ADMISSION = Admission()
