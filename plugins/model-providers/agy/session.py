"""Opt-in pool of long-lived AGY stream-json processes (prototype).

One AGY process holds one conversation. A request may reuse an idle process
only when its messages extend, unchanged, the conversation that process has
already seen; anything else starts a fresh process. A process is never put back
into the pool after a failed, timed-out, or rejected turn, because AGY keeps
running an unfinished turn and would merge it into the next one.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Mapping
from typing import Any

PERSISTENT_ENV = "HERMES_AGY_PERSISTENT"
IDLE_SECONDS_ENV = "HERMES_AGY_SESSION_IDLE_SECONDS"
MAX_SESSIONS_ENV = "HERMES_AGY_MAX_SESSIONS"
DEFAULT_IDLE_SECONDS = 900.0
DEFAULT_MAX_SESSIONS = 4
# AGY applies --print-timeout per turn and, when it fires, reports SUCCESS with
# partial output while the turn keeps running. The pool enforces its own turn
# deadline and kills the process instead, so AGY's timer must never fire first.
PRINT_TIMEOUT = "86400s"
_EOF = object()


def persistent_enabled() -> bool:
    return os.environ.get(PERSISTENT_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _env_number(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


class SessionTimeout(Exception):
    """The turn did not finish before its deadline."""


class SessionOverflow(Exception):
    """The turn produced more stdout than allowed."""


class SessionDied(Exception):
    """The AGY process exited or closed its pipes."""

    def __init__(self, returncode: int | None, stderr_tail: str) -> None:
        super().__init__(f"AGY session exited with status {returncode}")
        self.returncode = returncode
        self.stderr_tail = stderr_tail


def _ends_turn(line: bytes) -> bool:
    try:
        event = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return True  # let the stream parser reject it instead of waiting
    return not isinstance(event, dict) or event.get("event") == "result"


class AGYSession:
    """A running AGY process in stream-json input mode."""

    def __init__(
        self,
        key: tuple[Any, ...],
        argv: list[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        max_stderr_bytes: int,
        terminate_grace: float,
    ) -> None:
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        self.key = key
        self.history: list[dict[str, Any]] | None = None
        self.reply_ids: tuple[str, ...] = ()
        self.turns = 0
        self.last_used = time.monotonic()
        self.terminate_grace = terminate_grace
        self._lines: queue.Queue[Any] = queue.Queue()
        self._stderr = bytearray()
        self._max_stderr_bytes = max_stderr_bytes
        self._err_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self._err_thread.start()

    def _read_stdout(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            for line in iter(self.process.stdout.readline, b""):
                if line.strip():
                    self._lines.put(line)
        self._lines.put(_EOF)

    def _drain_stderr(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            for chunk in iter(lambda: self.process.stderr.read(65536), b""):
                self._stderr.extend(chunk)
                if len(self._stderr) > self._max_stderr_bytes:
                    del self._stderr[: -self._max_stderr_bytes]

    @property
    def alive(self) -> bool:
        return self.process.poll() is None

    def _died(self) -> SessionDied:
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.process.wait(timeout=self.terminate_grace)
        self._err_thread.join(timeout=self.terminate_grace)
        return SessionDied(
            self.process.poll(), self._stderr.decode("utf-8", errors="replace")
        )

    def run_turn(self, content: str, timeout: float, max_stdout_bytes: int) -> bytes:
        """Send one user message and return the stdout lines of its turn."""
        deadline = time.monotonic() + timeout
        if not self.alive:
            raise self._died()
        payload = (
            json.dumps(
                {"event": "user", "message": {"content": content}}, ensure_ascii=False
            )
            + "\n"
        ).encode("utf-8")
        write_errors: list[BaseException] = []

        def write() -> None:
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                write_errors.append(exc)

        writer = threading.Thread(target=write, daemon=True)
        writer.start()
        writer.join(max(0.0, deadline - time.monotonic()))
        if writer.is_alive():
            raise SessionTimeout()
        if write_errors:
            raise self._died()

        collected: list[bytes] = []
        size = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SessionTimeout()
            try:
                line = self._lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is _EOF:
                raise self._died()
            size += len(line)
            if size > max_stdout_bytes:
                raise SessionOverflow()
            collected.append(line)
            if _ends_turn(line):
                break
        self.turns += 1
        self.last_used = time.monotonic()
        return b"".join(collected)

    def stop(self) -> None:
        """Kill the process group; safe to call more than once."""
        with contextlib.suppress(OSError, ValueError):
            self.process.stdin.close()
        if self.process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:  # pragma: no cover - exercised on Windows CI
                    self.process.terminate()
                self.process.wait(timeout=self.terminate_grace)
            except (OSError, subprocess.TimeoutExpired):
                with contextlib.suppress(OSError):
                    if os.name == "posix":
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:  # pragma: no cover - exercised on Windows CI
                        self.process.kill()
        with contextlib.suppress(Exception):
            self.process.wait(timeout=self.terminate_grace)


def _delta(
    session: AGYSession, messages: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """New messages after the session's last reply, or None if they diverge."""
    previous = session.history
    if previous is None or len(messages) <= len(previous) + 1:
        return None
    if messages[: len(previous)] != previous:
        return None
    reply = messages[len(previous)]
    if reply.get("role") != "assistant":
        return None
    reply_ids = tuple(str(call.get("id", "")) for call in reply.get("tool_calls", []))
    if reply_ids != session.reply_ids:
        return None
    delta = messages[len(previous) + 1 :]
    if any(message.get("role") in {"system", "developer"} for message in delta):
        return None
    return delta


class SessionPool:
    """Idle AGY sessions waiting for the next turn of their conversation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._idle: list[AGYSession] = []
        self._reaper: threading.Thread | None = None

    def checkout(
        self, key: tuple[Any, ...], messages: list[dict[str, Any]]
    ) -> tuple[AGYSession | None, list[dict[str, Any]] | None, str]:
        """Take the idle session that continues ``messages``, if any."""
        expired = self._collect_expired()
        try:
            with self._lock:
                same_key = False
                for session in self._idle:
                    if session.key != key:
                        continue
                    same_key = True
                    delta = _delta(session, messages)
                    if delta is not None:
                        self._idle.remove(session)
                        return session, delta, "reuse"
                if same_key:
                    reason = "history diverged"
                elif self._idle:
                    reason = "model, tools or environment changed"
                else:
                    reason = "no idle session"
                return None, None, reason
        finally:
            for session in expired:
                session.stop()

    def checkin(self, session: AGYSession) -> None:
        evicted: list[AGYSession] = []
        if session.alive:
            with self._lock:
                self._idle.append(session)
                limit = int(_env_number(MAX_SESSIONS_ENV, DEFAULT_MAX_SESSIONS))
                self._idle.sort(key=lambda item: item.last_used)
                while len(self._idle) > limit:
                    evicted.append(self._idle.pop(0))
                if self._reaper is None:
                    self._reaper = threading.Thread(
                        target=self._reap_forever,
                        name="agy-session-reaper",
                        daemon=True,
                    )
                    self._reaper.start()
        else:
            evicted.append(session)
        for item in evicted:
            item.stop()

    def _collect_expired(self) -> list[AGYSession]:
        idle_limit = _env_number(IDLE_SECONDS_ENV, DEFAULT_IDLE_SECONDS)
        now = time.monotonic()
        with self._lock:
            expired = [
                session
                for session in self._idle
                if not session.alive or now - session.last_used > idle_limit
            ]
            for session in expired:
                self._idle.remove(session)
        return expired

    def _reap_forever(self) -> None:
        while True:
            idle_limit = _env_number(IDLE_SECONDS_ENV, DEFAULT_IDLE_SECONDS)
            time.sleep(min(30.0, max(0.05, idle_limit / 2)))
            for session in self._collect_expired():
                session.stop()

    def clear(self) -> None:
        with self._lock:
            sessions, self._idle = self._idle, []
        for session in sessions:
            session.stop()


POOL = SessionPool()
atexit.register(POOL.clear)
