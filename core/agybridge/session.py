"""Opt-in pool of long-lived AGY stream-json processes and conversation store."""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows CI
    fcntl = None  # type: ignore[assignment]

PERSISTENT_ENV = "HERMES_AGY_PERSISTENT"
IDLE_SECONDS_ENV = "HERMES_AGY_SESSION_IDLE_SECONDS"
MAX_SESSIONS_ENV = "HERMES_AGY_MAX_SESSIONS"
WARM_SPARE_ENV = "HERMES_AGY_WARM_SPARE"
MAX_SPARES = 2
DEFAULT_IDLE_SECONDS = 900.0
DEFAULT_MAX_SESSIONS = 4
PRINT_TIMEOUT = "86400s"
_EOF = object()

SESSION_STORE_ENV = "HERMES_AGY_SESSION_STORE"
STORE_MAX_ENTRIES = 500
STORE_MAX_AGE_SECONDS = 7 * 24 * 3600
_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def persistent_enabled() -> bool:
    return _env_flag(PERSISTENT_ENV)


def warm_spare_enabled() -> bool:
    """Keep a pre-started AGY process ready for the next new conversation."""
    return _env_flag(WARM_SPARE_ENV)


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
        self.conversation_id: str | None = None
        self.store_key: tuple[str, str] | None = None
        self.turns = 0
        self.from_spare = False
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
                    self._note_conversation(line)
                    self._lines.put(line)
        self._lines.put(_EOF)

    def _note_conversation(self, line: bytes) -> None:
        if self.conversation_id is not None or b"conversation_id" not in line:
            return
        with contextlib.suppress(ValueError):
            event = json.loads(line)
            for field in (None, "init", "result"):
                body = event if field is None else event.get(field)
                value = body.get("conversation_id") if isinstance(body, dict) else None
                if isinstance(value, str) and _CONVERSATION_ID_RE.fullmatch(value):
                    self.conversation_id = value
                    return

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


def history_digest(messages: list[dict[str, Any]]) -> str:
    """Stable digest of normalized messages; the store keeps this, never content."""
    encoded = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _continuation(
    messages: list[dict[str, Any]],
    prefix_len: int,
    prefix_matches: Callable[[list[dict[str, Any]]], bool],
    reply_ids: tuple[str, ...],
) -> list[dict[str, Any]] | None:
    """Messages after an already-answered prefix, or None if they diverge."""
    if len(messages) <= prefix_len + 1 or not prefix_matches(messages[:prefix_len]):
        return None
    reply = messages[prefix_len]
    if reply.get("role") != "assistant":
        return None
    ids = tuple(str(call.get("id", "")) for call in reply.get("tool_calls", []))
    if ids != reply_ids:
        return None
    delta = messages[prefix_len + 1 :]
    if any(message.get("role") in {"system", "developer"} for message in delta):
        return None
    return delta


def _delta(
    session: AGYSession, messages: list[dict[str, Any]]
) -> list[dict[str, Any]] | None:
    """New messages after the session's last reply, or None if they diverge."""
    previous = session.history
    if previous is None:
        return None
    return _continuation(
        messages, len(previous), lambda prefix: prefix == previous, session.reply_ids
    )


class SessionPool:
    """Idle AGY sessions waiting for the next turn of their conversation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._idle: list[AGYSession] = []
        self._spares: dict[tuple[Any, ...], AGYSession] = {}
        self._priming: set[tuple[Any, ...]] = set()
        self._generation = 0
        self._reaper: threading.Thread | None = None

    def checkout(
        self, key: tuple[Any, ...], messages: list[dict[str, Any]]
    ) -> tuple[AGYSession | None, list[dict[str, Any]] | None, str]:
        expired = self._collect_expired()
        try:
            with self._lock:
                same_conversation = [s for s in self._idle if s.key[0] == key[0]]
                for session in same_conversation:
                    self._idle.remove(session)
            match: AGYSession | None = None
            delta: list[dict[str, Any]] | None = None
            for session in same_conversation:
                candidate = _delta(session, messages) if session.key == key else None
                if match is None and candidate is not None:
                    match, delta = session, candidate
                else:
                    expired.append(session)
            if match is not None:
                return match, delta, "reuse"
            if not same_conversation:
                return None, None, "new conversation"
            if all(session.key != key for session in same_conversation):
                return None, None, "model, effort, tools or environment changed"
            return None, None, "history diverged"
        finally:
            for session in expired:
                session.stop()

    def checkin(self, session: AGYSession) -> None:
        evicted: list[AGYSession] = []
        if session.alive:
            with self._lock:
                for other in [s for s in self._idle if s.key[0] == session.key[0]]:
                    self._idle.remove(other)
                    evicted.append(other)
                self._idle.append(session)
                limit = int(_env_number(MAX_SESSIONS_ENV, DEFAULT_MAX_SESSIONS))
                self._idle.sort(key=lambda item: item.last_used)
                while len(self._idle) > limit:
                    evicted.append(self._idle.pop(0))
                self._ensure_reaper()
        else:
            evicted.append(session)
        for item in evicted:
            item.stop()

    def _ensure_reaper(self) -> None:
        """Start the idle reaper; the caller holds the lock."""
        if self._reaper is None:
            self._reaper = threading.Thread(
                target=self._reap_forever, name="agy-session-reaper", daemon=True
            )
            self._reaper.start()

    def take_spare(self, spare_key: tuple[Any, ...]) -> AGYSession | None:
        """A pre-started process for these launch settings, if one is ready."""
        with self._lock:
            spare = self._spares.pop(spare_key, None)
        if spare is not None and not spare.alive:
            spare.stop()
            return None
        return spare

    def prime(
        self, spare_key: tuple[Any, ...], start: Callable[[], AGYSession]
    ) -> None:
        """Start a spare for these launch settings in the background."""
        with self._lock:
            if spare_key in self._spares or spare_key in self._priming:
                return
            self._priming.add(spare_key)
            generation = self._generation

        def run() -> None:
            spare: AGYSession | None = None
            try:
                spare = start()
            except Exception:
                logger.debug("AGY warm spare could not be started", exc_info=True)
            evicted: list[AGYSession] = []
            with self._lock:
                self._priming.discard(spare_key)
                if spare is not None and generation == self._generation:
                    self._spares[spare_key] = spare
                    while len(self._spares) > MAX_SPARES:
                        oldest = min(
                            self._spares, key=lambda k: self._spares[k].last_used
                        )
                        evicted.append(self._spares.pop(oldest))
                    self._ensure_reaper()
                elif spare is not None:  # the pool was cleared meanwhile
                    evicted.append(spare)
            for item in evicted:
                item.stop()

        threading.Thread(target=run, name="agy-warm-spare", daemon=True).start()

    def _collect_expired(self) -> list[AGYSession]:
        idle_limit = _env_number(IDLE_SECONDS_ENV, DEFAULT_IDLE_SECONDS)
        now = time.monotonic()

        def stale(session: AGYSession) -> bool:
            return not session.alive or now - session.last_used > idle_limit

        with self._lock:
            expired = [session for session in self._idle if stale(session)]
            for session in expired:
                self._idle.remove(session)
            for spare_key, spare in list(self._spares.items()):
                if stale(spare):
                    expired.append(spare)
                    del self._spares[spare_key]
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
            spares, self._spares = self._spares, {}
            self._generation += 1
        sessions += list(spares.values())
        for session in sessions:
            session.stop()


def store_path() -> Path | None:
    """Where resumable conversations are recorded, or None when disabled."""
    raw = os.environ.get(SESSION_STORE_ENV)
    if raw is not None:
        raw = raw.strip()
        if raw.lower() in {"", "0", "off", "false", "no"}:
            return None
        return Path(raw).expanduser()
    home = os.environ.get("HERMES_HOME") or "~/.hermes"
    return Path(home).expanduser() / "state" / "agy-conversations.json"


def _load_store(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class ConversationStore:
    """On-disk map of session id to a resumable AGY conversation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _change(self, change: Callable[[dict[str, Any]], bool | None]) -> None:
        path = store_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with (
                self._lock,
                open(f"{path}.lock", "a+", encoding="utf-8") as lock_file,
            ):
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                data = _load_store(path)
                if change(data) is False:
                    return
                now = time.time()
                kept = sorted(
                    (
                        item
                        for item in data.items()
                        if isinstance(item[1], dict)
                        and now - float(item[1].get("updated", 0))
                        < STORE_MAX_AGE_SECONDS
                    ),
                    key=lambda item: float(item[1].get("updated", 0)),
                    reverse=True,
                )[:STORE_MAX_ENTRIES]
                temporary = path.with_name(
                    f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
                )
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(dict(kept), handle)
                os.replace(temporary, path)
        except (OSError, TypeError, ValueError):
            logger.debug("AGY conversation store update failed", exc_info=True)

    def resume_point(
        self, session_id: str, compat: str, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]] | None:
        path = store_path()
        if path is None:
            return None
        record = _load_store(path).get(session_id)
        if (
            not isinstance(record, dict)
            or record.get("in_flight")
            or record.get("compat") != compat
        ):
            return None
        conversation_id = record.get("conversation_id")
        length = record.get("history_len")
        digest = record.get("history_digest")
        reply_ids = record.get("reply_ids")
        if not (
            isinstance(conversation_id, str)
            and _CONVERSATION_ID_RE.fullmatch(conversation_id)
            and isinstance(length, int)
            and not isinstance(length, bool)
            and isinstance(digest, str)
            and isinstance(reply_ids, list)
        ):
            return None
        delta = _continuation(
            messages,
            length,
            lambda prefix: history_digest(prefix) == digest,
            tuple(str(item) for item in reply_ids),
        )
        return None if delta is None else (conversation_id, delta)

    def record(
        self,
        session_id: str,
        *,
        conversation_id: str,
        compat: str,
        history: list[dict[str, Any]],
        reply_ids: tuple[str, ...],
    ) -> None:
        entry = {
            "conversation_id": conversation_id,
            "compat": compat,
            "history_len": len(history),
            "history_digest": history_digest(history),
            "reply_ids": list(reply_ids),
            "in_flight": False,
            "updated": time.time(),
        }

        def change(data: dict[str, Any]) -> None:
            data[session_id] = entry

        self._change(change)

    def mark_in_flight(self, session_id: str) -> None:
        def change(data: dict[str, Any]) -> bool | None:
            entry = data.get(session_id)
            if not isinstance(entry, dict):
                return False
            entry["in_flight"] = True
            entry["updated"] = time.time()
            return None

        self._change(change)

    def forget(self, session_id: str) -> None:
        def change(data: dict[str, Any]) -> bool | None:
            if session_id not in data:
                return False
            del data[session_id]
            return None

        self._change(change)


POOL = SessionPool()
atexit.register(POOL.clear)
STORE = ConversationStore()
