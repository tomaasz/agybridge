"""Core OpenAI-compatible engine facade over AGY CLI."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import subprocess
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .accounts import ACCOUNTS
from .agylog import AGYLogWatcher, new_log_path
from .backends.agy import AGY_EFFORTS, DEFAULT_BACKEND, agy_effort
from .backends.base import CliBackend
from .ports import DEFAULT_PORTS, BridgePorts
from .process import run_process, stop_process
from .prompt import (
    _DELTA_HEADER,
    _DENIED_RETRY,
    _REPAIR_RETRY,
    TEXT_ONLY_CONTRACT,
    TOOL_BRIDGE_CONTRACT,
    _normalize_messages,
    render_prompt,
)
from .protocol import (
    AGYProcessError,
    AGYProtocolError,
    AGYQuotaError,
    AGYTimeoutError,
    ParsedOutput,
)
from .quota import QUOTA
from .security import (
    DEFAULT_MAX_ARGUMENT_BYTES,
    DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_MAX_STDERR_BYTES,
    DEFAULT_MAX_STDOUT_BYTES,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    _positive_int,
    _positive_number,
    _redact,
    _safe_child_env,
)
from .session import (
    POOL,
    PRINT_TIMEOUT,
    STORE,
    AGYSession,
    SessionDied,
    SessionOverflow,
    SessionQuota,
    SessionTimeout,
    persistent_enabled,
    warm_spare_enabled,
)
from .toolcalls import _TOOL_OPEN, _strict_tool_calls, _tool_policy

SESSION_ID_FIELD = "hermes_session_id"
# Pool accounts tried after the first one within a single request.
MAX_ACCOUNT_SWITCHES = 3
DEFAULT_BASE_URL = "acp://agy"
COMMAND_ENV_VARS = ("HERMES_AGY_COMMAND", "AGY_CLI_PATH")

logger = logging.getLogger(__name__)


def _process_cwd() -> Path:
    """The process directory, or home when it was deleted under us.

    A Hermes side agent may create a client without a cwd while a tool is
    cleaning up the directory the process is in; os.getcwd() then raises.
    """
    try:
        return Path(os.getcwd()).resolve()
    except FileNotFoundError:
        home = Path.home()
        logger.warning("AGY: process directory no longer exists; using %s", home)
        return home


def _log_denied_retry(parsed: ParsedOutput) -> None:
    logger.info(
        "AGY tried its own action (%s); retrying once with the Hermes tool contract",
        ", ".join(parsed.denied_names) or "unnamed",
    )


def _quota_error(message: str) -> AGYQuotaError:
    logger.warning(
        "AGY quota exhausted (%s); stopping AGY instead of waiting out its retries",
        message,
    )
    return AGYQuotaError(f"AGY quota exhausted: {message}", quota_message=message)


def _route(session: AGYSession | None, disposable: bool) -> str:
    if disposable:
        return "one-shot, warm spare"
    return "one-shot" if session is None else f"session turn {session.turns}"


def _format_seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


class AGYClient:
    """OpenAI-shaped client running bounded AGY subprocess turns."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | None = None,
        acp_cwd: str | None = None,
        write: bool = False,
        timeout: Any = DEFAULT_TIMEOUT_SECONDS,
        effort: str = "high",
        default_model: str = DEFAULT_MODEL,
        max_stdout_bytes: int = DEFAULT_MAX_STDOUT_BYTES,
        max_stderr_bytes: int = DEFAULT_MAX_STDERR_BYTES,
        max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
        max_argument_bytes: int = DEFAULT_MAX_ARGUMENT_BYTES,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        env_allowlist: Iterable[str] = (),
        terminate_grace: float = 2.0,
        persistent: bool | None = None,
        ports: BridgePorts | None = None,
        backend: CliBackend | None = None,
        base_url: Any = None,
        api_key: Any = None,
        **_: Any,
    ) -> None:
        if command is None or (isinstance(command, str) and not command.strip()):
            # Hermes passes an empty command when it cannot resolve one itself.
            command = next(
                (os.environ[name] for name in COMMAND_ENV_VARS if os.environ.get(name)),
                "agy",
            )
        if effort not in AGY_EFFORTS:
            raise ValueError("AGY effort must be one of: low, medium, high")
        if write:
            raise ValueError(
                "AGY write mode is disabled: Hermes must authorize and execute every write"
            )
        if not isinstance(command, str) or not command.strip():
            raise ValueError("AGY command must be a non-empty executable name")
        if args:
            raise ValueError(
                "AGY process args are disabled because they can select a subcommand before the sandbox flags; use a wrapper executable instead"
            )
        explicit_cwd = acp_cwd or cwd
        workdir = (
            Path(explicit_cwd).expanduser().resolve()
            if explicit_cwd
            else _process_cwd()
        )
        if not workdir.is_dir():
            raise ValueError("AGY cwd must be an existing directory")
        if not isinstance(default_model, str) or not default_model.strip():
            raise ValueError("AGY default model must be non-empty")

        self.command = command.strip()
        # Hermes reads these when it switches to this client as a fallback.
        self.base_url = str(base_url or DEFAULT_BASE_URL)
        self.api_key = api_key if isinstance(api_key, str) else ""
        self.cwd = str(workdir)
        self.timeout = _positive_number(
            timeout, DEFAULT_TIMEOUT_SECONDS, label="AGY timeout"
        )
        self.terminate_grace = _positive_number(
            terminate_grace, 2.0, label="AGY terminate grace"
        )
        self.effort = effort
        self.default_model = default_model.strip()
        self.max_stdout_bytes = _positive_int(
            max_stdout_bytes, DEFAULT_MAX_STDOUT_BYTES, label="max_stdout_bytes"
        )
        self.max_stderr_bytes = _positive_int(
            max_stderr_bytes, DEFAULT_MAX_STDERR_BYTES, label="max_stderr_bytes"
        )
        self.max_prompt_bytes = _positive_int(
            max_prompt_bytes, DEFAULT_MAX_PROMPT_BYTES, label="max_prompt_bytes"
        )
        self.max_argument_bytes = _positive_int(
            max_argument_bytes, DEFAULT_MAX_ARGUMENT_BYTES, label="max_argument_bytes"
        )
        self.max_tool_calls = _positive_int(
            max_tool_calls, DEFAULT_MAX_TOOL_CALLS, label="max_tool_calls"
        )
        configured_allowlist = os.environ.get("HERMES_AGY_ENV_ALLOWLIST", "").split(",")
        self._child_env = _safe_child_env([*configured_allowlist, *env_allowlist])
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.is_closed = False
        self._active_processes: set[subprocess.Popen[bytes]] = set()
        self._process_lock = threading.Lock()
        self.persistent = (
            persistent_enabled() if persistent is None else bool(persistent)
        )
        self._active_sessions: set[AGYSession] = set()
        self.ports = ports or DEFAULT_PORTS
        self.backend = backend or DEFAULT_BACKEND

    def close(self) -> None:
        """Stop an in-flight child; safe to call more than once."""
        with self._process_lock:
            self.is_closed = True
            processes = tuple(self._active_processes)
            self._active_processes.clear()
            sessions = tuple(self._active_sessions)
            self._active_sessions.clear()
        for process in processes:
            stop_process(process, self.terminate_grace)
        for session in sessions:
            session.stop()

    def _oneshot_turn(
        self, model: str, effort: str, prompt: str, effective_timeout: float
    ) -> ParsedOutput:
        request_deadline = time.monotonic() + effective_timeout
        current_prompt = prompt
        for attempt in range(2):
            remaining = request_deadline - time.monotonic()
            if remaining <= 0:
                raise AGYTimeoutError(
                    f"AGY exceeded the {effective_timeout:g}s request timeout"
                )
            log_path = new_log_path()
            watcher = AGYLogWatcher(log_path) if log_path is not None else None
            argv = self.backend.build_argv(
                self.command,
                model,
                f"{max(1, math.ceil(remaining))}s",
                effort,
                log_file=str(log_path) if log_path is not None else None,
            )
            stdin_bytes = self.backend.format_stdin(current_prompt)

            def quota_abort(
                watcher: AGYLogWatcher | None = watcher,
            ) -> AGYQuotaError | None:
                message = watcher.check() if watcher is not None else None
                return _quota_error(message) if message else None

            def on_start(proc: subprocess.Popen[bytes]) -> None:
                with self._process_lock:
                    if not self.is_closed:
                        self._active_processes.add(proc)

            def on_done(proc: subprocess.Popen[bytes]) -> None:
                with self._process_lock:
                    self._active_processes.discard(proc)

            try:
                stdout, _stderr = run_process(
                    argv,
                    cwd=self.cwd,
                    env=self._child_env,
                    timeout=remaining,
                    max_stdout_bytes=self.max_stdout_bytes,
                    max_stderr_bytes=self.max_stderr_bytes,
                    terminate_grace=self.terminate_grace,
                    stdin_data=stdin_bytes,
                    is_closed=lambda: self.is_closed,
                    on_process_start=on_start,
                    on_process_done=on_done,
                    abort=quota_abort,
                )
            finally:
                if watcher is not None:
                    watcher.close()
            parsed = self.backend.parse_output(stdout)
            if (
                not parsed.denied
                or _TOOL_OPEN in parsed.text
                or (attempt > 0 and bool(parsed.text.strip()))
            ):
                return parsed
            if attempt == 0:
                _log_denied_retry(parsed)
                current_prompt = prompt + "\n\n" + _DENIED_RETRY
        raise AGYProcessError(
            "AGY attempted an internal action twice; Hermes fallback is required"
        )

    def _persistent_turn(
        self,
        model: str,
        effort: str,
        session_id: str,
        contract: str,
        tool_sections: list[str],
        normalized: list[dict[str, Any]],
        prompt: str,
        effective_timeout: float,
        account: str = "",
    ) -> tuple[ParsedOutput, AGYSession]:
        deadline = time.monotonic() + effective_timeout
        fingerprint = hashlib.sha256(
            "\0".join([contract, *tool_sections]).encode("utf-8")
        ).hexdigest()
        key = (
            session_id,
            (
                self.command,
                self.cwd,
                tuple(sorted(self._child_env.items())),
                model,
                effort,
                fingerprint,
                # A pool account is part of the launch settings: a session or
                # stored conversation of one Google account is never resumed
                # on another. Without a pool the key stays as it always was.
                *((account,) if account else ()),
            ),
        )
        compat = hashlib.sha256(repr(key[1]).encode("utf-8")).hexdigest()
        # One stored conversation per session and launch settings, so a side
        # agent on another model does not overwrite the main agent's record.
        store_id = f"{session_id}:{compat[:16]}"
        prompt_bytes = len(prompt.encode("utf-8"))
        session, delta, reason = POOL.checkout(key, normalized)
        resumed_from: str | None = None
        if session is not None and delta is not None:
            content = _DELTA_HEADER + json.dumps(
                delta, ensure_ascii=False, separators=(",", ":")
            )
            logger.info(
                "AGY session reused (turn %d): %d new message(s), %d of %d prompt bytes",
                session.turns + 1,
                len(delta),
                len(content.encode("utf-8")),
                prompt_bytes,
            )
            STORE.mark_in_flight(store_id)
        else:
            resume = STORE.resume_point(store_id, compat, normalized)
            if resume is not None:
                resumed_from, delta = resume
                content = _DELTA_HEADER + json.dumps(
                    delta, ensure_ascii=False, separators=(",", ":")
                )
                session = self._start_session(key, model, effort, resumed_from, account)
                logger.info(
                    "AGY session resumed (conversation %s): %d new message(s), %d of %d prompt bytes",
                    resumed_from,
                    len(delta),
                    len(content.encode("utf-8")),
                    prompt_bytes,
                )
                STORE.mark_in_flight(store_id)
            else:
                STORE.forget(store_id)
                session = self._start_session(key, model, effort, None, account)
                content = prompt
                logger.info(
                    "AGY session started (%s%s)",
                    reason,
                    ", warm spare" if session.from_spare else "",
                )
        session.store_key = (store_id, compat)
        try:
            return (
                self._drive_session(session, content, deadline, effective_timeout),
                session,
            )
        except AGYProcessError:
            if resumed_from is None:
                STORE.forget(store_id)
                raise
            logger.warning(
                "AGY conversation %s could not be resumed; starting a fresh one",
                resumed_from,
            )
        except BaseException:
            STORE.forget(store_id)
            raise
        STORE.forget(store_id)
        session = self._start_session(key, model, effort, None, account)
        session.store_key = (store_id, compat)
        try:
            return (
                self._drive_session(session, prompt, deadline, effective_timeout),
                session,
            )
        except BaseException:
            STORE.forget(store_id)
            raise

    def _start_session(
        self,
        key: tuple[Any, ...],
        model: str,
        effort: str,
        conversation_id: str | None,
        account: str = "",
    ) -> AGYSession:
        if conversation_id is None:
            spare = self._take_spare(model, effort, account)
            if spare is not None:
                spare.key = key
                return spare
        return self._spawn(key, model, effort, conversation_id)

    def _take_spare(
        self, model: str, effort: str, account: str = ""
    ) -> AGYSession | None:
        """A pre-started process for these launch settings; primes the next one."""
        if not warm_spare_enabled():
            return None
        # Launch settings only: the process takes its prompt later over stdin.
        launch = (
            self.command,
            self.cwd,
            tuple(sorted(self._child_env.items())),
            model,
            effort,
            *((account,) if account else ()),
        )
        spare = POOL.take_spare(launch)
        POOL.prime(launch, lambda: self._spawn(launch, model, effort, None))
        if spare is not None:
            spare.from_spare = True
        return spare

    def _spawn(
        self,
        key: tuple[Any, ...],
        model: str,
        effort: str,
        conversation_id: str | None,
    ) -> AGYSession:
        log_path = new_log_path()
        argv = self.backend.build_argv(
            self.command,
            model,
            PRINT_TIMEOUT,
            effort,
            conversation_id,
            log_file=str(log_path) if log_path is not None else None,
        )
        try:
            return AGYSession(
                key,
                argv,
                cwd=self.cwd,
                env=self._child_env,
                max_stderr_bytes=self.max_stderr_bytes,
                terminate_grace=self.terminate_grace,
                watcher=AGYLogWatcher(log_path) if log_path is not None else None,
            )
        except FileNotFoundError as exc:
            raise AGYProcessError(
                f"AGY executable {Path(self.command).name!r} was not found"
            ) from exc
        except OSError as exc:
            raise AGYProcessError(
                f"AGY executable {Path(self.command).name!r} could not be started"
            ) from exc

    def _drive_session(
        self,
        session: AGYSession,
        content: str,
        deadline: float,
        effective_timeout: float,
    ) -> ParsedOutput:
        with self._process_lock:
            closed = self.is_closed
            if not closed:
                self._active_sessions.add(session)
        if closed:
            session.stop()
            raise AGYProcessError("AGY client is closed")
        succeeded = False
        try:
            for attempt in range(2):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AGYTimeoutError(
                        f"AGY exceeded the {effective_timeout:g}s request timeout"
                    )
                try:
                    stdout = session.run_turn(content, remaining, self.max_stdout_bytes)
                except SessionTimeout as exc:
                    raise AGYTimeoutError(
                        f"AGY exceeded the {effective_timeout:g}s request timeout"
                    ) from exc
                except SessionQuota as exc:
                    raise _quota_error(str(exc)) from exc
                except SessionOverflow as exc:
                    raise AGYProtocolError(
                        f"AGY stdout exceeded the {self.max_stdout_bytes}-byte limit"
                    ) from exc
                except SessionDied as exc:
                    tail = _redact(exc.stderr_tail).strip()
                    detail = f": {tail}" if tail else ""
                    raise AGYProcessError(
                        f"AGY session exited with status {exc.returncode}{detail}"
                    ) from exc
                parsed = self.backend.parse_output(stdout)
                if (
                    not parsed.denied
                    or _TOOL_OPEN in parsed.text
                    or (attempt > 0 and bool(parsed.text.strip()))
                ):
                    succeeded = True
                    return parsed
                if attempt == 0:
                    _log_denied_retry(parsed)
                    content = _DENIED_RETRY
            raise AGYProcessError(
                "AGY attempted an internal action twice; Hermes fallback is required"
            )
        finally:
            with self._process_lock:
                self._active_sessions.discard(session)
            if not succeeded:
                session.stop()

    def _validate_reply(
        self, parsed: ParsedOutput, policy: Any
    ) -> tuple[list[Any], str]:
        if len(parsed.text.encode("utf-8")) > self.max_stdout_bytes:
            raise AGYProtocolError(
                "AGY response text exceeds the configured size limit"
            )
        tool_calls, clean_text = _strict_tool_calls(
            parsed.text,
            policy,
            max_argument_bytes=self.max_argument_bytes,
            max_tool_calls=self.max_tool_calls,
            tool_call_factory=self.ports.tool_call_factory,
        )
        if not tool_calls and not clean_text:
            raise AGYProtocolError("AGY returned an empty response")
        return tool_calls, clean_text

    def _create(self, *, model: str | None = None, **call: Any) -> Any:
        quota_model = (model or self.default_model).strip()
        for attempt in range(MAX_ACCOUNT_SWITCHES + 1):
            # The agent-lb pool decides the Google account; a new one makes
            # the idle sessions of the old one useless.
            if ACCOUNTS.ensure():
                POOL.clear()
            account = ACCOUNTS.current_id()
            try:
                # A remembered spent quota answers at once instead of launching AGY.
                probing = QUOTA.check(quota_model, account) if quota_model else False
                result = self._create_unchecked(model=model, account=account, **call)
            except AGYQuotaError as exc:
                message = exc.quota_message or str(exc)
                if not exc.remembered:
                    exc.retry_after = QUOTA.record(quota_model, message, account)
                if attempt < MAX_ACCOUNT_SWITCHES and ACCOUNTS.report_quota(
                    message, exc.retry_after
                ):
                    POOL.clear()
                    logger.info("retrying the AGY request on the next pool account")
                    continue
                raise
            if probing:
                QUOTA.clear(quota_model, account)
                logger.info("AGY quota for %s is available again", quota_model)
            return result
        raise AssertionError("unreachable")  # pragma: no cover

    def _create_unchecked(
        self,
        *,
        account: str = "",
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        timeout: Any = None,
        reasoning_effort: Any = None,
        extra_body: Any = None,
        **_: Any,
    ) -> Any:
        with self._process_lock:
            if self.is_closed:
                raise AGYProcessError("AGY client is closed")
        selected_model = (model or self.default_model).strip()
        if not selected_model:
            raise ValueError("AGY model must be non-empty")
        effective_timeout = _positive_number(
            timeout, self.timeout, label="AGY request timeout"
        )
        policy = _tool_policy(tools, tool_choice)
        tool_sections = self.ports.tool_schema_renderer(
            policy.tools, tool_choice if tool_choice not in (None, "none") else None
        )
        contract = TOOL_BRIDGE_CONTRACT if policy.tools else TEXT_ONLY_CONTRACT
        normalized = _normalize_messages(messages)
        conversation = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        prompt = render_prompt(contract, tool_sections, conversation)
        if len(prompt.encode("utf-8")) > self.max_prompt_bytes:
            raise AGYProtocolError(
                f"Hermes prompt exceeds the {self.max_prompt_bytes}-byte AGY limit"
            )
        agy_model, effort = self.backend.map_model_and_effort(
            selected_model, agy_effort(reasoning_effort), self.effort
        )
        session_id = (
            extra_body.get(SESSION_ID_FIELD)
            if isinstance(extra_body, Mapping)
            else None
        )
        started = time.monotonic()
        deadline = started + effective_timeout
        session: AGYSession | None = None
        disposable = False
        if self.persistent and isinstance(session_id, str) and session_id.strip():
            parsed, session = self._persistent_turn(
                agy_model,
                effort,
                session_id.strip(),
                contract,
                tool_sections,
                normalized,
                prompt,
                effective_timeout,
                account,
            )
        else:
            # A one-shot request may run in a warm spare, which is discarded
            # afterwards so nothing carries over between unrelated requests.
            session = self._take_spare(agy_model, effort, account)
            disposable = session is not None
            if session is None:
                parsed = self._oneshot_turn(
                    agy_model, effort, prompt, effective_timeout
                )
            else:
                parsed = self._drive_session(
                    session, prompt, deadline, effective_timeout
                )
        repaired = False
        try:
            try:
                tool_calls, clean_text = self._validate_reply(parsed, policy)
            except AGYProtocolError as exc:
                if session is None:
                    raise
                # A live session keeps the conversation, so asking AGY to fix its
                # reply costs one short turn instead of a cold full-prompt retry.
                logger.warning(
                    "AGY reply rejected (%s); asking for a corrected one", exc
                )
                parsed = self._drive_session(
                    session,
                    _REPAIR_RETRY.format(error=exc),
                    deadline,
                    effective_timeout,
                )
                tool_calls, clean_text = self._validate_reply(parsed, policy)
                repaired = True
        except BaseException:
            if session is not None:
                session.stop()
                if session.store_key is not None:
                    STORE.forget(session.store_key[0])
            raise
        if disposable:
            session.stop()
        elif session is not None:
            session.history = normalized
            session.reply_ids = tuple(call.id for call in tool_calls)
            if self.is_closed:
                session.stop()
            else:
                POOL.checkin(session)
            if session.store_key is not None and session.conversation_id:
                STORE.record(
                    session.store_key[0],
                    conversation_id=session.conversation_id,
                    compat=session.store_key[1],
                    history=normalized,
                    reply_ids=session.reply_ids,
                )
        logger.info(
            "AGY request done in %.1fs (%s%s; model %s, %d prompt tokens)",
            time.monotonic() - started,
            _route(session, disposable),
            ", repaired" if repaired else "",
            _format_seconds(parsed.model_seconds),
            parsed.usage.get("prompt_tokens", 0),
        )
        usage = SimpleNamespace(
            **parsed.usage, prompt_tokens_details=SimpleNamespace(cached_tokens=0)
        )
        message = SimpleNamespace(
            content=clean_text,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
            usage=usage,
            model=selected_model,
        )
        return self.ports.stream_codec(completion) if stream else completion
