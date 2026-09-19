"""Backward-compatibility shim for agy client module."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_DIR = _REPO_ROOT / "core"
for _path in (str(_REPO_ROOT), str(_CORE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agybridge.backends.agy import (
    _EFFORT_ALIASES,
    AGY_EFFORTS,
    _agy_model_and_effort,
    agy_effort,
)
from agybridge.engine import SESSION_ID_FIELD
from agybridge.prompt import (
    _DELTA_HEADER,
    _DENIED_RETRY,
    TEXT_ONLY_CONTRACT,
    TOOL_BRIDGE_CONTRACT,
    _message_content,
    _normalize_messages,
    render_prompt,
)
from agybridge.protocol import (
    AGYError,
    AGYProcessError,
    AGYProtocolError,
    AGYTimeoutError,
    _extract_usage,
    _parse_stream_json,
)
from agybridge.security import (
    _SAFE_ENV_NAMES,
    _SECRET_PATTERNS,
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
from agybridge.toolcalls import (
    _ID_RE,
    _NAME_RE,
    _TOOL_CLOSE,
    _TOOL_OPEN,
    _strict_tool_calls,
    _tool_policy,
    _ToolPolicy,
)

from adapters.hermes.bridge import HermesAGYClient

from .session import (
    POOL,
    PRINT_TIMEOUT,
    STORE,
    AGYSession,
    SessionDied,
    SessionOverflow,
    SessionTimeout,
    persistent_enabled,
)

AGYClient = HermesAGYClient

__all__ = [
    "AGY_EFFORTS",
    "DEFAULT_MAX_ARGUMENT_BYTES",
    "DEFAULT_MAX_PROMPT_BYTES",
    "DEFAULT_MAX_STDERR_BYTES",
    "DEFAULT_MAX_STDOUT_BYTES",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "POOL",
    "PRINT_TIMEOUT",
    "SESSION_ID_FIELD",
    "STORE",
    "TEXT_ONLY_CONTRACT",
    "TOOL_BRIDGE_CONTRACT",
    "_DELTA_HEADER",
    "_DENIED_RETRY",
    "_EFFORT_ALIASES",
    "_ID_RE",
    "_NAME_RE",
    "_SAFE_ENV_NAMES",
    "_SECRET_PATTERNS",
    "_TOOL_CLOSE",
    "_TOOL_OPEN",
    "AGYClient",
    "AGYError",
    "AGYProcessError",
    "AGYProtocolError",
    "AGYSession",
    "AGYTimeoutError",
    "SessionDied",
    "SessionOverflow",
    "SessionTimeout",
    "_ToolPolicy",
    "_agy_model_and_effort",
    "_extract_usage",
    "_message_content",
    "_normalize_messages",
    "_parse_stream_json",
    "_positive_int",
    "_positive_number",
    "_redact",
    "_safe_child_env",
    "_strict_tool_calls",
    "_tool_policy",
    "agy_effort",
    "persistent_enabled",
    "render_prompt",
]
