"""Security, environment allowlisting, redaction, and limit checks."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable
from typing import Any

DEFAULT_MODEL = "gemini-3.8-flash-high"
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_MAX_STDOUT_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_STDERR_BYTES = 64 * 1024
DEFAULT_MAX_PROMPT_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_ARGUMENT_BYTES = 64 * 1024
DEFAULT_MAX_TOOL_CALLS = 16

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[^\s,;]+"),
    re.compile(
        r"(?i)\b(api[_-]?key|token|secret|password|passwd|authorization)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{12,}\b"),
)

_SAFE_ENV_NAMES = {
    "APPDATA",
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "LOGNAME",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TERM",
    "TMP",
    "TMPDIR",
    "USER",
    "USERPROFILE",
    "WINDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
}


def _redact(text: str) -> str:
    """Mask secrets and credential tokens in diagnostic text."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda match: (
                f"{match.group(1)} [REDACTED]" if match.lastindex else "[REDACTED]"
            ),
            redacted,
        )
    return redacted


def _safe_child_env(extra_names: Iterable[str] = ()) -> dict[str, str]:
    """Provide a restricted environment to child processes."""
    requested = {
        name.strip() for name in extra_names if isinstance(name, str) and name.strip()
    }
    return {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_ENV_NAMES or name in requested
    }


def _positive_number(value: Any, default: float, *, label: str) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        candidates = [
            getattr(value, field, None)
            for field in ("read", "write", "connect", "pool", "timeout")
        ]
        numeric = [float(item) for item in candidates if isinstance(item, (int, float))]
        number = max(numeric) if numeric else default
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _positive_int(value: Any, default: int, *, label: str) -> int:
    number = default if value is None else int(value)
    if number <= 0:
        raise ValueError(f"{label} must be positive")
    return number
