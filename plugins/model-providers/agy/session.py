"""Backward-compatibility shim for agy session module."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_DIR = _REPO_ROOT / "core"
for _path in (str(_REPO_ROOT), str(_CORE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agybridge.session import (
    DEFAULT_IDLE_SECONDS,
    DEFAULT_MAX_SESSIONS,
    IDLE_SECONDS_ENV,
    MAX_SESSIONS_ENV,
    PERSISTENT_ENV,
    POOL,
    PRINT_TIMEOUT,
    SESSION_STORE_ENV,
    STORE,
    STORE_MAX_AGE_SECONDS,
    STORE_MAX_ENTRIES,
    AGYSession,
    ConversationStore,
    SessionDied,
    SessionOverflow,
    SessionPool,
    SessionTimeout,
    history_digest,
    persistent_enabled,
    store_path,
)

__all__ = [
    "DEFAULT_IDLE_SECONDS",
    "DEFAULT_MAX_SESSIONS",
    "IDLE_SECONDS_ENV",
    "MAX_SESSIONS_ENV",
    "PERSISTENT_ENV",
    "POOL",
    "PRINT_TIMEOUT",
    "SESSION_STORE_ENV",
    "STORE",
    "STORE_MAX_AGE_SECONDS",
    "STORE_MAX_ENTRIES",
    "AGYSession",
    "ConversationStore",
    "SessionDied",
    "SessionOverflow",
    "SessionPool",
    "SessionTimeout",
    "history_digest",
    "persistent_enabled",
    "store_path",
]
