"""Hermes Agent compatibility shim for agybridge."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CORE_DIR = _REPO_ROOT / "core"
for _path in (str(_REPO_ROOT), str(_CORE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from adapters.hermes import (
    CONTEXT_LENGTH,
    FAST_MODEL,
    MODEL,
    AGYClient,
    AGYError,
    AGYProcessError,
    AGYProfile,
    AGYProtocolError,
    AGYTimeoutError,
    agy,
    agy_fast,
)

from . import client, session

__all__ = [
    "CONTEXT_LENGTH",
    "FAST_MODEL",
    "MODEL",
    "AGYClient",
    "AGYError",
    "AGYProcessError",
    "AGYProfile",
    "AGYProtocolError",
    "AGYTimeoutError",
    "agy",
    "agy_fast",
    "client",
    "session",
]
