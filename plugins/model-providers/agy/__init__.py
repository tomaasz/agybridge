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

# ``adapters.hermes`` registers both profiles as an import side effect, which runs once per
# process. Hermes imports this plugin once per HERMES_HOME (each profile gets its own provider
# layer), so a second profile served by the same process (gateway cron, desktop backend) would
# see a cached import and an empty layer: "Could not find the 'agy' CLI command '(none
# configured)'". Registering here again lands in whichever home layer is importing us.
from providers import register_provider as _register_provider

_register_provider(agy)
_register_provider(agy_fast)

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
