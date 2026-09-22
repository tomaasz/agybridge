"""agybridge: Universal modular bridge to the AGY CLI reasoning engine."""

from .backends.agy import _agy_model_and_effort, agy_effort
from .engine import SESSION_ID_FIELD, AGYClient
from .ports import DEFAULT_PORTS, BridgePorts
from .prompt import _DENIED_RETRY
from .protocol import (
    AGYError,
    AGYProcessError,
    AGYProtocolError,
    AGYQuotaError,
    AGYTimeoutError,
)
from .security import (
    DEFAULT_MAX_ARGUMENT_BYTES,
    DEFAULT_MAX_PROMPT_BYTES,
    DEFAULT_MAX_STDERR_BYTES,
    DEFAULT_MAX_STDOUT_BYTES,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
)

__version__ = "0.3.0"

__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_MAX_ARGUMENT_BYTES",
    "DEFAULT_MAX_PROMPT_BYTES",
    "DEFAULT_MAX_STDERR_BYTES",
    "DEFAULT_MAX_STDOUT_BYTES",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_MODEL",
    "DEFAULT_PORTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "SESSION_ID_FIELD",
    "_DENIED_RETRY",
    "AGYClient",
    "AGYError",
    "AGYProcessError",
    "AGYProtocolError",
    "AGYQuotaError",
    "AGYTimeoutError",
    "BridgePorts",
    "_agy_model_and_effort",
    "agy_effort",
]
