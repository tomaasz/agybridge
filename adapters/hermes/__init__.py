"""AGY external-process provider profiles for Hermes Agent."""

from __future__ import annotations

from typing import Any

from agybridge.backends.agy import agy_effort
from agybridge.engine import SESSION_ID_FIELD
from agybridge.protocol import (
    AGYError,
    AGYProcessError,
    AGYProtocolError,
    AGYTimeoutError,
)

from .bridge import HermesAGYClient

try:
    from providers import register_provider
    from providers.base import ProviderProfile
except ImportError as exc:  # pragma: no cover - exercised by an import subprocess
    raise ImportError(
        "hermes-agy-plugin requires Hermes Agent 0.21.2 or newer with the "
        "external-process provider API"
    ) from exc

MODEL = "gemini-3.8-flash-high"
FAST_MODEL = "gemini-3.8-flash-low"
CONTEXT_LENGTH = 1_048_576


class AGYProfile(ProviderProfile):
    """Provider metadata plus construction of the AGY process client."""

    def __init__(
        self,
        *args: Any,
        effort: str = "high",
        default_model: str = MODEL,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.effort = effort
        self.default_model = default_model

    def get_model_context_length(self, model: str) -> int | None:
        """Provider-qualified context bound for AGY models (1M tokens)."""
        return CONTEXT_LENGTH

    def create_client(self, **kwargs: Any) -> HermesAGYClient:
        """Use AGY's stream-json process instead of an HTTP client."""
        if not kwargs.get("base_url"):
            kwargs["base_url"] = getattr(self, "base_url", None)
        return HermesAGYClient(
            effort=self.effort, default_model=self.default_model, **kwargs
        )

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Hand Hermes' reasoning effort and session id to the AGY client.

        Hermes calls this with ``session_id`` for main agent requests only;
        auxiliary requests omit it, which keeps them on one-shot processes.
        Without a reasoning config the profile's own effort applies.
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                top_level["reasoning_effort"] = "low"
            else:
                effort = agy_effort(reasoning_config.get("effort"))
                if effort:
                    top_level["reasoning_effort"] = effort
        if isinstance(session_id, str) and session_id.strip():
            extra_body[SESSION_ID_FIELD] = session_id.strip()
        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """AGY has no unauthenticated machine-readable model catalog."""
        return list(self.fallback_models)


_COMMON_PROFILE = {
    "api_mode": "chat_completions",
    "env_vars": (),
    "base_url": "acp://agy",
    "auth_type": "external_process",
    "process_command": "agy",
    "process_args": (),
    "process_command_env_vars": ("HERMES_AGY_COMMAND", "AGY_CLI_PATH"),
    "supports_health_check": False,
}

agy = AGYProfile(
    name="agy",
    aliases=("antigravity",),
    display_name="AGY",
    description="AGY CLI through a bounded read-only stream-json subprocess",
    fallback_models=(MODEL,),
    effort="high",
    default_model=MODEL,
    **_COMMON_PROFILE,
)
agy_fast = AGYProfile(
    name="agy-fast",
    aliases=("antigravity-fast",),
    display_name="AGY Fast",
    description="AGY CLI through a bounded read-only stream-json subprocess (low effort)",
    fallback_models=(FAST_MODEL,),
    effort="low",
    default_model=FAST_MODEL,
    **{**_COMMON_PROFILE, "base_url": "acp://agy-fast"},
)

register_provider(agy)
register_provider(agy_fast)

AGYClient = HermesAGYClient

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
]
