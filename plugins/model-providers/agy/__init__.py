"""AGY external-process provider profiles for Hermes Agent."""

from __future__ import annotations

from typing import Any

try:
    from providers import register_provider
    from providers.base import ProviderProfile
except ImportError as exc:  # pragma: no cover - exercised by an import subprocess
    raise ImportError(
        "hermes-agy-plugin requires Hermes Agent 0.21.2 or newer with the "
        "external-process provider API"
    ) from exc

from .client import AGYClient, AGYError, AGYProcessError, AGYProtocolError, AGYTimeoutError

MODEL = "gemini-3.8-flash-high"
FAST_MODEL = "gemini-3.8-flash-low"


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

    def create_client(self, **kwargs: Any) -> AGYClient:
        """Use AGY's stream-json process instead of an HTTP client."""
        return AGYClient(effort=self.effort, default_model=self.default_model, **kwargs)

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
    "process_args_env_var": "HERMES_AGY_ARGS",
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

__all__ = [
    "AGYClient",
    "AGYError",
    "AGYProcessError",
    "AGYProtocolError",
    "AGYTimeoutError",
    "agy",
    "agy_fast",
]
