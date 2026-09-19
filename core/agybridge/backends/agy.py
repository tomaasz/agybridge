"""AGY CLI backend implementation."""

from __future__ import annotations

import json
from typing import Any

from ..protocol import ParsedOutput, _parse_stream_json

AGY_EFFORTS = ("low", "medium", "high")

_EFFORT_ALIASES = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
    "ultra": "high",
}


def agy_effort(value: Any) -> str | None:
    """Map a reasoning effort onto AGY's low|medium|high, or None if unknown."""
    if not isinstance(value, str):
        return None
    return _EFFORT_ALIASES.get(value.strip().lower())


def _agy_model_and_effort(
    model: str, requested: str | None, default: str
) -> tuple[str, str]:
    """Pick the AGY model id and --effort for one request.

    AGY rejects a model id whose effort suffix disagrees with --effort
    (``gemini-3.8-flash-high`` runs only with ``high``), while the bare id
    accepts every effort. Without a requested effort the suffix wins; a
    different requested effort switches to the bare id.
    """
    base, _, suffix = model.rpartition("-")
    if not base or suffix not in AGY_EFFORTS:
        return model, requested or default
    if requested is None or requested == suffix:
        return model, suffix
    return base, requested


class AGYBackend:
    """Backend managing arguments, stdin format, and output parsing for AGY CLI."""

    def build_argv(
        self,
        command: str,
        model: str,
        print_timeout: str,
        effort: str,
        conversation_id: str | None = None,
    ) -> list[str]:
        argv = [
            command,
            "--model",
            model,
            "--effort",
            effort,
            "--mode",
            "plan",
            "--sandbox",
            "--disable-slash-commands",
            "--print-timeout",
            print_timeout,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
        ]
        if conversation_id:
            argv += ["--conversation", conversation_id]
        return argv

    def map_model_and_effort(
        self, model: str, requested: str | None, default_effort: str
    ) -> tuple[str, str]:
        return _agy_model_and_effort(model, requested, default_effort)

    def format_stdin(self, prompt: str) -> bytes:
        message = json.dumps(
            {"event": "user", "message": {"content": prompt}},
            ensure_ascii=False,
        )
        return (message + "\n").encode("utf-8")

    def parse_output(self, stdout: bytes) -> ParsedOutput:
        return _parse_stream_json(stdout)


DEFAULT_BACKEND = AGYBackend()
