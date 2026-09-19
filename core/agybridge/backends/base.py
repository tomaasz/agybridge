"""Protocol defining the CLI backend contract."""

from __future__ import annotations

from typing import Protocol

from ..protocol import ParsedOutput


class CliBackend(Protocol):
    """Contract for a CLI execution backend (e.g. AGY)."""

    def build_argv(
        self,
        command: str,
        model: str,
        print_timeout: str,
        effort: str,
        conversation_id: str | None = None,
    ) -> list[str]:
        """Construct safe command-line arguments."""
        ...

    def map_model_and_effort(
        self, model: str, requested: str | None, default_effort: str
    ) -> tuple[str, str]:
        """Resolve model id and effort parameter."""
        ...

    def format_stdin(self, prompt: str) -> bytes:
        """Serialize user prompt for the process stdin."""
        ...

    def parse_output(self, stdout: bytes) -> ParsedOutput:
        """Parse raw process output into ParsedOutput."""
        ...
