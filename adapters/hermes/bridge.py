"""Hermes Agent ACP bridge adapter for agybridge."""

from __future__ import annotations

import sys
from typing import Any

from agybridge.engine import AGYClient
from agybridge.ports import BridgePorts

try:
    from agent.acp_openai_bridge import (
        build_openai_tool_call,
        completion_to_stream_chunks,
        render_tool_bridge_sections,
    )
except ImportError as exc:  # pragma: no cover - exercised by an import subprocess
    raise ImportError(
        "hermes-agy-plugin requires Hermes Agent 0.21.2 or newer with "
        "agent.acp_openai_bridge"
    ) from exc


def get_hermes_bridge_ports() -> BridgePorts:
    """Retrieve active Hermes bridge functions, respecting monkeypatched sys.modules."""
    bridge = sys.modules.get("agent.acp_openai_bridge")
    return BridgePorts(
        tool_call_factory=getattr(
            bridge, "build_openai_tool_call", build_openai_tool_call
        ),
        tool_schema_renderer=getattr(
            bridge, "render_tool_bridge_sections", render_tool_bridge_sections
        ),
        stream_codec=getattr(
            bridge, "completion_to_stream_chunks", completion_to_stream_chunks
        ),
    )


class HermesAGYClient(AGYClient):
    """AGYClient preconfigured with Hermes Agent bridge ports."""

    def __init__(
        self, *args: Any, ports: BridgePorts | None = None, **kwargs: Any
    ) -> None:
        if ports is None:
            ports = get_hermes_bridge_ports()
        super().__init__(*args, ports=ports, **kwargs)
