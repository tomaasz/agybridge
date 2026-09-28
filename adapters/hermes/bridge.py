"""Hermes Agent ACP bridge adapter for agybridge."""

from __future__ import annotations

import sys
import os
import hashlib
import json
from typing import Any

from agybridge.engine import AGYClient, SESSION_ID_FIELD
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
        # A session id is still required; unrelated auxiliary calls stay one-shot.
        if "HERMES_AGY_PERSISTENT" not in os.environ:
            kwargs.setdefault("persistent", True)
        super().__init__(*args, ports=ports, **kwargs)

    def _create(self, **call: Any) -> Any:
        extra = dict(call.get("extra_body") or {})
        # Tool-using auxiliary loops have an ambient conversation context. Give
        # each initial task its own namespace so it cannot evict the main agent.
        if not extra.get(SESSION_ID_FIELD) and call.get("tools"):
            try:
                from agent.portal_tags import get_conversation_context
            except ImportError:
                scope = None
            else:
                scope = get_conversation_context()
            if scope:
                prefix = []
                for message in call.get("messages") or []:
                    if message.get("role") in {"assistant", "tool"}:
                        break
                    prefix.append(message)
                digest = hashlib.sha256(json.dumps(prefix, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
                extra[SESSION_ID_FIELD] = f"aux:{scope}:{digest}"
                call["extra_body"] = extra
        return super()._create(**call)
