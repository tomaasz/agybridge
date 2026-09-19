"""Ports and protocols for agybridge core dependency inversion."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol


class ToolCallFactory(Protocol):
    """Factory creating a host-compatible tool call object."""

    def __call__(self, *, call_id: str, name: str, arguments: str) -> Any: ...


class ToolSchemaRenderer(Protocol):
    """Renderer creating tool specification prompt sections."""

    def __call__(
        self, tools: list[dict[str, Any]] | None, tool_choice: Any = None
    ) -> list[str]: ...


class StreamCodec(Protocol):
    """Codec converting a final completion object into stream chunks."""

    def __call__(self, completion: Any) -> Iterable[Any]: ...


def default_build_openai_tool_call(
    *, call_id: str, name: str, arguments: str
) -> SimpleNamespace:
    """Stdlib default OpenAI-shaped tool call factory."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def default_render_tool_bridge_sections(
    tools: list[dict[str, Any]] | None, tool_choice: Any = None
) -> list[str]:
    """Stdlib default OpenAI tool schema prompt renderer."""
    specs = [
        {
            "name": t["function"]["name"],
            "description": t["function"].get("description", ""),
            "parameters": t["function"].get("parameters", {}),
        }
        for t in tools or []
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    ]
    sections: list[str] = []
    if specs:
        sections.append(
            "Available tools (OpenAI function schema). When using a tool, emit ONLY <tool_call>{...}</tool_call>\n"
            + json.dumps(specs, ensure_ascii=False)
        )
    if tool_choice is not None:
        sections.append(
            "Tool choice hint: " + json.dumps(tool_choice, ensure_ascii=False)
        )
    return sections


class StreamChunks(list):
    """Default iterable container for stream chunks."""


def default_completion_to_stream_chunks(completion: Any) -> Iterable[Any]:
    """Stdlib default stream chunk generator matching OpenAI format."""
    choice = completion.choices[0]
    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(
                        role="assistant",
                        content=choice.message.content or None,
                        tool_calls=choice.message.tool_calls or None,
                        reasoning=None,
                        reasoning_content=None,
                    ),
                    finish_reason=choice.finish_reason,
                )
            ],
            model=completion.model,
            usage=None,
        ),
        SimpleNamespace(
            choices=[],
            model=completion.model,
            usage=completion.usage,
        ),
    ]
    return StreamChunks(chunks)


@dataclass(frozen=True)
class BridgePorts:
    """Bundle of ports required by AGYClient to interact with host representations."""

    tool_call_factory: ToolCallFactory = default_build_openai_tool_call
    tool_schema_renderer: ToolSchemaRenderer = default_render_tool_bridge_sections
    stream_codec: StreamCodec = default_completion_to_stream_chunks


DEFAULT_PORTS = BridgePorts()
