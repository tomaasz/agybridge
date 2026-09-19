"""OpenAI HTTP formatters and serializer for agybridge."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from agybridge.ports import (
    BridgePorts,
    default_build_openai_tool_call,
    default_render_tool_bridge_sections,
)


def format_chat_completion_chunk(
    chunk_id: str,
    model: str,
    *,
    delta_content: str | None = None,
    delta_tool_calls: list[Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> str:
    """Format a single OpenAI SSE chunk string."""
    choice: dict[str, Any] = {
        "index": 0,
        "delta": {},
        "finish_reason": finish_reason,
    }
    if delta_content is not None:
        choice["delta"]["content"] = delta_content
    if delta_tool_calls:
        formatted_calls = []
        for idx, call in enumerate(delta_tool_calls):
            call_dict = {
                "index": idx,
                "id": getattr(call, "id", str(uuid.uuid4())),
                "type": "function",
                "function": {
                    "name": getattr(call.function, "name", ""),
                    "arguments": getattr(call.function, "arguments", "{}"),
                },
            }
            formatted_calls.append(call_dict)
        choice["delta"]["tool_calls"] = formatted_calls

    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def serialize_chat_completion(completion: Any) -> dict[str, Any]:
    """Convert an internal completion object to an OpenAI API response dict."""
    choice = completion.choices[0]
    message_obj = choice.message
    tool_calls_data = None
    if message_obj.tool_calls:
        tool_calls_data = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message_obj.tool_calls
        ]

    message_dict: dict[str, Any] = {
        "role": "assistant",
        "content": message_obj.content,
    }
    if tool_calls_data:
        message_dict["tool_calls"] = tool_calls_data

    usage_dict = {
        "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
        "total_tokens": getattr(completion.usage, "total_tokens", 0),
    }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": completion.model,
        "choices": [
            {
                "index": 0,
                "message": message_dict,
                "finish_reason": choice.finish_reason,
            }
        ],
        "usage": usage_dict,
    }


class HttpBridgePorts(BridgePorts):
    """Bridge ports tailored for HTTP responses."""


HTTP_PORTS = HttpBridgePorts(
    tool_call_factory=default_build_openai_tool_call,
    tool_schema_renderer=default_render_tool_bridge_sections,
)
