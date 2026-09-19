"""Strict validation and extraction of <tool_call> blocks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .ports import ToolCallFactory, default_build_openai_tool_call
from .protocol import AGYProtocolError

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class _ToolPolicy:
    tools: list[dict[str, Any]]
    allowed_names: frozenset[str]
    required: bool


def _tool_policy(tools: list[dict[str, Any]] | None, tool_choice: Any) -> _ToolPolicy:
    by_name: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ValueError("AGY tools must use OpenAI function schemas")
        function = tool.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name.strip()):
            raise ValueError("AGY received a tool with an invalid function name")
        name = name.strip()
        if name in by_name:
            raise ValueError(f"AGY received duplicate tool schema {name!r}")
        by_name[name] = tool

    required = False
    selected = by_name
    if tool_choice in (None, "auto"):
        pass
    elif tool_choice == "none":
        selected = {}
    elif tool_choice == "required":
        required = True
        if not selected:
            raise ValueError("tool_choice='required' needs at least one tool")
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if tool_choice.get("type") != "function" or not isinstance(name, str):
            raise ValueError("AGY received an invalid forced tool choice")
        name = name.strip()
        if name not in by_name:
            raise ValueError(f"forced tool {name!r} is not present in tools")
        selected = {name: by_name[name]}
        required = True
    else:
        raise ValueError(f"unsupported AGY tool_choice: {tool_choice!r}")
    return _ToolPolicy(list(selected.values()), frozenset(selected), required)


def _strict_tool_calls(
    text: str,
    policy: _ToolPolicy,
    *,
    max_argument_bytes: int,
    max_tool_calls: int,
    tool_call_factory: ToolCallFactory = default_build_openai_tool_call,
) -> tuple[list[Any], str]:
    if "<tool_call" in text and _TOOL_OPEN not in text:
        raise AGYProtocolError("AGY emitted a malformed <tool_call> opening tag")
    if _TOOL_CLOSE in text and _TOOL_OPEN not in text:
        raise AGYProtocolError("AGY emitted an unmatched </tool_call> tag")
    raw_calls: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(_TOOL_OPEN, cursor)
        if start < 0:
            break
        end = text.find(_TOOL_CLOSE, start + len(_TOOL_OPEN))
        if end < 0:
            raise AGYProtocolError("AGY emitted an unclosed <tool_call> block")
        if text.find(_TOOL_OPEN, start + len(_TOOL_OPEN), end) >= 0:
            raise AGYProtocolError("AGY emitted nested <tool_call> blocks")
        raw_calls.append(text[start + len(_TOOL_OPEN) : end].strip())
        spans.append((start, end + len(_TOOL_CLOSE)))
        cursor = end + len(_TOOL_CLOSE)
    if text.find(_TOOL_CLOSE, cursor) >= 0:
        raise AGYProtocolError("AGY emitted an unmatched </tool_call> tag")
    if len(raw_calls) > max_tool_calls:
        raise AGYProtocolError(f"AGY emitted more than {max_tool_calls} tool calls")

    calls: list[Any] = []
    seen_ids: set[str] = set()
    seen_requests: set[tuple[str, str]] = set()
    for raw in raw_calls:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AGYProtocolError("AGY emitted malformed tool-call JSON") from exc
        if not isinstance(obj, dict) or set(obj) != {"id", "type", "function"}:
            raise AGYProtocolError(
                "AGY tool call must contain only id, type, and function"
            )
        call_id = obj.get("id")
        if not isinstance(call_id, str) or not _ID_RE.fullmatch(call_id.strip()):
            raise AGYProtocolError("AGY tool call has an invalid or missing id")
        call_id = call_id.strip()
        if call_id in seen_ids:
            raise AGYProtocolError(f"AGY emitted duplicate tool-call id {call_id!r}")
        if obj.get("type") != "function":
            raise AGYProtocolError("AGY tool call type must be 'function'")
        function = obj.get("function")
        if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
            raise AGYProtocolError(
                "AGY tool call function must contain only name and arguments"
            )
        name = function.get("name")
        if not isinstance(name, str) or not _NAME_RE.fullmatch(name.strip()):
            raise AGYProtocolError("AGY tool call has an invalid function name")
        name = name.strip()
        if name not in policy.allowed_names:
            raise AGYProtocolError(f"AGY requested unavailable Hermes tool {name!r}")
        arguments = function.get("arguments")
        if isinstance(arguments, Mapping):
            parsed_arguments: Any = dict(arguments)
        elif isinstance(arguments, str):
            if len(arguments.encode("utf-8")) > max_argument_bytes:
                raise AGYProtocolError(
                    "AGY tool-call arguments exceed the configured size limit"
                )
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise AGYProtocolError(
                    "AGY tool-call arguments are not valid JSON"
                ) from exc
        else:
            raise AGYProtocolError(
                "AGY tool-call arguments must be a JSON object string"
            )
        if not isinstance(parsed_arguments, dict):
            raise AGYProtocolError(
                "AGY tool-call arguments must decode to a JSON object"
            )
        normalized_arguments = json.dumps(
            parsed_arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        if len(normalized_arguments.encode("utf-8")) > max_argument_bytes:
            raise AGYProtocolError(
                "AGY tool-call arguments exceed the configured size limit"
            )
        signature = (name, normalized_arguments)
        if signature in seen_requests:
            raise AGYProtocolError("AGY emitted a duplicate tool request")
        seen_ids.add(call_id)
        seen_requests.add(signature)
        calls.append(
            tool_call_factory(
                call_id=call_id, name=name, arguments=normalized_arguments
            )
        )

    if policy.required and not calls:
        raise AGYProtocolError("AGY did not emit the required Hermes tool call")
    clean_parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if cursor < start:
            clean_parts.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        clean_parts.append(text[cursor:])
    clean_text = "\n".join(part.strip() for part in clean_parts if part.strip()).strip()
    return calls, clean_text
