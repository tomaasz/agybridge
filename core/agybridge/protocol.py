"""Stream-json protocol parser, token usage extractor, and AGY error hierarchy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class AGYError(RuntimeError):
    """Base error exposed by the provider."""


class AGYProcessError(AGYError):
    """The external process could not produce a successful response."""


class AGYProtocolError(AGYError):
    """AGY emitted output that violates the stream or tool-call contract."""


class AGYTimeoutError(AGYError, TimeoutError):
    """AGY exceeded the wall-clock deadline."""


@dataclass(frozen=True)
class ParsedOutput:
    """Parsed response from AGY stream-json stdout."""

    text: str
    denied: bool
    usage: dict[str, int]
    model_seconds: float | None = None
    denied_names: tuple[str, ...] = ()


def _seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _extract_usage(result: Mapping[str, Any]) -> dict[str, int]:
    raw = result.get("usage")
    if not isinstance(raw, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def count(*names: str) -> int:
        for name in names:
            value = raw.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
        return 0

    prompt, completion = (
        count("prompt_tokens", "input_tokens"),
        count("completion_tokens", "output_tokens"),
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": count("total_tokens") or prompt + completion,
    }


def _parse_stream_json(stdout: bytes) -> ParsedOutput:
    try:
        decoded = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AGYProtocolError("AGY stdout is not valid UTF-8 stream-json") from exc
    lines = [line for line in decoded.splitlines() if line.strip()]
    if not lines:
        raise AGYProtocolError("AGY returned empty stdout")
    results: list[Mapping[str, Any]] = []
    streamed_text: list[str] = []
    model_seconds: float | None = None
    for number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AGYProtocolError(
                f"AGY emitted invalid stream-json on line {number}"
            ) from exc
        if not isinstance(event, dict):
            raise AGYProtocolError(f"AGY emitted a non-object event on line {number}")
        if event.get("event") == "result":
            result = event.get("result")
            if not isinstance(result, Mapping):
                raise AGYProtocolError("AGY result event is missing its result object")
            results.append(result)
            continue
        step = event.get("step_update")
        if isinstance(step, Mapping) and step.get("step_type") == "agent_response":
            spent = _seconds(step.get("duration_seconds"))
            if spent is not None:
                model_seconds = (model_seconds or 0.0) + spent
        for key in ("content", "text", "message", "delta"):
            value = event.get(key)
            if isinstance(value, str):
                streamed_text.append(value)
                break
            if isinstance(value, Mapping):
                nested = (
                    value.get("response") or value.get("text") or value.get("content")
                )
                if isinstance(nested, str):
                    streamed_text.append(nested)
                    break
    if len(results) != 1:
        label = "no" if not results else "multiple"
        detail = (
            "partial streamed text was discarded"
            if streamed_text and not results
            else "response rejected"
        )
        raise AGYProtocolError(f"AGY emitted {label} result event; {detail}")
    result = results[0]
    response = result.get("response")
    if response is None:
        response = ""
    if not isinstance(response, str):
        raise AGYProtocolError("AGY result response must be text")
    denied = result.get("denied_actions", [])
    if denied is None:
        denied = []
    if not isinstance(denied, list):
        raise AGYProtocolError("AGY denied_actions must be a list")
    return ParsedOutput(
        response,
        bool(denied),
        _extract_usage(result),
        model_seconds,
        tuple(_denied_name(item) for item in denied),
    )


def _denied_name(item: Any) -> str:
    """Short label for one denied AGY action, for diagnostics only."""
    if isinstance(item, Mapping):
        for key in ("action", "tool", "name", "type"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:64]
    if isinstance(item, str) and item.strip():
        return item.strip()[:64]
    return "unknown"
