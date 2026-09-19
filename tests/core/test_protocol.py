from __future__ import annotations

import json

import pytest
from agybridge.protocol import (
    AGYProtocolError,
    _parse_stream_json,
)


def test_parse_stream_json_valid():
    lines = [
        json.dumps({"event": "delta", "content": "chunk 1"}),
        json.dumps(
            {
                "event": "result",
                "result": {
                    "response": "Final answer",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                    "denied_actions": [],
                },
            }
        ),
    ]
    raw = "\n".join(lines).encode("utf-8")
    parsed = _parse_stream_json(raw)
    assert parsed.text == "Final answer"
    assert parsed.denied is False
    assert parsed.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_parse_stream_json_multiple_results_fail():
    lines = [
        json.dumps({"event": "result", "result": {"response": "first"}}),
        json.dumps({"event": "result", "result": {"response": "second"}}),
    ]
    raw = "\n".join(lines).encode("utf-8")
    with pytest.raises(AGYProtocolError, match="multiple result event"):
        _parse_stream_json(raw)


def test_parse_stream_json_no_result_fail():
    lines = [
        json.dumps({"event": "delta", "content": "only delta"}),
    ]
    raw = "\n".join(lines).encode("utf-8")
    with pytest.raises(AGYProtocolError, match="no result event"):
        _parse_stream_json(raw)


def test_parse_stream_json_invalid_utf8():
    with pytest.raises(AGYProtocolError, match="UTF-8"):
        _parse_stream_json(b"\xff\xfe\xfd")


def test_parse_stream_json_denied_actions():
    lines = [
        json.dumps(
            {
                "event": "result",
                "result": {
                    "response": "blocked output",
                    "denied_actions": [{"action": "terminal"}],
                },
            }
        ),
    ]
    raw = "\n".join(lines).encode("utf-8")
    parsed = _parse_stream_json(raw)
    assert parsed.denied is True
    assert parsed.text == "blocked output"
