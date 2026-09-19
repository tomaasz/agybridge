from __future__ import annotations

import pytest
from agybridge.ports import default_build_openai_tool_call
from agybridge.protocol import AGYProtocolError
from agybridge.toolcalls import _strict_tool_calls, _tool_policy


def test_tool_policy_validation():
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        {"type": "function", "function": {"name": "write_file", "parameters": {}}},
    ]
    policy = _tool_policy(tools, "auto")
    assert policy.allowed_names == {"read_file", "write_file"}
    assert policy.required is False

    forced = _tool_policy(
        tools, {"type": "function", "function": {"name": "read_file"}}
    )
    assert forced.allowed_names == {"read_file"}
    assert forced.required is True

    with pytest.raises(ValueError, match="duplicate tool schema"):
        _tool_policy([*tools, tools[0]], "auto")


def test_strict_tool_calls_valid():
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    policy = _tool_policy(tools, "auto")
    text = (
        "Thinking before call\n"
        '<tool_call>{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"a.txt\\"}"}}</tool_call>\n'
        "Some trailing text"
    )
    calls, clean = _strict_tool_calls(
        text,
        policy,
        max_argument_bytes=1024,
        max_tool_calls=16,
        tool_call_factory=default_build_openai_tool_call,
    )
    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].function.name == "read_file"
    assert "Thinking before call" in clean
    assert "Some trailing text" in clean


def test_strict_tool_calls_malformed():
    tools = [{"type": "function", "function": {"name": "read_file"}}]
    policy = _tool_policy(tools, "auto")

    # Unclosed tag
    with pytest.raises(AGYProtocolError, match="unclosed"):
        _strict_tool_calls(
            "<tool_call>abc", policy, max_argument_bytes=1024, max_tool_calls=16
        )

    # Nested tag
    with pytest.raises(AGYProtocolError, match="nested"):
        _strict_tool_calls(
            "<tool_call><tool_call></tool_call></tool_call>",
            policy,
            max_argument_bytes=1024,
            max_tool_calls=16,
        )

    # Unknown tool
    with pytest.raises(AGYProtocolError, match="unavailable Hermes tool"):
        _strict_tool_calls(
            '<tool_call>{"id":"call_1","type":"function","function":{"name":"unknown","arguments":"{}"}}</tool_call>',
            policy,
            max_argument_bytes=1024,
            max_tool_calls=16,
        )
