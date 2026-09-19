from __future__ import annotations

from types import SimpleNamespace

import pytest

from adapters.mcp.server import HAS_MCP, create_mcp_server


class DummyChatCompletions:
    def create(self, **kwargs):
        msg = SimpleNamespace(content="Reasoned response: 42", tool_calls=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        return SimpleNamespace(choices=[choice])


class DummyClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=DummyChatCompletions())


def test_create_mcp_server():
    if not HAS_MCP:
        pytest.skip("mcp library not installed")

    client = DummyClient()
    server = create_mcp_server(client=client)
    assert server.name == "agybridge"


def test_mcp_tool_invocation():
    if not HAS_MCP:
        pytest.skip("mcp library not installed")

    client = DummyClient()
    server = create_mcp_server(client=client)
    # The tool was registered; find it in server._tools or call it
    assert "agy_reason" in server._tool_manager._tools
    tool_func = server._tool_manager._tools["agy_reason"].fn
    result = tool_func(prompt="What is 2+2?", effort="high")
    assert result == "Reasoned response: 42"
