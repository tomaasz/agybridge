from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).parents[1] / "plugins" / "model-providers" / "agy"
INIT_FILE = PLUGIN_DIR / "__init__.py"


@pytest.fixture
def hermes_stubs(monkeypatch):
    bridge = types.ModuleType("agent.acp_openai_bridge")

    def build_openai_tool_call(*, call_id, name, arguments):
        return types.SimpleNamespace(
            id=call_id,
            type="function",
            function=types.SimpleNamespace(name=name, arguments=arguments),
        )

    def render_tool_bridge_sections(tools, tool_choice=None):
        return ["tools"]

    def completion_to_stream_chunks(completion):
        return [completion]

    bridge.build_openai_tool_call = build_openai_tool_call
    bridge.render_tool_bridge_sections = render_tool_bridge_sections
    bridge.completion_to_stream_chunks = completion_to_stream_chunks

    monkeypatch.setitem(sys.modules, "agent", types.ModuleType("agent"))
    monkeypatch.setitem(sys.modules, "agent.acp_openai_bridge", bridge)

    providers = types.ModuleType("providers")
    providers.register_provider = lambda profile: profile
    base = types.ModuleType("providers.base")

    class ProviderProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    base.ProviderProfile = ProviderProfile
    monkeypatch.setitem(sys.modules, "providers", providers)
    monkeypatch.setitem(sys.modules, "providers.base", base)


def test_shim_exports_and_submodules(hermes_stubs):
    name = "agy_backcompat_test_mod"
    spec = importlib.util.spec_from_file_location(
        name, INIT_FILE, submodule_search_locations=[str(PLUGIN_DIR)]
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)

    # Verify primary exports
    assert hasattr(mod, "AGYClient")
    assert hasattr(mod, "AGYProfile")
    assert hasattr(mod, "agy")
    assert hasattr(mod, "agy_fast")
    assert hasattr(mod, "MODEL")
    assert hasattr(mod, "FAST_MODEL")
    assert hasattr(mod, "AGYError")
    assert hasattr(mod, "AGYProcessError")
    assert hasattr(mod, "AGYProtocolError")
    assert hasattr(mod, "AGYTimeoutError")

    # Verify submodules
    assert hasattr(mod, "client")
    assert hasattr(mod, "session")

    # Verify client submodule exports
    assert hasattr(mod.client, "POOL")
    assert hasattr(mod.client, "STORE")
    assert hasattr(mod.client, "_DENIED_RETRY")
    assert hasattr(mod.client, "_agy_model_and_effort")
    assert hasattr(mod.client, "SESSION_ID_FIELD")
    assert hasattr(mod.client, "agy_effort")
    assert hasattr(mod.client, "AGYClient")

    # Verify session submodule exports
    assert hasattr(mod.session, "POOL")
    assert hasattr(mod.session, "STORE")
    assert hasattr(mod.session, "AGYSession")
    assert hasattr(mod.session, "SessionPool")
    assert hasattr(mod.session, "ConversationStore")
    assert hasattr(mod.session, "SessionDied")
    assert hasattr(mod.session, "SessionOverflow")
    assert hasattr(mod.session, "SessionTimeout")
    assert hasattr(mod.session, "persistent_enabled")
    assert hasattr(mod.session, "history_digest")
    assert hasattr(mod.session, "PRINT_TIMEOUT")

    # Verify POOL and STORE identity
    assert mod.client.POOL is mod.session.POOL
    assert mod.client.STORE is mod.session.STORE
