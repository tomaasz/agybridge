from __future__ import annotations

import importlib.util
import json
import stat
import sys
import types
from pathlib import Path


PLUGIN = Path(__file__).parents[1] / "plugins" / "model-providers" / "agy" / "__init__.py"


def _install_hermes_stubs(monkeypatch):
    bridge = types.ModuleType("agent.acp_openai_bridge")

    def render_tool_bridge_sections(tools, tool_choice=None):
        specs = [tool["function"] for tool in (tools or []) if isinstance(tool, dict) and "function" in tool]
        if not specs:
            return []
        return [
            "Available tools (OpenAI function schema). "
            "When using a tool, emit ONLY <tool_call>{...}</tool_call>\n"
            + json.dumps(specs)
        ]

    def extract_tool_calls_from_text(text):
        start, end = text.find("<tool_call>"), text.find("</tool_call>")
        if start < 0 or end < 0:
            return [], text.strip()
        raw = json.loads(text[start + len("<tool_call>") : end])
        function = types.SimpleNamespace(**raw["function"])
        call = types.SimpleNamespace(id=raw["id"], function=function)
        return [call], (text[:start] + text[end + len("</tool_call>") :]).strip()

    bridge.render_tool_bridge_sections = render_tool_bridge_sections
    bridge.extract_tool_calls_from_text = extract_tool_calls_from_text
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


def _load_plugin(monkeypatch):
    _install_hermes_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location("agy_provider_test", PLUGIN)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_provider_exposes_high_and_fast_profiles(monkeypatch):
    module = _load_plugin(monkeypatch)

    assert module.agy.name == "agy"
    assert module.agy.fallback_models == (module.MODEL,)
    assert module.agy_fast.name == "agy-fast"
    assert module.agy_fast.fallback_models == (module.FAST_MODEL,)


def test_client_forwards_hermes_tool_schema_and_returns_tool_call(monkeypatch, tmp_path):
    module = _load_plugin(monkeypatch)
    executable = tmp_path / "agy"
    args_file = tmp_path / "args.txt"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {args_file}\n"
        "printf '%s\\n' '{\"event\":\"result\",\"result\":{\"response\":\"<tool_call>{\\\"id\\\":\\\"call_1\\\",\\\"function\\\":{\\\"name\\\":\\\"read_file\\\",\\\"arguments\\\":\\\"{\\\\\\\"path\\\\\\\":\\\\\\\"README.md\\\\\\\"}\\\"}}</tool_call>\"}}'\n"
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

    response = module.AGYClient(command=str(executable)).chat.completions.create(
        model=module.MODEL,
        messages=[{"role": "user", "content": "Read the documentation."}],
        tools=[{"type": "function", "function": {
            "name": "read_file", "description": "Read a text file", "parameters": {"type": "object"},
        }}],
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "read_file"
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"path":"README.md"}'
    prompt = args_file.read_text()
    assert "Available tools (OpenAI function schema)." in prompt
    assert "Do not use AGY tools" in prompt


def test_write_mode_requires_a_git_worktree(monkeypatch, tmp_path):
    module = _load_plugin(monkeypatch)

    try:
        module.AGYClient(cwd=str(tmp_path), write=True)
    except ValueError as exc:
        assert "isolated worktree" in str(exc)
    else:
        raise AssertionError("write mode must reject a non-worktree directory")
