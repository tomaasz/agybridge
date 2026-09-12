"""AGY external-process model provider for Hermes Agent.

AGY owns its authentication. Hermes starts a bounded ``agy`` subprocess,
forwards its current tool schemas through the ACP text bridge, and executes
only the tool calls returned to Hermes' normal dispatcher.
"""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace
from typing import Any

from agent.acp_openai_bridge import extract_tool_calls_from_text, render_tool_bridge_sections
from providers import register_provider
from providers.base import ProviderProfile

MODEL = "gemini-3.8-flash-high"
FAST_MODEL = "gemini-3.8-flash-low"

TEXT_ONLY_CONTRACT = (
    "You are a text-only reasoning worker inside Hermes. Do not use AGY tools, terminal, filesystem, "
    "browser, network, permissions, or subagents. Hermes executes tools separately. Return a complete "
    "response based solely on the supplied conversation context. If evidence is absent, state what Hermes must check."
)
TOOL_BRIDGE_CONTRACT = (
    "The Hermes tools below are the only tools you may request. Do not use AGY tools, terminal, filesystem, "
    "browser, network, permissions, or subagents. When a Hermes tool is needed, emit a <tool_call> block exactly "
    "as specified. Hermes validates, authorizes, executes, and returns the result."
)


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    return ""


def _extract_text(payload: dict[str, Any]) -> str:
    if payload.get("event") == "result" and isinstance(payload.get("result"), dict):
        return str(payload["result"].get("response") or "")
    for key in ("content", "text", "message", "delta"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return str(value.get("response") or value.get("text") or value.get("content") or "")
    return ""


class AGYClient:
    """Small OpenAI-compatible facade over one bounded AGY invocation."""

    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(
        self,
        *,
        command: str = "agy",
        args: list[str] | None = None,
        cwd: str | None = None,
        write: bool = False,
        timeout: float = 900.0,
        effort: str = "high",
        **_: Any,
    ):
        if effort not in {"high", "low"}:
            raise ValueError("AGY effort must be 'high' or 'low'")
        self.command = command
        self.args = list(args or [])
        self.cwd = os.path.abspath(cwd or os.getcwd())
        self.timeout = float(timeout)
        self.effort = effort
        self.write = write
        if write and not os.path.isfile(os.path.join(self.cwd, ".git", "HEAD")):
            raise ValueError("AGY write mode requires an isolated worktree")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def close(self) -> None:
        return None

    def _create(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        **_: Any,
    ) -> Any:
        conversation = "\n\n".join(
            f"{message.get('role', 'context').upper()}: {_message_text(message.get('content'))}"
            for message in (messages or [])
            if isinstance(message, dict)
        )
        tool_sections = render_tool_bridge_sections(tools, tool_choice)
        contract = TOOL_BRIDGE_CONTRACT if tool_sections else TEXT_ONLY_CONTRACT
        prompt = "\n\n".join([contract, *tool_sections, f"CONVERSATION:\n{conversation}"])
        argv = [
            self.command,
            *self.args,
            "--model",
            model or MODEL,
            "--effort",
            self.effort,
            "--mode",
            "accept-edits" if self.write else "plan",
        ]
        if not self.write:
            argv.append("--sandbox")
        argv += ["--output-format", "stream-json", "--print", prompt]
        completed = subprocess.run(
            argv,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
        )
        if completed.returncode:
            raise RuntimeError(f"AGY exited with status {completed.returncode}: {completed.stderr[-1000:]}")

        text_parts: list[str] = []
        denied_actions: list[str] = []
        for line in completed.stdout.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if text := _extract_text(payload):
                text_parts.append(text)
            result = payload.get("result")
            if isinstance(result, dict):
                denied_actions.extend(
                    str(action["action"])
                    for action in result.get("denied_actions", []) or []
                    if isinstance(action, dict) and action.get("action")
                )

        text = "".join(text_parts)
        if not text and denied_actions:
            raise RuntimeError(
                "AGY attempted denied action(s): " + ", ".join(sorted(set(denied_actions))) + "; triggering fallback"
            )
        tool_calls, clean_text = extract_tool_calls_from_text(text)
        message = SimpleNamespace(content=clean_text, tool_calls=tool_calls, reasoning=None, reasoning_content=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls" if tool_calls else "stop")],
            usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model=model or MODEL,
        )


class AGYProfile(ProviderProfile):
    def __init__(self, *args: Any, effort: str = "high", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.effort = effort

    def create_client(self, **kwargs: Any) -> AGYClient:
        return AGYClient(effort=self.effort, **kwargs)

    def fetch_models(self, **_: Any) -> list[str]:
        return list(self.fallback_models)


agy = AGYProfile(
    name="agy",
    aliases=("antigravity",),
    display_name="AGY",
    description="AGY CLI through a bounded stream-json subprocess (high effort)",
    base_url="acp://agy",
    auth_type="external_process",
    process_command="agy",
    process_args=(),
    fallback_models=(MODEL,),
    supports_health_check=False,
    effort="high",
)
agy_fast = AGYProfile(
    name="agy-fast",
    aliases=("antigravity-fast",),
    display_name="AGY Fast",
    description="AGY CLI through a bounded stream-json subprocess (low effort)",
    base_url="acp://agy",
    auth_type="external_process",
    process_command="agy",
    process_args=(),
    fallback_models=(FAST_MODEL,),
    supports_health_check=False,
    effort="low",
)
register_provider(agy)
register_provider(agy_fast)
