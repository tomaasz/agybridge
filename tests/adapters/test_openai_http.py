from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from adapters.openai_http.server import create_server


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DummyChatCompletions:
    def __init__(self):
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        msg = SimpleNamespace(content="Hello from AGY", tool_calls=None)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
            model=kwargs.get("model", "gemini-3.8-flash-high"),
        )


class DummyClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=DummyChatCompletions())


@pytest.fixture
def http_server():
    port = _find_free_port()
    client = DummyClient()
    server = create_server("127.0.0.1", port, token="test-secret-token", client=client)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", client
    server.shutdown()
    server.server_close()


def test_health_check(http_server):
    base_url, _ = http_server
    req = urllib.request.Request(f"{base_url}/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "ok"


def test_auth_unauthorized_and_authorized(http_server):
    base_url, _ = http_server

    # Missing auth
    req_no_auth = urllib.request.Request(f"{base_url}/v1/models")
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req_no_auth)
    assert exc_info.value.code == 401

    # Wrong token
    req_wrong_auth = urllib.request.Request(
        f"{base_url}/v1/models",
        headers={"Authorization": "Bearer wrong-token"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req_wrong_auth)
    assert exc_info.value.code == 401

    # Valid token
    req_valid = urllib.request.Request(
        f"{base_url}/v1/models",
        headers={"Authorization": "Bearer test-secret-token"},
    )
    with urllib.request.urlopen(req_valid) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "list"
        model_ids = [m["id"] for m in data["data"]]
        assert "gemini-3.8-flash-high" in model_ids


def test_chat_completions_non_streaming(http_server):
    base_url, dummy_client = http_server
    payload = {
        "model": "gemini-3.8-flash-high",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer test-secret-token",
            "Content-Type": "application/json",
            "X-AGY-Session": "session-12345",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello from AGY"
        assert data["usage"]["total_tokens"] == 15

    # Verify session header was forwarded to client
    assert dummy_client.chat.completions.last_call["extra_body"] == {
        "hermes_session_id": "session-12345"
    }


def test_chat_completions_streaming_sse(http_server):
    base_url, _ = http_server
    payload = {
        "model": "gemini-3.8-flash-high",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer test-secret-token",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers["Content-Type"]
        lines = []
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if line:
                lines.append(line)
            if line == "data: [DONE]":
                break
        assert "data: [DONE]" in lines
        data_lines = [
            l for l in lines if l.startswith("data: ") and l != "data: [DONE]"
        ]
        assert len(data_lines) >= 2


def test_payload_too_large(http_server):
    base_url, _ = http_server
    # Content-Length exceeds 8 MiB
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=b"a",
        headers={
            "Authorization": "Bearer test-secret-token",
            "Content-Length": str(10 * 1024 * 1024),
        },
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == 413


def test_hermes_and_http_parity(http_server):
    base_url, dummy_client = http_server
    prompt = "Hello parity check"

    # Call directly as Hermes would
    hermes_completion = dummy_client.chat.completions.create(
        model="gemini-3.8-flash-high",
        messages=[{"role": "user", "content": prompt}],
    )

    # Call through HTTP endpoint
    payload = {
        "model": "gemini-3.8-flash-high",
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer test-secret-token",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        http_data = json.loads(resp.read().decode("utf-8"))

    assert (
        http_data["choices"][0]["message"]["content"]
        == hermes_completion.choices[0].message.content
    )
    assert (
        http_data["choices"][0]["finish_reason"]
        == hermes_completion.choices[0].finish_reason
    )
    assert http_data["usage"]["total_tokens"] == hermes_completion.usage.total_tokens
    assert http_data["usage"]["prompt_tokens"] == hermes_completion.usage.prompt_tokens
    assert (
        http_data["usage"]["completion_tokens"]
        == hermes_completion.usage.completion_tokens
    )


def test_quota_error_maps_to_429(http_server):
    from agybridge.protocol import AGYQuotaError

    base_url, client = http_server

    def exhausted(**kwargs):
        raise AGYQuotaError(
            "AGY quota exhausted: Individual quota reached. Resets in 94h."
        )

    client.chat.completions.create = exhausted
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={
            "Authorization": "Bearer test-secret-token",
            "Content-Type": "application/json",
        },
    )
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(req)
    assert info.value.code == 429
    error = json.loads(info.value.read().decode("utf-8"))["error"]
    assert error["code"] == "insufficient_quota"
    assert "Resets in 94h" in error["message"]


def test_quota_error_sends_retry_after(http_server):
    from agybridge.protocol import AGYQuotaError

    base_url, client = http_server

    def exhausted(**kwargs):
        raise AGYQuotaError("AGY quota exhausted: x", retry_after=3600.4)

    client.chat.completions.create = exhausted
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={
            "Authorization": "Bearer test-secret-token",
            "Content-Type": "application/json",
        },
    )
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(req)
    assert info.value.code == 429
    assert info.value.headers["Retry-After"] == "3600"
