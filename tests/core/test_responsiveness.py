from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from agybridge.accounts import AccountPool
from agybridge.engine import AGYClient
from agybridge.protocol import AGYBusyError, AGYProcessError, AGYProtocolError
from agybridge.session import POOL


@pytest.fixture(autouse=True)
def isolate_sessions(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_AGY_WARM_SPARE", "0")
    monkeypatch.setenv("HERMES_AGY_SESSION_STORE", str(tmp_path / "sessions.json"))
    yield
    POOL.clear()


def stub(tmp_path, body):
    path = tmp_path / "agy-stub"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\nfrom pathlib import Path\n"
        "def event(value): print(json.dumps(value), flush=True)\n"
        "event({'event':'init','init':{'conversation_id':'c'+str(os.getpid())}})\n"
        "for line in sys.stdin:\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\n"
    )
    path.chmod(0o700)
    return str(path)


@pytest.mark.parametrize("persistent", [False, True])
def test_text_arrives_before_final_result_and_usage_is_preserved(tmp_path, persistent):
    release = tmp_path / "release"
    command = stub(
        tmp_path,
        f"""
event({{'event':'delta','content':'zażółć '}})
until = time.monotonic()+4
while not Path({str(release)!r}).exists() and time.monotonic()<until: time.sleep(.01)
event({{'event':'delta','content':'gęślą'}})
event({{'event':'result','result':{{'response':'zażółć gęślą','usage':{{'input_tokens':2,'output_tokens':3}}}}}})
""",
    )
    client = AGYClient(
        command=command, cwd=str(tmp_path), persistent=persistent, timeout=6
    )
    stream = client.chat.completions.create(
        messages=[], stream=True, extra_body={"hermes_session_id": "a"}
    )
    try:
        first = next(stream)
        assert first.choices[0].delta.content == "zażółć "
        assert not release.exists()
        release.touch()
        rest = list(stream)
        text = first.choices[0].delta.content + "".join(
            c.choices[0].delta.content or "" for c in rest if c.choices
        )
        assert text == "zażółć gęślą"
        assert rest[-1].usage.total_tokens == 5
    finally:
        release.touch()
        stream.close()
        client.close()


@pytest.mark.parametrize("persistent", [False, True])
def test_closing_stream_stops_only_its_process(tmp_path, persistent):
    command = stub(
        tmp_path,
        """
event({'event':'delta','content':'hello'})
time.sleep(30)
""",
    )
    client = AGYClient(
        command=command,
        cwd=str(tmp_path),
        persistent=persistent,
        timeout=10,
        terminate_grace=0.1,
    )
    stream = client.chat.completions.create(
        messages=[], stream=True, extra_body={"hermes_session_id": "a"}
    )
    assert next(stream).choices[0].delta.content == "hello"
    stream.close()
    stream._worker.join(2)
    try:
        assert not stream._worker.is_alive()
        assert not client._active_processes
        assert not client._active_sessions
        assert not client.is_closed
    finally:
        client.close()


def test_stream_error_does_not_retry_or_report_success(tmp_path):
    command = stub(
        tmp_path,
        """
event({'event':'delta','content':'hello'})
event({'event':'result','result':{'response':'different'}})
""",
    )
    client = AGYClient(command=command, cwd=str(tmp_path), timeout=2)
    stream = client.chat.completions.create(messages=[], stream=True)
    assert next(stream).choices[0].delta.content == "hello"
    with pytest.raises(AGYProtocolError, match="differs"):
        list(stream)


def test_tool_enabled_stream_buffers_unvalidated_deltas(tmp_path):
    command = stub(
        tmp_path,
        """
event({'event':'delta','content':'SECRET UNVALIDATED TOOL'})
event({'event':'result','result':{'response':'safe'}})
""",
    )
    client = AGYClient(command=command, cwd=str(tmp_path), timeout=2)
    chunks = list(
        client.chat.completions.create(
            messages=[],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read_file", "parameters": {}},
                }
            ],
        )
    )
    assert (
        "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "safe"
    )


def test_capacity_applies_across_clients_and_same_session_is_rejected(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_AGY_MAX_ACTIVE", "2")
    release, reached = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = peak = 0

    def execute(self, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                reached.set()
        try:
            assert release.wait(3)
            return "ok"
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(AGYClient, "_create_unchecked", execute)
    clients = [AGYClient(cwd=str(tmp_path)) for _ in range(6)]
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(
                c.chat.completions.create,
                messages=[],
                extra_body={"hermes_session_id": str(i)},
            )
            for i, c in enumerate(clients)
        ]
        try:
            assert reached.wait(2)
            with pytest.raises(AGYBusyError):
                clients[2].chat.completions.create(
                    messages=[], extra_body={"hermes_session_id": "0"}
                )
        finally:
            release.set()
        assert [f.result() for f in futures] == ["ok"] * 6
    assert peak == 2


def test_account_refresh_does_not_block_and_applies_on_next_request(
    monkeypatch, tmp_path
):
    pool = AccountPool()
    monkeypatch.setenv("AGENT_LB_URL", "https://example.invalid")
    monkeypatch.setenv("AGENT_LB_API_KEY", "test")
    state_path = tmp_path / "account.json"
    state_path.write_text(json.dumps({"id": "old"}))
    monkeypatch.setenv("HERMES_AGY_ACCOUNT_STATE", str(state_path))
    entered, release, fetched = threading.Event(), threading.Event(), threading.Event()

    def request(*args):
        entered.set()
        assert release.wait(3)
        fetched.set()
        return 200, {"account": {"id": "new"}, "token": {"test": "credential"}}

    monkeypatch.setattr(pool, "_request", request)
    assert pool.ensure() is False
    try:
        assert entered.wait(1)
        # Would deadlock against a synchronous refresh holding the pool lock.
        assert pool.ensure() is False
        assert pool.current_id() == "old"
    finally:
        release.set()
    assert fetched.wait(1)
    until = time.monotonic() + 2
    while pool._refreshing and time.monotonic() < until:
        time.sleep(0.01)
    assert pool.ensure() is True
    assert pool.current_id() == "new"


def test_denied_retry_can_fail_fast(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_AGY_DENIED_RETRY", "0")
    command = stub(
        tmp_path,
        "event({'event':'result','result':{'response':'','denied_actions':['command']}})",
    )
    client = AGYClient(command=command, cwd=str(tmp_path), timeout=2)
    with pytest.raises(AGYProcessError, match="retry disabled"):
        client.chat.completions.create(messages=[])


def test_persistent_stdout_without_newline_is_bounded(tmp_path):
    command = stub(
        tmp_path, "sys.stdout.write('x'*65536); sys.stdout.flush(); time.sleep(30)"
    )
    client = AGYClient(
        command=command,
        cwd=str(tmp_path),
        persistent=True,
        timeout=2,
        max_stdout_bytes=1024,
        terminate_grace=0.1,
    )
    with pytest.raises(AGYProtocolError, match="stdout"):
        client.chat.completions.create(
            messages=[], extra_body={"hermes_session_id": "bounded"}
        )
    assert not client._active_sessions
