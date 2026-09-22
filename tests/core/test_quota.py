from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest
from agybridge import agylog
from agybridge.agylog import AGYLogWatcher, new_log_path
from agybridge.engine import AGYClient
from agybridge.protocol import (
    AGYProcessError,
    AGYProtocolError,
    AGYQuotaError,
    _parse_stream_json,
    quota_message,
)
from agybridge.session import POOL

# Verbatim shape of AGY's log line when the account quota is spent.
QUOTA_LINE = (
    "I0922 11:07:08.017637     273 run.go:395] Run: attempt 1 failed "
    "(RESOURCE_EXHAUSTED (code 429): Individual quota reached. Please upgrade "
    "your subscription to increase your limits. Resets in 94h6m56s.), retrying in 4s"
)


@pytest.fixture(autouse=True)
def _log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(agylog.LOG_DIR_ENV, str(tmp_path / "agy-logs"))
    monkeypatch.setenv("HERMES_AGY_SESSION_STORE", str(tmp_path / "store.json"))


def _result(**result) -> bytes:
    return (json.dumps({"event": "result", "result": result}) + "\n").encode()


def test_quota_message_matches_only_a_spent_quota():
    assert quota_message(QUOTA_LINE) == (
        "Individual quota reached. Please upgrade your subscription to increase "
        "your limits. Resets in 94h6m56s."
    )
    # A plain rate limit clears within AGY's own retries; do not cut it short.
    assert quota_message("RESOURCE_EXHAUSTED (code 429): Too many requests") is None
    assert quota_message("UNAVAILABLE (code 503): overloaded") is None


def test_error_result_raises_quota_error_instead_of_empty_reply():
    error = "API error (attempt 3): " + QUOTA_LINE.split("failed (", 1)[1]
    with pytest.raises(AGYQuotaError, match="Resets in 94h6m56s"):
        _parse_stream_json(_result(status="ERROR", response="", error=error))


def test_error_result_without_quota_is_a_process_error():
    with pytest.raises(AGYProcessError, match="AGY run failed: boom") as info:
        _parse_stream_json(_result(status="ERROR", response="", error="boom"))
    assert not isinstance(info.value, (AGYQuotaError, AGYProtocolError))


def test_error_status_with_text_keeps_the_text():
    parsed = _parse_stream_json(_result(status="ERROR", response="partial answer"))
    assert parsed.text == "partial answer"


def test_log_watcher_handles_late_file_and_split_lines(tmp_path):
    log = tmp_path / "agy.log"
    watcher = AGYLogWatcher(log, interval=0)
    assert watcher.check() is None  # AGY has not created the file yet
    head, tail = QUOTA_LINE[:60], QUOTA_LINE[60:]
    log.write_text("I0922 starting\n" + head)
    assert watcher.check() is None
    with log.open("a") as f:
        f.write(tail + "\n")
    assert "Resets in 94h6m56s" in watcher.check()
    watcher.close()


def test_new_log_path_prunes_old_bridge_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(agylog, "KEEP_LOGS", 3)
    directory = tmp_path / "agy-logs"
    directory.mkdir()
    for i in range(5):
        old = directory / f"agybridge-old-{i}.log"
        old.write_text("x")
        os.utime(old, (1000 + i, 1000 + i))
    (directory / "cli-20260922_103207.log").write_text("AGY's own log")
    path = new_log_path()
    assert path is not None and path.parent == directory
    remaining = sorted(p.name for p in directory.glob("agybridge-*.log"))
    assert remaining == [
        "agybridge-old-2.log",
        "agybridge-old-3.log",
        "agybridge-old-4.log",
    ]
    assert (directory / "cli-20260922_103207.log").exists()


def _stub(tmp_path: Path, body: str) -> Path:
    """AGY stand-in that logs a spent quota, then keeps retrying like AGY does."""
    script = tmp_path / "agy-quota.py"
    script.write_text(
        f"#!{sys.executable}\nimport json, sys, time\n"
        "log = sys.argv[sys.argv.index('--log-file') + 1]\n"
        "def quota():\n"
        f"    open(log, 'a').write({QUOTA_LINE!r} + '\\n')\n" + body,
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_oneshot_stops_agy_as_soon_as_the_quota_is_logged(tmp_path):
    stub = _stub(
        tmp_path,
        "sys.stdin.read()\n"
        "print(json.dumps({'event': 'init', 'init': {}}), flush=True)\n"
        "quota()\n"
        "time.sleep(60)\n",
    )
    client = AGYClient(
        command=str(stub), cwd=str(tmp_path), timeout=30, terminate_grace=0.2
    )
    started = time.monotonic()
    with pytest.raises(AGYQuotaError, match="Resets in 94h6m56s"):
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert time.monotonic() - started < 10
    assert not client._active_processes


def test_persistent_turn_stops_agy_as_soon_as_the_quota_is_logged(tmp_path, request):
    request.addfinalizer(POOL.clear)
    stub = _stub(
        tmp_path,
        "print(json.dumps({'event': 'init', 'init': {'conversation_id': 'c1'}}), flush=True)\n"
        "for line in iter(sys.stdin.readline, ''):\n"
        "    quota()\n"
        "    time.sleep(60)\n",
    )
    client = AGYClient(
        command=str(stub),
        cwd=str(tmp_path),
        timeout=30,
        terminate_grace=0.2,
        persistent=True,
    )
    started = time.monotonic()
    with pytest.raises(AGYQuotaError):
        client.chat.completions.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"hermes_session_id": "s1"},
        )
    assert time.monotonic() - started < 10
    assert not client._active_sessions


def test_agy_gets_a_private_log_file(tmp_path):
    capture = tmp_path / "argv.json"
    script = tmp_path / "agy-ok.py"
    script.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(capture)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'event': 'result', 'result': {'response': 'ok'}}), flush=True)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    client = AGYClient(command=str(script), cwd=str(tmp_path), timeout=10)
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert result.choices[0].message.content == "ok"
    argv = json.loads(capture.read_text())
    log_file = Path(argv[argv.index("--log-file") + 1])
    assert log_file.parent == tmp_path / "agy-logs"
    assert log_file.name.startswith("agybridge-")
    assert stat.S_IMODE(log_file.parent.stat().st_mode) == 0o700
