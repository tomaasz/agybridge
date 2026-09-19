from __future__ import annotations

import os
import sys

import pytest
from agybridge.process import run_process
from agybridge.protocol import AGYProcessError, AGYProtocolError, AGYTimeoutError


def test_run_process_success(tmp_path):
    # Pure Python command
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('hello world')"]
    stdout, stderr = run_process(
        cmd,
        cwd=str(tmp_path),
        env=dict(os.environ),
        timeout=5.0,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )
    assert stdout == b"hello world"
    assert stderr == b""


def test_run_process_stdin_pipeline(tmp_path):
    cmd = [
        sys.executable,
        "-c",
        "import sys; data = sys.stdin.read(); sys.stdout.write('echo:' + data)",
    ]
    stdout, stderr = run_process(
        cmd,
        cwd=str(tmp_path),
        env=dict(os.environ),
        stdin_data=b"input_data",
        timeout=5.0,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )
    assert stdout == b"echo:input_data"


def test_run_process_stdout_overflow(tmp_path):
    cmd = [sys.executable, "-c", "import sys; sys.stdout.write('A' * 200)"]
    with pytest.raises(AGYProtocolError, match="limit"):
        run_process(
            cmd,
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=5.0,
            max_stdout_bytes=50,
            max_stderr_bytes=1024,
        )


def test_run_process_timeout(tmp_path):
    cmd = [sys.executable, "-c", "import time; time.sleep(5)"]
    with pytest.raises(AGYTimeoutError):
        run_process(
            cmd,
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=0.2,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )


def test_run_process_error_exit(tmp_path):
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('fatal failure'); sys.exit(2)",
    ]
    with pytest.raises(AGYProcessError, match="status 2: fatal failure"):
        run_process(
            cmd,
            cwd=str(tmp_path),
            env=dict(os.environ),
            timeout=5.0,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )
