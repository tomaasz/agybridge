"""Subprocess lifecycle management, bounded streaming, and process-tree termination."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .protocol import AGYProcessError, AGYProtocolError, AGYTimeoutError
from .security import _redact


def stop_process(
    process: subprocess.Popen[bytes], terminate_grace: float = 2.0
) -> None:
    """Stop a child process and its process group cleanly, then forcefully."""
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:  # pragma: no cover - exercised on Windows CI
            process.terminate()
        process.wait(timeout=terminate_grace)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    with contextlib.suppress(OSError):
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows CI
            process.kill()
    with contextlib.suppress(Exception):
        process.wait(timeout=terminate_grace)


def run_process(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    timeout: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    terminate_grace: float = 2.0,
    stdin_data: bytes | None = None,
    is_closed: Callable[[], bool] | None = None,
    on_process_start: Callable[[subprocess.Popen[bytes]], None] | None = None,
    on_process_done: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> tuple[bytes, bytes]:
    """Execute a subprocess with bounded stream readers and strict deadline."""
    popen_kwargs: dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL if stdin_data is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        raise AGYProcessError(
            f"AGY executable {Path(argv[0]).name!r} was not found"
        ) from exc
    except OSError as exc:
        raise AGYProcessError(
            f"AGY executable {Path(argv[0]).name!r} could not be started"
        ) from exc

    if on_process_start is not None:
        on_process_start(process)

    if is_closed is not None and is_closed():
        stop_process(process, terminate_grace)
        if on_process_done is not None:
            on_process_done(process)
        raise AGYProcessError("AGY client is closed")

    try:
        stdout, stderr = bytearray(), bytearray()
        overflow = threading.Event()

        def drain(
            stream: Any, target: bytearray, limit: int, *, keep_tail: bool
        ) -> None:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    return
                if keep_tail:
                    target.extend(chunk)
                    if len(target) > limit:
                        del target[:-limit]
                else:
                    if len(target) + len(chunk) > limit:
                        remaining = max(0, limit - len(target))
                        target.extend(chunk[:remaining])
                        overflow.set()
                        return
                    target.extend(chunk)

        out_thread = threading.Thread(
            target=drain,
            args=(process.stdout, stdout, max_stdout_bytes),
            kwargs={"keep_tail": False},
            daemon=True,
        )
        err_thread = threading.Thread(
            target=drain,
            args=(process.stderr, stderr, max_stderr_bytes),
            kwargs={"keep_tail": True},
            daemon=True,
        )
        out_thread.start()
        err_thread.start()

        if stdin_data is not None:

            def feed(stream: Any) -> None:
                with contextlib.suppress(OSError, ValueError):
                    stream.write(stdin_data)
                with contextlib.suppress(OSError, ValueError):
                    stream.close()

            threading.Thread(target=feed, args=(process.stdin,), daemon=True).start()

        deadline = time.monotonic() + timeout
        timed_out = False
        while process.poll() is None:
            if overflow.is_set():
                stop_process(process, terminate_grace)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                stop_process(process, terminate_grace)
                break
            time.sleep(0.02)

        out_thread.join(timeout=terminate_grace)
        err_thread.join(timeout=terminate_grace)

        if timed_out:
            raise AGYTimeoutError(f"AGY exceeded the {timeout:g}s request timeout")
        if overflow.is_set():
            raise AGYProtocolError(
                f"AGY stdout exceeded the {max_stdout_bytes}-byte limit"
            )
        if process.returncode:
            tail = _redact(stderr.decode("utf-8", errors="replace")).strip()
            detail = f": {tail}" if tail else ""
            raise AGYProcessError(
                f"AGY exited with status {process.returncode}{detail}"
            )
        return bytes(stdout), bytes(stderr)
    finally:
        if on_process_done is not None:
            on_process_done(process)
