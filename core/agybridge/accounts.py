"""The Google account AGY runs on, taken from the agent-lb pool.

agent-lb keeps the operator's Google accounts for AGY and decides which one a
station uses. AGY reads its login from
``~/.gemini/antigravity-cli/antigravity-oauth-token``
when a process starts, so the bridge puts the chosen account's token there
before launching AGY. When AGY reports a spent quota the bridge tells agent-lb,
which marks that account and answers with the next one; the request is then
retried on it.

Without an agent-lb URL and station key the bridge leaves the file alone and
AGY runs on whatever account was logged in with ``agy`` on this machine.

The URL and key come from ``AGENT_LB_URL`` / ``AGENT_LB_API_KEY`` or from
``~/.config/agybridge/agybridge.env`` (written by agy-setup.sh), so a Hermes
gateway needs no extra configuration. They are never forwarded to AGY.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows CI
    fcntl = None  # type: ignore[assignment]

CONFIG_FILE_ENV = "AGYBRIDGE_CONFIG"
TOKEN_FILE_ENV = "HERMES_AGY_TOKEN_FILE"
STATE_FILE_ENV = "HERMES_AGY_ACCOUNT_STATE"
REFRESH_SECONDS = 300.0
RETRY_AFTER_FAILURE_SECONDS = 60.0
HTTP_TIMEOUT = 10.0

logger = logging.getLogger(__name__)


def config_file() -> Path:
    raw = os.environ.get(CONFIG_FILE_ENV, "").strip()
    return (
        Path(raw).expanduser()
        if raw
        else Path.home() / ".config" / "agybridge" / "agybridge.env"
    )


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        values[name.strip()] = value.strip().strip("'\"")
    return values


def pool_settings() -> tuple[str, str] | None:
    """(agent-lb URL, station key), or None when the pool is not configured."""
    url = os.environ.get("AGENT_LB_URL", "").strip()
    key = os.environ.get("AGENT_LB_API_KEY", "").strip()
    if not (url and key):
        stored = _read_env_file(config_file())
        url = url or stored.get("AGENT_LB_URL", "")
        key = key or stored.get("AGENT_LB_API_KEY", "")
    if not (url and key):
        return None
    return url.rstrip("/"), key


def token_file() -> Path:
    raw = os.environ.get(TOKEN_FILE_ENV, "").strip()
    return (
        Path(raw).expanduser()
        if raw
        else Path.home() / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    )


def state_file() -> Path:
    raw = os.environ.get(STATE_FILE_ENV, "").strip()
    return (
        Path(raw).expanduser()
        if raw
        else token_file().with_name("agybridge-account.json")
    )


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


class AccountPool:
    """Keeps AGY's token file on the account agent-lb picked for this station."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_check = 0.0

    def current(self) -> dict[str, str] | None:
        """The pool account AGY is set up with ({"id", "email"}), if any."""
        try:
            data = json.loads(state_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if isinstance(data, dict) and isinstance(data.get("id"), str):
            return {"id": data["id"], "email": str(data.get("email", ""))}
        return None

    def current_id(self) -> str:
        """Account id for keying sessions and quota records ("" without a pool)."""
        if pool_settings() is None:
            return ""
        account = self.current()
        return account["id"] if account else ""

    def _request(
        self, settings: tuple[str, str], method: str, path: str, body: Any = None
    ) -> tuple[int, dict[str, Any]]:
        url, key = settings
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            url + path,
            data=data,
            method=method,
            headers={"x-api-key": key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            with contextlib.suppress(ValueError, OSError):
                return exc.code, json.loads(exc.read() or b"{}")
            return exc.code, {}

    def _apply(self, reply: dict[str, Any]) -> bool:
        """Install the account agent-lb answered with; True if it changed."""
        account = reply.get("account")
        token = reply.get("token")
        if not isinstance(account, dict) or not isinstance(token, dict):
            return False
        account_id = str(account.get("id", ""))
        if not account_id:
            return False
        lock_path = state_file().with_name(state_file().name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            current = self.current()
            if current and current["id"] == account_id:
                # Same account: leave the file alone, AGY refreshes its own
                # access token in it.
                return False
            target = token_file()
            backup = target.with_name(target.name + ".bak-agybridge")
            if current is None and target.exists() and not backup.exists():
                # The login made with `agy` before the pool took over.
                _write_private(backup, target.read_text(encoding="utf-8"))
            _write_private(target, json.dumps(token))
            _write_private(
                state_file(),
                json.dumps(
                    {
                        "id": account_id,
                        "email": str(account.get("email", "")),
                        "since": time.time(),
                    }
                ),
            )
        logger.info(
            "AGY account switched to %s (agent-lb pool)",
            account.get("email") or account_id,
        )
        return True

    def _apply_safely(self, reply: dict[str, Any]) -> bool:
        # A file that cannot be written must not fail the request: AGY then
        # keeps running on the account it has.
        try:
            return self._apply(reply)
        except OSError as exc:
            logger.warning("could not install the AGY account from agent-lb (%s)", exc)
            return False

    def ensure(self) -> bool:
        """Sync with agent-lb at most every few minutes; True if the account changed.

        A network or server failure keeps the current account: AGY can still
        run on it.
        """
        settings = pool_settings()
        if settings is None:
            return False
        with self._lock:
            now = time.monotonic()
            if now < self._next_check:
                return False
            current = self.current()
            path = "/agy/credential" + (f"?current={current['id']}" if current else "")
            try:
                status, reply = self._request(settings, "GET", path)
            except (OSError, ValueError) as exc:
                self._next_check = now + RETRY_AFTER_FAILURE_SECONDS
                logger.warning(
                    "agent-lb AGY pool unreachable (%s); keeping the current account",
                    exc,
                )
                return False
            if status != 200:
                self._next_check = now + RETRY_AFTER_FAILURE_SECONDS
                if status != 404:
                    logger.warning(
                        "agent-lb AGY pool answered %s (%s); keeping the current account",
                        status,
                        reply.get("error", ""),
                    )
                return False
            self._next_check = now + REFRESH_SECONDS
            return self._apply_safely(reply)

    def report_quota(self, message: str, retry_after: float | None) -> bool:
        """Tell agent-lb the current account is spent; True if another one was installed."""
        settings = pool_settings()
        if settings is None:
            return False
        current = self.current()
        body = {
            "accountId": current["id"] if current else None,
            "message": message[:300],
            "resetSeconds": retry_after,
        }
        with self._lock:
            try:
                status, reply = self._request(settings, "POST", "/agy/quota", body)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "could not report the spent AGY quota to agent-lb (%s)", exc
                )
                return False
            if status != 200:
                logger.warning(
                    "agent-lb has no other AGY account with free quota (%s)",
                    reply.get("error", status),
                )
                return False
            self._next_check = time.monotonic() + REFRESH_SECONDS
            return self._apply_safely(reply)


ACCOUNTS = AccountPool()
