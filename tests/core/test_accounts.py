from __future__ import annotations

import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from agybridge import accounts
from agybridge.accounts import ACCOUNTS
from agybridge.engine import AGYClient
from agybridge.protocol import AGYQuotaError
from agybridge.quota import QUOTA

QUOTA_LINE = (
    "Run: attempt 1 failed (RESOURCE_EXHAUSTED (code 429): Individual quota "
    "reached. Please upgrade your subscription. Resets in 5h0m0s.), retrying in 4s"
)


def _token(name: str) -> dict:
    return {"token": {"refresh_token": f"refresh-{name}"}, "id_token": f"id-{name}"}


class FakeAgentLB:
    """agent-lb's /agy/credential and /agy/quota over a list of accounts."""

    def __init__(self, names: list[str]):
        self.accounts = [{"id": f"agy-{n}", "email": f"{n}@example.com"} for n in names]
        self.spent: set[str] = set()
        self.reports: list[dict] = []
        self.requests = 0
        pool = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _pick(self, current=None):
                free = [a for a in pool.accounts if a["id"] not in pool.spent]
                chosen = next((a for a in free if a["id"] == current), None) or (
                    free[0] if free else None
                )
                if not chosen:
                    return self._reply(
                        404, {"ok": False, "error": "no AGY account with free quota"}
                    )
                name = chosen["id"].removeprefix("agy-")
                self._reply(200, {"ok": True, "account": chosen, "token": _token(name)})

            def do_GET(self):
                pool.requests += 1
                assert self.headers["x-api-key"] == "tc-station"
                current = self.path.partition("current=")[2] or None
                self._pick(current)

            def do_POST(self):
                pool.requests += 1
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pool.reports.append(body)
                if body.get("accountId"):
                    pool.spent.add(body["accountId"])
                self._pick()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def pool(monkeypatch):
    fake = FakeAgentLB(["a", "b"])
    monkeypatch.setenv("AGENT_LB_URL", fake.url)
    monkeypatch.setenv("AGENT_LB_API_KEY", "tc-station")
    monkeypatch.setattr(ACCOUNTS, "_next_check", 0.0)
    yield fake
    fake.close()


def _agy_stub(tmp_path: Path) -> Path:
    """AGY stand-in: account "a" has a spent quota, any other account answers."""
    token = os.environ["HERMES_AGY_TOKEN_FILE"]
    script = tmp_path / "agy-pool.py"
    script.write_text(
        f"#!{sys.executable}\nimport json, sys, time\n"
        "log = sys.argv[sys.argv.index('--log-file') + 1]\n"
        "sys.stdin.read()\n"
        f"who = json.load(open({token!r}))['token']['refresh_token'].removeprefix('refresh-')\n"
        "if who == 'a':\n"
        f"    open(log, 'a').write({QUOTA_LINE!r} + '\\n'); time.sleep(60)\n"
        "print(json.dumps({'event': 'result', 'result': {'response': 'answer from ' + who}}), flush=True)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def test_without_a_pool_the_agy_login_is_left_alone(tmp_path):
    token = Path(os.environ["HERMES_AGY_TOKEN_FILE"])
    token.write_text(json.dumps(_token("local")))
    assert ACCOUNTS.ensure() is False
    assert ACCOUNTS.current_id() == ""
    assert json.loads(token.read_text()) == _token("local")


def test_pool_account_is_installed_and_the_local_login_kept(pool):
    token = Path(os.environ["HERMES_AGY_TOKEN_FILE"])
    token.write_text(json.dumps(_token("local")))
    assert ACCOUNTS.ensure() is True
    assert json.loads(token.read_text()) == _token("a")
    assert stat.S_IMODE(token.stat().st_mode) == 0o600
    backup = token.with_name(token.name + ".bak-agybridge")
    assert json.loads(backup.read_text()) == _token("local")
    assert ACCOUNTS.current() == {"id": "agy-a", "email": "a@example.com"}
    assert ACCOUNTS.current_id() == "agy-a"


def test_same_account_keeps_the_token_agy_refreshed(pool):
    token = Path(os.environ["HERMES_AGY_TOKEN_FILE"])
    ACCOUNTS.ensure()
    refreshed = {
        **_token("a"),
        "token": {"refresh_token": "refresh-a", "access_token": "new"},
    }
    token.write_text(json.dumps(refreshed))
    ACCOUNTS._next_check = 0.0
    assert ACCOUNTS.ensure() is False
    assert json.loads(token.read_text()) == refreshed


def test_sync_is_rate_limited(pool):
    ACCOUNTS.ensure()
    requests = pool.requests
    ACCOUNTS.ensure()
    ACCOUNTS.ensure()
    assert pool.requests == requests


def test_unreachable_pool_keeps_the_current_account(pool, monkeypatch):
    ACCOUNTS.ensure()
    monkeypatch.setenv("AGENT_LB_URL", "http://127.0.0.1:9")  # nothing listens
    ACCOUNTS._next_check = 0.0
    assert ACCOUNTS.ensure() is False
    assert ACCOUNTS.current_id() == "agy-a"


def test_spent_quota_moves_the_request_to_the_next_account(pool, tmp_path):
    client = AGYClient(
        command=str(_agy_stub(tmp_path)),
        cwd=str(tmp_path),
        timeout=30,
        terminate_grace=0.2,
    )
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}]
    )
    assert result.choices[0].message.content == "answer from b"
    assert pool.reports[0]["accountId"] == "agy-a"
    assert pool.reports[0]["resetSeconds"] == pytest.approx(5 * 3600, abs=5)
    assert ACCOUNTS.current_id() == "agy-b"
    # The spent quota is remembered for account "a" only.
    assert set(QUOTA.entries()) == {"agy-a/gemini-3.8-flash"}


def test_every_account_spent_raises_the_quota_error(pool, tmp_path):
    pool.spent.add("agy-b")
    client = AGYClient(
        command=str(_agy_stub(tmp_path)),
        cwd=str(tmp_path),
        timeout=30,
        terminate_grace=0.2,
    )
    with pytest.raises(AGYQuotaError, match="Resets in 5h"):
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert [r["accountId"] for r in pool.reports] == ["agy-a"]


def test_settings_come_from_the_agybridge_config_file(monkeypatch):
    config = Path(os.environ["AGYBRIDGE_CONFIG"])
    config.write_text(
        "# agybridge\nAGENT_LB_URL=https://lb.example/\nAGENT_LB_API_KEY='tc-x'\n"
    )
    assert accounts.pool_settings() == ("https://lb.example", "tc-x")
