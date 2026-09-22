from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"

for path_str in (str(ROOT), str(CORE)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture(autouse=True)
def _agy_logs_in_tmp(monkeypatch, tmp_path_factory):
    """Keep AGY logs, quota records and the AGY login out of the real home."""
    monkeypatch.setenv("HERMES_AGY_LOG_DIR", str(tmp_path_factory.mktemp("agy-logs")))
    # A remembered quota would leak between tests and into the real ~/.hermes.
    monkeypatch.setenv(
        "HERMES_AGY_QUOTA_STATE",
        str(tmp_path_factory.mktemp("agy-quota") / "agy-quota.json"),
    )
    # Never touch the real AGY login or pick up a real agent-lb pool.
    isolated = tmp_path_factory.mktemp("agy-account")
    monkeypatch.setenv("AGYBRIDGE_CONFIG", str(isolated / "agybridge.env"))
    monkeypatch.setenv("HERMES_AGY_TOKEN_FILE", str(isolated / "token.json"))
    monkeypatch.setenv("HERMES_AGY_ACCOUNT_STATE", str(isolated / "account.json"))
    monkeypatch.delenv("AGENT_LB_URL", raising=False)
    monkeypatch.delenv("AGENT_LB_API_KEY", raising=False)
