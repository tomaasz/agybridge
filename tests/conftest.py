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
    """Keep AGY logs and quota records out of the real home during tests."""
    monkeypatch.setenv("HERMES_AGY_LOG_DIR", str(tmp_path_factory.mktemp("agy-logs")))
    # A remembered quota would leak between tests and into the real ~/.hermes.
    monkeypatch.setenv(
        "HERMES_AGY_QUOTA_STATE",
        str(tmp_path_factory.mktemp("agy-quota") / "agy-quota.json"),
    )
