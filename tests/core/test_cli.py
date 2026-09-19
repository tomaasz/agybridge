from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from agybridge.cli import main


def test_cli_config_opencode():
    f = io.StringIO()
    with redirect_stdout(f):
        main(
            [
                "config",
                "opencode",
                "--base-url",
                "http://localhost:8791/v1",
                "--token",
                "test-tok",
            ]
        )
    output = f.getvalue()
    data = json.loads(output)
    assert data["providers"][0]["baseURL"] == "http://localhost:8791/v1"
    assert data["providers"][0]["apiKey"] == "test-tok"


def test_cli_orca_install_dry_run():
    f = io.StringIO()
    with redirect_stdout(f):
        main(["orca", "install", "--path", "/tmp/dummy-orca-agent", "--dry-run"])
    output = f.getvalue()
    assert "[dry-run]" in output
    assert "/tmp/dummy-orca-agent" in output
