from __future__ import annotations

import stat
from pathlib import Path

from adapters.orca.launcher import generate_orca_launcher


def test_generate_orca_launcher_dry_run():
    path = generate_orca_launcher(target_path="/tmp/test-agy-agent", dry_run=True)
    assert path == "/tmp/test-agy-agent"
    assert not Path("/tmp/test-agy-agent").exists()


def test_generate_orca_launcher_write(tmp_path):
    target = tmp_path / "bin" / "agy-bridge-agent"
    path = generate_orca_launcher(target_path=target, dry_run=False)
    assert Path(path).exists()
    content = target.read_text(encoding="utf-8")
    assert "#!/usr/bin/env python3" in content
    assert "Orca ADE CLI Agent" in content
    # Verify executable bit
    mode = target.stat().st_mode
    assert mode & stat.S_IXUSR != 0
