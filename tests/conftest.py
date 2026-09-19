from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"

for path_str in (str(ROOT), str(CORE)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
