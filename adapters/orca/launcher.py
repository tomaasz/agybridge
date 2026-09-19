"""Orca ADE launcher and executable wrapper generator for agybridge."""

from __future__ import annotations

import stat
from pathlib import Path

WRAPPER_SCRIPT_TEMPLATE = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"Orca ADE CLI Agent wrapper for agybridge.\"\"\"

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _p in (_here, _here.parent / "lib" / "python", _here / "core"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from agybridge.engine import AGYClient
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Orca ADE AGY Bridge Agent")
    parser.add_argument("--model", default="gemini-3.8-flash-high", help="AGY model")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high"], help="Reasoning effort")
    parser.add_argument("prompt", nargs="?", default=None, help="Prompt text (or read from stdin)")
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            parser.print_help()
            sys.exit(1)

    client = AGYClient(effort=args.effort, default_model=args.model)
    completion = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=args.effort,
    )
    print(completion.choices[0].message.content or "")


if __name__ == "__main__":
    main()
"""


def generate_orca_launcher(
    target_path: Path | str | None = None,
    dry_run: bool = False,
) -> str:
    """Generate and optionally write the agy-bridge-agent executable wrapper."""
    path = (
        Path(target_path).expanduser().resolve()
        if target_path
        else Path("~/.local/bin/agy-bridge-agent").expanduser().resolve()
    )
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(WRAPPER_SCRIPT_TEMPLATE, encoding="utf-8")
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)
