"""Unified command-line interface for agybridge."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections.abc import Sequence


def _cmd_serve(args: argparse.Namespace) -> None:
    from adapters.openai_http.server import run_server

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_server(host=args.host, port=args.port, token=args.token)


def _cmd_mcp(args: argparse.Namespace) -> None:
    from adapters.mcp.server import run_mcp_server_async

    asyncio.run(run_mcp_server_async(transport=args.transport, port=args.port))


def _cmd_config_opencode(args: argparse.Namespace) -> None:
    from adapters.opencode.config import generate_opencode_config

    cfg = generate_opencode_config(
        base_url=args.base_url, api_key=args.token, version=args.format_version
    )
    print(json.dumps(cfg, indent=2, ensure_ascii=False))


def _cmd_orca_install(args: argparse.Namespace) -> None:
    from adapters.orca.launcher import generate_orca_launcher

    path = generate_orca_launcher(target_path=args.path, dry_run=args.dry_run)
    if args.dry_run:
        print(f"[dry-run] Would create Orca launcher at: {path}")
    else:
        print(f"Installed Orca launcher at: {path}")


def _cmd_quota(args: argparse.Namespace) -> None:
    from .quota import QUOTA, format_duration, state_path

    if args.clear:
        QUOTA.clear()
        print("Forgot every remembered AGY quota.")
        return
    entries = QUOTA.entries()
    if not entries:
        print(
            f"No spent AGY quota remembered ({state_path() or 'remembering is off'})."
        )
        return
    now = time.time()
    for model, entry in sorted(entries.items()):
        until = float(entry.get("until", 0))
        probe = float(entry.get("probe_at", 0))
        print(
            f"{model}: spent, resets in {format_duration(until - now)}"
            f" (next probe in {format_duration(probe - now)}): {entry.get('message', '')}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agybridge",
        description="agybridge: Universal modular AGY reasoning bridge",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: serve
    serve_parser = subparsers.add_parser(
        "serve", help="Start local OpenAI-compatible HTTP server"
    )
    serve_parser.add_argument(
        "--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8791, help="Port number (default: 8791)"
    )
    serve_parser.add_argument(
        "--token", default=None, help="Bearer token for authentication"
    )
    serve_parser.set_defaults(func=_cmd_serve)

    # Subcommand: mcp
    mcp_parser = subparsers.add_parser(
        "mcp", help="Start MCP server exposing agy_reason"
    )
    mcp_parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "http"],
        help="MCP transport (default: stdio)",
    )
    mcp_parser.add_argument(
        "--port", type=int, default=8792, help="Port for SSE transport (default: 8792)"
    )
    mcp_parser.set_defaults(func=_cmd_mcp)

    # Subcommand: quota
    quota_parser = subparsers.add_parser(
        "quota", help="Show or forget remembered spent AGY quotas"
    )
    quota_parser.add_argument(
        "--clear", action="store_true", help="Forget every remembered quota"
    )
    quota_parser.set_defaults(func=_cmd_quota)

    # Subcommand: config
    config_parser = subparsers.add_parser(
        "config", help="Generate configuration snippets"
    )
    config_sub = config_parser.add_subparsers(dest="target", required=True)

    opencode_parser = config_sub.add_parser(
        "opencode", help="Generate OpenCode provider configuration JSON"
    )
    opencode_parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8791/v1",
        help="Base URL of agybridge HTTP server",
    )
    opencode_parser.add_argument("--token", default="agybridge-local", help="API token")
    opencode_parser.add_argument(
        "--format-version",
        default="v2",
        choices=["v1", "v2"],
        help="OpenCode config version (v1 or v2)",
    )
    opencode_parser.set_defaults(func=_cmd_config_opencode)

    # Subcommand: orca
    orca_parser = subparsers.add_parser("orca", help="Orca ADE integration commands")
    orca_sub = orca_parser.add_subparsers(dest="orca_action", required=True)

    orca_install = orca_sub.add_parser(
        "install", help="Install agy-bridge-agent executable wrapper"
    )
    orca_install.add_argument(
        "--path",
        default=None,
        help="Target installation path (default: ~/.local/bin/agy-bridge-agent)",
    )
    orca_install.add_argument(
        "--dry-run", action="store_true", help="Print target path without writing"
    )
    orca_install.set_defaults(func=_cmd_orca_install)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
