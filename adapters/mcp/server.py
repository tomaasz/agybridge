"""Universal MCP server exposing agy_reason for OpenClaw and any MCP hosts."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from agybridge.engine import SESSION_ID_FIELD, AGYClient

logger = logging.getLogger(__name__)

try:
    from mcp.server.mcpserver import MCPServer

    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    MCPServer = None  # type: ignore[assignment, misc]


def create_mcp_server(client: AGYClient | None = None) -> Any:
    """Create and configure the MCPServer instance exposing agy_reason."""
    if not HAS_MCP:
        raise RuntimeError(
            "The 'mcp' package is required to run the MCP server. "
            "Please install it with: pip install agybridge[mcp]"
        )

    agy_client = client or AGYClient()
    server = MCPServer("agybridge")

    @server.tool()
    def agy_reason(
        prompt: str,
        effort: str = "high",
        model: str = "gemini-3.8-flash-high",
        session_id: str | None = None,
    ) -> str:
        """Reason using the AGY engine within a secure sandbox."""
        extra_body = {}
        if session_id:
            extra_body[SESSION_ID_FIELD] = session_id

        completion = agy_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=effort,
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        return choice.message.content or ""

    return server


async def run_mcp_server_async(
    transport: str = "stdio",
    port: int = 8792,
    client: AGYClient | None = None,
) -> None:
    """Run the MCP server with the specified transport."""
    server = create_mcp_server(client=client)
    if transport == "stdio":
        await server.run_stdio_async()
    elif transport in ("sse", "http"):
        await server.run_sse_async(port=port)
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agybridge MCP server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "http"],
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8792,
        help="Port for SSE/HTTP transport (default: 8792)",
    )
    args = parser.parse_args()
    asyncio.run(run_mcp_server_async(transport=args.transport, port=args.port))


if __name__ == "__main__":
    main()
