"""Zero-dependency, localhost-only OpenAI-compatible HTTP server using stdlib http.server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agybridge.engine import SESSION_ID_FIELD, AGYClient
from agybridge.protocol import (
    AGYProcessError,
    AGYProtocolError,
    AGYQuotaError,
    AGYTimeoutError,
)

from .bridge import HTTP_PORTS, format_chat_completion_chunk, serialize_chat_completion

logger = logging.getLogger(__name__)

MAX_PAYLOAD_BYTES = 8 * 1024 * 1024  # 8 MiB strict limit


class OpenAIHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Request handler implementing OpenAI-compatible completions and models."""

    # Protocol version
    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def server_token(self) -> str | None:
        return getattr(self.server, "token", None)

    @property
    def agy_client(self) -> AGYClient:
        return self.server.client

    def _send_json(
        self,
        status: int,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(
        self,
        status: int,
        message: str,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        payload = {
            "error": {
                "message": message,
                "type": error_type,
            }
        }
        if code:
            payload["error"]["code"] = code
        self._send_json(status, payload, headers)

    def _check_auth(self) -> bool:
        required_token = self.server_token
        if not required_token:
            return True
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            self._send_error(
                401, "Missing or invalid Bearer token", code="unauthorized"
            )
            return False
        token = auth_header[len("Bearer ") :].strip()
        if not secrets.compare_digest(token, required_token):
            self._send_error(401, "Invalid Bearer token", code="unauthorized")
            return False
        return True

    def do_GET(self) -> None:
        if self.path in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok"})
            return

        if not self._check_auth():
            return

        if self.path == "/v1/models":
            now = int(time.time())
            models = {
                "object": "list",
                "data": [
                    {
                        "id": "gemini-3.8-flash-high",
                        "object": "model",
                        "created": now,
                        "owned_by": "agybridge",
                    },
                    {
                        "id": "gemini-3.8-flash-low",
                        "object": "model",
                        "created": now,
                        "owned_by": "agybridge",
                    },
                ],
            }
            self._send_json(200, models)
            return

        self._send_error(404, f"Path not found: {self.path}", code="not_found")

    def do_POST(self) -> None:
        if not self._check_auth():
            return

        if self.path != "/v1/chat/completions":
            self._send_error(404, f"Path not found: {self.path}", code="not_found")
            return

        content_length_str = self.headers.get("Content-Length")
        if not content_length_str:
            self._send_error(411, "Length Required", code="length_required")
            return

        try:
            content_length = int(content_length_str)
        except ValueError:
            self._send_error(400, "Invalid Content-Length", code="bad_request")
            return

        if content_length > MAX_PAYLOAD_BYTES:
            self._send_error(
                413,
                f"Payload exceeds {MAX_PAYLOAD_BYTES} bytes limit",
                code="payload_too_large",
            )
            return

        body_bytes = self.rfile.read(content_length)
        if len(body_bytes) != content_length:
            self._send_error(400, "Incomplete body read", code="bad_request")
            return

        try:
            payload = json.loads(body_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_error(400, f"Invalid JSON payload: {exc}", code="bad_request")
            return

        if not isinstance(payload, dict):
            self._send_error(400, "JSON body must be an object", code="bad_request")
            return

        # Session mapping
        session_id = self.headers.get("X-AGY-Session") or payload.get("session_id")
        extra_body: dict[str, Any] = {}
        if isinstance(session_id, str) and session_id.strip():
            extra_body[SESSION_ID_FIELD] = session_id.strip()

        model = payload.get("model") or "gemini-3.8-flash-high"
        messages = payload.get("messages", [])
        tools = payload.get("tools")
        tool_choice = payload.get("tool_choice")
        stream = bool(payload.get("stream", False))
        reasoning_effort = payload.get("reasoning_effort")

        try:
            completion = self.agy_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,  # We fetch the completion object first
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
            )
        except AGYTimeoutError as exc:
            self._send_error(504, str(exc), error_type="timeout_error", code="timeout")
            return
        except AGYQuotaError as exc:
            # 429 lets OpenAI-compatible clients fall back to another provider.
            retry = (
                {"Retry-After": str(max(1, int(exc.retry_after)))}
                if exc.retry_after
                else None
            )
            self._send_error(
                429,
                str(exc),
                error_type="rate_limit_error",
                code="insufficient_quota",
                headers=retry,
            )
            return
        except (AGYProcessError, AGYProtocolError) as exc:
            self._send_error(502, str(exc), error_type="api_error", code="bad_gateway")
            return
        except ValueError as exc:
            self._send_error(400, str(exc), code="bad_request")
            return
        except Exception as exc:
            logger.exception("Unexpected error in AGY HTTP handler")
            self._send_error(
                500,
                f"Internal error: {exc}",
                error_type="server_error",
                code="internal_error",
            )
            return

        if not stream:
            response_dict = serialize_chat_completion(completion)
            self._send_json(200, response_dict)
            return

        # Streaming response
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        choice = completion.choices[0]
        msg = choice.message

        # First chunk: role
        self.wfile.write(format_chat_completion_chunk(chunk_id, model).encode("utf-8"))
        self.wfile.flush()

        # Content or tool calls delta
        if msg.content or msg.tool_calls:
            self.wfile.write(
                format_chat_completion_chunk(
                    chunk_id,
                    model,
                    delta_content=msg.content or None,
                    delta_tool_calls=msg.tool_calls or None,
                ).encode("utf-8")
            )
            self.wfile.flush()

        # Finish reason and usage chunk
        usage_dict = {
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(completion.usage, "completion_tokens", 0),
            "total_tokens": getattr(completion.usage, "total_tokens", 0),
        }
        self.wfile.write(
            format_chat_completion_chunk(
                chunk_id,
                model,
                finish_reason=choice.finish_reason,
                usage=usage_dict,
            ).encode("utf-8")
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(
            "%s - - [%s] %s",
            self.address_string(),
            self.log_date_time_string(),
            format % args,
        )


def create_server(
    host: str = "127.0.0.1",
    port: int = 8791,
    *,
    token: str | None = None,
    client: AGYClient | None = None,
) -> ThreadingHTTPServer:
    """Create a configured ThreadingHTTPServer instance."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Binding to non-localhost address %s may expose the sandbox!", host
        )

    if token is None:
        token = os.environ.get("AGYBRIDGE_TOKEN") or os.environ.get("OPENAI_API_KEY")

    if client is None:
        client = AGYClient(ports=HTTP_PORTS)

    server = ThreadingHTTPServer((host, port), OpenAIHTTPHandler)
    server.token = token  # type: ignore[attr-defined]
    server.client = client  # type: ignore[attr-defined]
    return server


def run_server(
    host: str = "127.0.0.1",
    port: int = 8791,
    *,
    token: str | None = None,
    client: AGYClient | None = None,
) -> None:
    """Run the OpenAI-compatible HTTP server indefinitely."""
    server = create_server(host, port, token=token, client=client)
    actual_token = server.token  # type: ignore[attr-defined]
    logger.info(
        "Starting agybridge HTTP server on http://%s:%d (auth: %s)",
        host,
        port,
        "token required" if actual_token else "open/no-token",
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping agybridge HTTP server")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the agybridge OpenAI-compatible HTTP server"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=8791, help="Port number (default: 8791)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Bearer authorization token (or set AGYBRIDGE_TOKEN)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    run_server(host=args.host, port=args.port, token=args.token)


if __name__ == "__main__":
    main()
