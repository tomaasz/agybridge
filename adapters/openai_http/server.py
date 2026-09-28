"""Zero-dependency, localhost-only OpenAI-compatible HTTP server using stdlib http.server."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import time
import uuid
import itertools
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agybridge.engine import SESSION_ID_FIELD, AGYClient
from agybridge.protocol import (
    AGYBusyError,
    AGYProcessError,
    AGYProtocolError,
    AGYQuotaError,
    AGYTimeoutError,
)
from agybridge.ports import default_completion_to_stream_chunks

from .bridge import HTTP_PORTS, format_chat_completion_chunk, serialize_chat_completion

logger = logging.getLogger(__name__)

MAX_PAYLOAD_BYTES = 8 * 1024 * 1024  # 8 MiB strict limit


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound slow clients as well as requests waiting for AGY capacity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._connections = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self, request: Any, client_address: Any) -> None:
        if not self._connections.acquire(blocking=False):
            try:
                request.settimeout(1.0)
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\nRetry-After: 1\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._connections.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connections.release()


class OpenAIHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Request handler implementing OpenAI-compatible completions and models."""

    # Protocol version
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        self.request.settimeout(15.0)
        super().setup()

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
        self.close_connection = True
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

        if content_length < 0 or self.headers.get("Transfer-Encoding"):
            self._send_error(400, "Invalid request framing", code="bad_request")
            return

        if content_length > MAX_PAYLOAD_BYTES:
            self._send_error(
                413,
                f"Payload exceeds {MAX_PAYLOAD_BYTES} bytes limit",
                code="payload_too_large",
            )
            return

        try:
            body_bytes = self.rfile.read(content_length)
        except TimeoutError:
            self._send_error(408, "Request body timed out", code="request_timeout")
            return
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
                stream=stream,
                reasoning_effort=reasoning_effort,
                extra_body=extra_body,
            )
            if stream:
                self._send_stream(completion, model)
                return
        except AGYBusyError as exc:
            self._send_error(503, str(exc), code="overloaded", headers={"Retry-After": "1"})
            return
        except (BrokenPipeError, ConnectionResetError):
            return
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

        response_dict = serialize_chat_completion(completion)
        self._send_json(200, response_dict)

    def _send_stream(self, completion: Any, model: str) -> None:
        stream = completion if not hasattr(completion, "choices") else default_completion_to_stream_chunks(completion)
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        started = False
        try:
            iterator = iter(stream)
            # Preserve HTTP errors for preflight failures, including quota and
            # overload. A live stream yields a heartbeat within one second.
            first = next(iterator)
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            started = True
            for chunk in itertools.chain([first], iterator):
                usage = getattr(chunk, "usage", None)
                usage_dict = {name: getattr(usage, name, 0) for name in ("prompt_tokens", "completion_tokens", "total_tokens")} if usage is not None else None
                if chunk.choices:
                    choice = chunk.choices[0]
                    data = format_chat_completion_chunk(
                        chunk_id, model,
                        delta_content=getattr(choice.delta, "content", None),
                        delta_tool_calls=getattr(choice.delta, "tool_calls", None),
                        finish_reason=choice.finish_reason, usage=usage_dict,
                    )
                elif usage_dict is not None:
                    data = "data: " + json.dumps({"id": chunk_id, "object": "chat.completion.chunk", "model": model, "choices": [], "usage": usage_dict}) + "\n\n"
                else:
                    data = ": keep-alive\n\n"
                self.wfile.write(data.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            if not started:
                raise
            if isinstance(exc, TimeoutError) and not isinstance(exc, AGYTimeoutError):
                return
            # HTTP status is committed; terminate with an explicit stream error.
            payload = {"error": {"message": str(exc), "type": "api_error"}}
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()

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
        client = AGYClient(ports=HTTP_PORTS, persistent=True)
        client.prewarm()

    server = BoundedHTTPServer((host, port), OpenAIHTTPHandler)
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
        server.client.close()


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
