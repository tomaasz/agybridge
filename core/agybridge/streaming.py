"""Incremental text delivery with bounded buffering and per-request cancellation."""

from __future__ import annotations

import contextvars
import json
import queue
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from .protocol import AGYProcessError, AGYProtocolError


@dataclass
class StreamState:
    emit: Callable[[str], None]
    cancelled: threading.Event
    text: str = ""
    pending: str = ""
    enabled: bool = True

    def abort(self) -> BaseException | None:
        return AGYProcessError("AGY stream cancelled") if self.cancelled.is_set() else None

    def line(self, raw: bytes) -> None:
        """Only explicit text-delta events are safe to interpret as increments.

        Step updates are status/snapshots, not tokens. Never expose reasoning,
        internal actions, or unvalidated tool requests as answer text.
        """
        if self.abort():
            raise self.abort()
        if not self.enabled:
            return
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError):
            return  # authoritative parser rejects malformed output
        if not isinstance(event, dict) or event.get("event") not in {"delta", "text_delta"}:
            return
        delta = event.get("delta", event.get("content", event.get("text")))
        if isinstance(delta, dict):
            delta = delta.get("content", delta.get("text"))
        if not isinstance(delta, str):
            return
        self.pending += delta
        # Keep markup-looking tails until validation. This also handles tags
        # split across deltas. Tool-enabled calls are buffered entirely.
        safe = self.pending.find("<")
        if safe < 0:
            safe = len(self.pending)
        text, self.pending = self.pending[:safe], self.pending[safe:]
        self.text += text
        if text:
            self.emit(text)

    def finish(self, text: str) -> str:
        error = self.abort()
        if error is not None:
            raise error
        if not text.startswith(self.text):
            raise AGYProtocolError("AGY final response differs from streamed text")
        return text[len(self.text):]


STATE: contextvars.ContextVar[StreamState | None] = contextvars.ContextVar("agy_stream", default=None)


def text_chunk(model: str, text: str) -> Any:
    delta = SimpleNamespace(content=text, tool_calls=None, role=None,
                            reasoning=None, reasoning_content=None)
    return SimpleNamespace(model=model, usage=None,
                           choices=[SimpleNamespace(index=0, delta=delta, finish_reason=None)])


class CompletionStream:
    """Lazy iterator; closing it cancels this request, not the shared client."""

    def __init__(self, client: Any, model: str | None, call: dict[str, Any]) -> None:
        self.client, self.model, self.call = client, model or client.default_model, dict(call)
        self._context = contextvars.copy_context()
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=32)
        self._cancelled = threading.Event()
        self._worker: threading.Thread | None = None
        self._done = False

    def _put(self, kind: str, value: Any) -> None:
        while not self._cancelled.is_set():
            try:
                self._queue.put((kind, value), timeout=0.1)
                return
            except queue.Full:
                pass
        raise AGYProcessError("AGY stream cancelled")

    def _run(self) -> None:
        def emit(text: str) -> None:
            for start in range(0, len(text), 4096):
                self._put("chunk", text_chunk(self.model, text[start:start + 4096]))

        state = StreamState(emit, self._cancelled, enabled=not self.call.get("tools") or self.call.get("tool_choice") == "none")
        token = STATE.set(state)
        try:
            self.call["stream"] = False
            completion = self.client._create(model=self.model, **self.call)
            remainder = state.finish(completion.choices[0].message.content or "")
            # Preserve host-specific chunk shape and usage via its codec.
            completion.choices[0].message.content = remainder
            for chunk in self.client.ports.stream_codec(completion):
                self._put("chunk", chunk)
            self._put("done", None)
        except BaseException as exc:
            if not self._cancelled.is_set():
                self._put("error", exc)
        finally:
            STATE.reset(token)

    def __iter__(self) -> CompletionStream:
        return self

    def __next__(self) -> Any:
        if self._done:
            raise StopIteration
        if self._worker is None:
            self._worker = threading.Thread(target=self._context.run, args=(self._run,), daemon=True, name="agy-stream")
            self._worker.start()
        try:
            kind, value = self._queue.get(timeout=1.0)
        except queue.Empty:
            # A heartbeat is not a token. It keeps SSE alive when upstream emits
            # only a final result (or when tool calls need full validation).
            return SimpleNamespace(choices=[], usage=None, model=self.model)
        if kind == "chunk":
            return value
        self._done = True
        if kind == "error":
            raise value
        raise StopIteration

    def close(self) -> None:
        self._done = True
        self._cancelled.set()

    def __enter__(self) -> CompletionStream:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
