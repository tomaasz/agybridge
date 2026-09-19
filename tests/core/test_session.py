from __future__ import annotations

from agybridge.session import (
    ConversationStore,
    history_digest,
)


def test_history_digest_deterministic():
    msgs1 = [{"role": "user", "content": "hello"}]
    msgs2 = [{"role": "user", "content": "hello"}]
    assert history_digest(msgs1) == history_digest(msgs2)

    msgs3 = [{"role": "user", "content": "world"}]
    assert history_digest(msgs1) != history_digest(msgs3)


def test_conversation_store_record_and_resume(monkeypatch, tmp_path):
    store_file = tmp_path / "test-store.json"
    monkeypatch.setenv("HERMES_AGY_SESSION_STORE", str(store_file))

    store = ConversationStore()
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello", "tool_calls": [{"id": "tc1"}]},
    ]
    store.record(
        "session_abc",
        conversation_id="conv_123",
        compat="compat_v1",
        history=history[:1],
        reply_ids=("tc1",),
    )

    extended = [
        *history,
        {"role": "tool", "content": "result"},
    ]
    resume = store.resume_point("session_abc", "compat_v1", extended)
    assert resume is not None
    conv_id, delta = resume
    assert conv_id == "conv_123"
    assert len(delta) == 1
    assert delta[0]["role"] == "tool"

    # Incompatible digest
    assert store.resume_point("session_abc", "compat_v2", extended) is None

    # Forget
    store.forget("session_abc")
    assert store.resume_point("session_abc", "compat_v1", extended) is None
