from __future__ import annotations

from agybridge.security import (
    _positive_int,
    _positive_number,
    _redact,
    _safe_child_env,
)


def test_redact_secrets():
    raw = "Bearer secret123, api_key: topsecret, sk_live_123456789012"
    redacted = _redact(raw)
    assert "secret123" not in redacted
    assert "topsecret" not in redacted
    assert "sk_live" not in redacted
    assert "[REDACTED]" in redacted


def test_safe_child_env(monkeypatch):
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("GOOGLE_API_KEY", "sensitive")
    monkeypatch.setenv("GEMINI_SECRET", "sensitive2")
    monkeypatch.setenv("CUSTOM_ALLOWED", "allowed_value")

    child_env = _safe_child_env(extra_names=["CUSTOM_ALLOWED"])
    assert child_env.get("HOME") == "/home/test"
    assert "GOOGLE_API_KEY" not in child_env
    assert "GEMINI_SECRET" not in child_env
    assert child_env.get("CUSTOM_ALLOWED") == "allowed_value"


def test_positive_number_and_int():
    assert _positive_number(10, 1.0, label="x") == 10.0
    assert _positive_number(None, 5.0, label="x") == 5.0
    assert _positive_int(10, 1, label="x") == 10
    assert _positive_int(None, 5, label="x") == 5
