from __future__ import annotations

from adapters.opencode.config import generate_opencode_config


def test_generate_opencode_config_v1():
    cfg = generate_opencode_config(
        base_url="http://127.0.0.1:9999/v1",
        api_key="secret-key",
        version="v1",
    )
    assert "provider" in cfg
    assert "agybridge" in cfg["provider"]
    assert (
        cfg["provider"]["agybridge"]["options"]["baseURL"] == "http://127.0.0.1:9999/v1"
    )
    assert cfg["provider"]["agybridge"]["options"]["apiKey"] == "secret-key"


def test_generate_opencode_config_v2():
    cfg = generate_opencode_config(
        base_url="http://127.0.0.1:9999/v1",
        api_key="secret-key",
        version="v2",
    )
    assert "providers" in cfg
    provider = cfg["providers"][0]
    assert provider["id"] == "agybridge"
    assert provider["baseURL"] == "http://127.0.0.1:9999/v1"
    assert provider["apiKey"] == "secret-key"
    model_ids = [m["id"] for m in provider["models"]]
    assert "gemini-3.8-flash-high" in model_ids
