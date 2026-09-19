"""OpenCode configuration generator for agybridge."""

from __future__ import annotations

from typing import Any


def generate_opencode_config(
    base_url: str = "http://127.0.0.1:8791/v1",
    api_key: str = "agybridge-local",
    version: str = "v2",
) -> dict[str, Any]:
    """Generate OpenCode provider configuration JSON (v1 or v2 format)."""
    if version == "v1":
        return {
            "provider": {
                "agybridge": {
                    "name": "AGY Bridge",
                    "npm": "@opencode/ai/providers/openai-compatible",
                    "options": {
                        "baseURL": base_url,
                        "apiKey": api_key,
                    },
                    "models": {
                        "gemini-3.8-flash-high": {
                            "name": "AGY Flash High",
                            "reasoning": True,
                        },
                        "gemini-3.8-flash-low": {
                            "name": "AGY Flash Low",
                            "reasoning": False,
                        },
                    },
                }
            }
        }

    return {
        "providers": [
            {
                "id": "agybridge",
                "name": "AGY Bridge",
                "type": "openai-compatible",
                "baseURL": base_url,
                "apiKey": api_key,
                "models": [
                    {
                        "id": "gemini-3.8-flash-high",
                        "name": "AGY Flash High",
                        "reasoning": True,
                    },
                    {
                        "id": "gemini-3.8-flash-low",
                        "name": "AGY Flash Low",
                        "reasoning": False,
                    },
                ],
            }
        ]
    }
