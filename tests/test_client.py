"""Tests for client routing."""

import pytest

openai = pytest.importorskip("openai", reason="openai not installed")

from src.client import get_client  # noqa: E402


class TestGetClientRouting:
    """get_client should route ollama models to localhost."""

    def test_ollama_routes_to_localhost(self):
        client = get_client("ollama/qwen2.5:3b")
        assert "localhost:11434" in str(client.base_url)

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            get_client("meta-llama/llama-3.1-8b-instruct")
