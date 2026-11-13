"""Tests for the hosted (OpenAI-compatible) provider."""

import httpx
import pytest

from evalharness.providers import OpenAIProvider, get_provider


def _mock_client(content):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1",
        headers={"Authorization": "Bearer test-key"},
    )


def test_openai_provider_returns_model_content():
    provider = OpenAIProvider(api_key="test-key", client=_mock_client("Paris"))
    assert provider.generate("What is the capital of France?") == "Paris"
    assert provider.name == "openai:gpt-4o-mini"


def test_openai_provider_model_override():
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o", client=_mock_client("x"))
    assert provider.name == "openai:gpt-4o"


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_get_provider_parses_openai_model_suffix(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = get_provider("openai:gpt-4o")
    assert provider.name == "openai:gpt-4o"


def test_get_provider_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown provider"):
        get_provider("anthropic")
