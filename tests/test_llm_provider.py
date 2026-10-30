import json
import urllib.error
from unittest.mock import patch

import pytest

from taxguard.llm_provider import OpenAICompatibleProvider, LLMProviderError


def test_groq_endpoint_is_built_correctly():
    p = OpenAICompatibleProvider(base_url="https://api.groq.com/openai/v1", api_key="fake", model="openai/gpt-oss-20b")
    assert p._endpoint() == "https://api.groq.com/openai/v1/chat/completions"


def test_full_chat_endpoint_is_not_duplicated():
    url = "https://api.groq.com/openai/v1/chat/completions"
    p = OpenAICompatibleProvider(base_url=url, api_key="fake", model="x")
    assert p._endpoint() == url


def test_malformed_endpoint_has_actionable_error():
    p = OpenAICompatibleProvider(base_url="https:hdskjdheh", api_key="fake", model="x")
    with pytest.raises(LLMProviderError, match="Invalid TAXGUARD_LLM_BASE_URL"):
        p._endpoint()


def test_missing_key_is_rejected_without_network():
    p = OpenAICompatibleProvider(base_url="https://api.groq.com/openai/v1", api_key="", model="x")
    with pytest.raises(LLMProviderError, match="not configured"):
        p._endpoint()


def test_success_response_extracts_content(tmp_path, monkeypatch):
    p = OpenAICompatibleProvider(base_url="https://api.groq.com/openai/v1", api_key="fake", model="openai/gpt-oss-20b")
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"choices":[{"message":{"content":"{\"answer\":\"Use a dated receipt.\"}"}}]}).encode()
    monkeypatch.setattr("taxguard.llm_provider.urllib.request.urlopen", lambda *a, **k: FakeResponse())
    result = p.answer("question", "US", "2025-08-01", "fictional evidence")
    assert "dated receipt" in result


def test_403_error_has_helpful_message(monkeypatch):
    p = OpenAICompatibleProvider(base_url="https://api.groq.com/openai/v1", api_key="fake", model="openai/gpt-oss-20b")
    error = urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, None)
    error.read = lambda: b'{"error":{"message":"invalid API key"}}'
    monkeypatch.setattr("taxguard.llm_provider.urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(error))
    with pytest.raises(LLMProviderError, match="HTTP 403.*invalid API key"):
        p.answer("question", "US", "2025-08-01", "fictional evidence")
