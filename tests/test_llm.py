import httpx
import pytest

import app.services.llm as llm


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def use_gemini(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "test-key")


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "not-a-provider")

    with pytest.raises(llm.LLMError):
        llm.generate_answer("system", "user")


def test_gemini_without_api_key_raises(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "GEMINI_API_KEY", "")

    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.generate_answer("system", "user")


def test_gemini_response_is_parsed_and_key_stays_out_of_the_url(monkeypatch):
    use_gemini(monkeypatch)
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse(
            {"candidates": [{"content": {"parts": [{"text": "Hello "}, {"text": "world"}]}}]}
        )

    monkeypatch.setattr(llm.httpx, "post", fake_post)

    assert llm.generate_answer("system", "user") == "Hello world"
    assert captured["headers"]["x-goog-api-key"] == "test-key"
    assert "test-key" not in captured["url"]


def test_network_failure_becomes_llm_error(monkeypatch):
    use_gemini(monkeypatch)

    def broken_post(*args, **kwargs):
        raise httpx.ConnectError("no internet")

    monkeypatch.setattr(llm.httpx, "post", broken_post)

    with pytest.raises(llm.LLMError):
        llm.generate_answer("system", "user")


def test_blocked_gemini_response_becomes_llm_error(monkeypatch):
    use_gemini(monkeypatch)
    blocked = FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}})
    monkeypatch.setattr(llm.httpx, "post", lambda *args, **kwargs: blocked)

    with pytest.raises(llm.LLMError):
        llm.generate_answer("system", "user")


def test_ollama_response_is_parsed(monkeypatch):
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    reply = FakeResponse({"message": {"content": " Hi there "}})
    monkeypatch.setattr(llm.httpx, "post", lambda *args, **kwargs: reply)

    assert llm.generate_answer("system", "user") == "Hi there"