"""Unit tests for LLM client factory and prompt loading."""

from __future__ import annotations

import pytest
from src.llm.factory import get_llm_client
from src.llm.gemini_client import GeminiClient
from src.llm.groq_client import GroqClient
from src.llm.ollama_client import OllamaClient
from src.llm.prompt_loader import render_prompt


def test_factory_selects_provider():
    assert isinstance(get_llm_client("groq"), GroqClient)
    assert isinstance(get_llm_client("gemini"), GeminiClient)
    assert isinstance(get_llm_client("ollama"), OllamaClient)


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_llm_client("not-a-provider")


def test_availability_depends_on_key(monkeypatch):
    _clear_groq_env(monkeypatch)  # .env may define real keys; isolate the test
    assert GroqClient().is_available() is False
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    assert GroqClient().is_available() is True


def _clear_groq_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("GROQ_API_KEY"):
            monkeypatch.delenv(name, raising=False)


def test_groq_loads_keys_from_csv(monkeypatch):
    _clear_groq_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEYS", "key-a, key-b ,key-c")
    client = GroqClient()
    assert client.is_available() is True
    assert client._api_keys == ["key-a", "key-b", "key-c"]


def test_groq_loads_numbered_keys_in_order(monkeypatch):
    _clear_groq_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY_2", "key-b")
    monkeypatch.setenv("GROQ_API_KEY_1", "key-a")
    assert GroqClient()._api_keys == ["key-a", "key-b"]


def _fake_rate_limit_error():
    import httpx
    from groq import RateLimitError

    response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
    return RateLimitError("rate limit", response=response, body=None)


def _fake_sdk_client(create_fn):
    completions = type("Completions", (), {"create": staticmethod(create_fn)})()
    chat = type("Chat", (), {"completions": completions})()
    return type("FakeGroq", (), {"chat": chat})()


def _fake_response(text: str):
    message = type("Msg", (), {"content": text})()
    choice = type("Choice", (), {"message": message})()
    return type("Resp", (), {"choices": [choice]})()


def test_groq_rotates_to_next_key_on_rate_limit(monkeypatch):
    client = GroqClient(api_key="key-1,key-2")
    used: list[int] = []

    def fake_get_client(index: int):
        def create(**kwargs):
            used.append(index)
            if index == 0:
                raise _fake_rate_limit_error()
            return _fake_response("jawaban dari key kedua")

        return _fake_sdk_client(create)

    monkeypatch.setattr(client, "_get_client", fake_get_client)
    assert client.generate("halo") == "jawaban dari key kedua"
    assert used == [0, 1]
    # Next call sticks with the working key (no pointless retry of key 1).
    assert client.generate("halo lagi") == "jawaban dari key kedua"
    assert used == [0, 1, 1]


def test_groq_all_keys_exhausted_raises_llmerror(monkeypatch):
    from src.llm.base import LLMError

    client = GroqClient(api_key="key-1,key-2,key-3")

    def fake_get_client(index: int):
        def create(**kwargs):
            raise _fake_rate_limit_error()

        return _fake_sdk_client(create)

    monkeypatch.setattr(client, "_get_client", fake_get_client)
    with pytest.raises(LLMError, match="3 Groq API key"):
        client.generate("halo")


def test_render_prompt_substitutes_fields():
    out = render_prompt(
        "chat.txt",
        history="Pengguna: halo",
        reviews="[t_1] (anger) buruk",
        question="kenapa?",
    )
    assert "[t_1] (anger) buruk" in out
    assert "kenapa?" in out
    assert "Pengguna: halo" in out
