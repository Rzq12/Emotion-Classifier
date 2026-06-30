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
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert GroqClient().is_available() is False
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    assert GroqClient().is_available() is True


def test_render_prompt_substitutes_fields():
    out = render_prompt("chat.txt", reviews="[t_1] (anger) buruk", question="kenapa?")
    assert "[t_1] (anger) buruk" in out
    assert "kenapa?" in out
