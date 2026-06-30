"""Select an LLM client based on the ``LLM_PROVIDER`` environment variable."""

from __future__ import annotations

import os

from src.llm.base import LLMClient
from src.llm.gemini_client import GeminiClient
from src.llm.groq_client import GroqClient
from src.llm.ollama_client import OllamaClient

_PROVIDERS: dict[str, type[LLMClient]] = {
    "groq": GroqClient,
    "gemini": GeminiClient,
    "ollama": OllamaClient,
}


def get_llm_client(provider: str | None = None) -> LLMClient:
    """Instantiate the configured LLM client.

    Provider resolution order: explicit arg -> ``LLM_PROVIDER`` env -> ``groq``.
    """
    name = (provider or os.getenv("LLM_PROVIDER", "groq")).strip().lower()
    if name not in _PROVIDERS:
        valid = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Valid options: {valid}.")
    return _PROVIDERS[name]()
