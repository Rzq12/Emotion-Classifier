"""Abstract LLM client interface.

Route handlers and RAG logic depend only on this interface, never on a concrete
provider SDK. This keeps provider switching (Groq / Gemini / Ollama) a config
change, and makes the pipeline testable with a fake client.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Raised when an LLM call fails (network, auth, rate limit, timeout)."""


class LLMClient(ABC):
    """Minimal provider-agnostic chat interface."""

    #: Human-readable provider name (e.g. "groq", "gemini").
    provider: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Return the model's text completion for ``prompt``.

        Implementations must raise :class:`LLMError` (not provider-specific
        exceptions) on failure so callers can handle errors uniformly.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the client is configured (e.g. API key present)."""
