"""Groq LLM client (OpenAI-compatible chat completions)."""

from __future__ import annotations

import os

from src.llm.base import LLMClient, LLMError


class GroqClient(LLMClient):
    """Wrapper around the Groq SDK."""

    provider = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._client = None

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError("GROQ_API_KEY is not set.")
            from groq import Groq

            self._client = Groq(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            raise LLMError(f"Groq call failed: {exc}") from exc
