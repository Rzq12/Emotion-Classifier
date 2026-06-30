"""Google Gemini LLM client (google-genai SDK)."""

from __future__ import annotations

import os

from src.llm.base import LLMClient, LLMError


class GeminiClient(LLMClient):
    """Wrapper around the google-genai SDK."""

    provider = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._client = None

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError("GEMINI_API_KEY is not set.")
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        from google.genai import types

        try:
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            resp = self._get_client().models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return resp.text or ""
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize all provider errors
            raise LLMError(f"Gemini call failed: {exc}") from exc
