"""Groq LLM client (OpenAI-compatible chat completions).

Supports multiple API keys with automatic rotation: when the active key hits a
rate limit (429) or is rejected (401), the client switches to the next key and
retries, cycling through every configured key before giving up. Keys can be
provided as:

- ``GROQ_API_KEYS`` — comma-separated list (recommended for multiple keys), or
- ``GROQ_API_KEY_1`` .. ``GROQ_API_KEY_<n>`` — numbered variables, or
- ``GROQ_API_KEY`` — single key (backward compatible).
"""

from __future__ import annotations

import logging
import os
import re
import threading

from src.llm.base import LLMClient, LLMError

logger = logging.getLogger(__name__)


def _load_api_keys(explicit: str | None) -> list[str]:
    """Collect API keys from the explicit arg or environment (see module doc)."""
    if explicit:
        raw = explicit
    elif os.getenv("GROQ_API_KEYS"):
        raw = os.environ["GROQ_API_KEYS"]
    else:
        numbered = sorted(
            (k for k in os.environ if re.fullmatch(r"GROQ_API_KEY_\d+", k)),
            key=lambda k: int(k.rsplit("_", 1)[1]),
        )
        if numbered:
            raw = ",".join(os.environ[k] for k in numbered)
        else:
            raw = os.getenv("GROQ_API_KEY", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


class GroqClient(LLMClient):
    """Wrapper around the Groq SDK with multi-key failover."""

    provider = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._api_keys = _load_api_keys(api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._key_index = 0
        self._client = None
        self._client_index: int | None = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return bool(self._api_keys)

    def _get_client(self, index: int):
        from groq import Groq

        with self._lock:
            if self._client is None or self._client_index != index:
                self._client = Groq(api_key=self._api_keys[index])
                self._client_index = index
            return self._client

    def _rotate(self, from_index: int) -> None:
        """Advance to the next key (no-op if another thread already rotated)."""
        with self._lock:
            if self._key_index == from_index:
                self._key_index = (from_index + 1) % len(self._api_keys)
                logger.warning(
                    "Groq key #%d limited/rejected; switching to key #%d of %d",
                    from_index + 1,
                    self._key_index + 1,
                    len(self._api_keys),
                )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        if not self._api_keys:
            raise LLMError("No Groq API key configured (GROQ_API_KEY / GROQ_API_KEYS).")

        from groq import AuthenticationError, RateLimitError

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc: Exception | None = None
        for _ in range(len(self._api_keys)):
            with self._lock:
                index = self._key_index
            try:
                resp = self._get_client(index).chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, AuthenticationError) as exc:
                last_exc = exc
                self._rotate(index)
            except LLMError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize all provider errors
                raise LLMError(f"Groq call failed: {exc}") from exc

        raise LLMError(
            f"All {len(self._api_keys)} Groq API key(s) exhausted "
            f"(rate-limited or rejected): {last_exc}"
        ) from last_exc
