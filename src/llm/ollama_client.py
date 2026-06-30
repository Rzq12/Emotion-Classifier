"""Ollama LLM client for local/offline development (no API key, no cost)."""

from __future__ import annotations

import os

from src.llm.base import LLMClient, LLMError


class OllamaClient(LLMClient):
    """Wrapper around a local Ollama server's REST API."""

    provider = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")

    def is_available(self) -> bool:
        # Assume a configured base URL means the user intends to run Ollama.
        return bool(self.base_url)

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        import urllib.error
        import urllib.request

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        import json

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body.get("response", "")
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc
