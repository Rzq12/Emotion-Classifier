"""API settings loaded from environment variables (.env supported)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173")
    return [o.strip() for o in raw.split(",") if o.strip()]


@dataclass(frozen=True)
class Settings:
    model_dir: str = os.getenv("MODEL_DIR", "artifacts/indobert")
    rag_config: str = os.getenv("RAG_CONFIG", "configs/rag.yaml")
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    cors_allow_origins: list[str] = field(default_factory=_origins)


def get_settings() -> Settings:
    return Settings()
