"""Pydantic request/response schemas for the API (contract per SYSTEM_DESIGN §4)."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_TEXT_LEN = 2000


# --- /classify ------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN, description="Teks review.")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Emosi: anger | happiness | sadness.")
    confidence: float = Field(..., ge=0.0, le=1.0)


# --- /insight -------------------------------------------------------------

class InsightRequest(BaseModel):
    query: str = Field(
        "keluhan utama pengguna",
        min_length=1,
        max_length=MAX_TEXT_LEN,
        description="Fokus insight (mis. 'masalah pembayaran').",
    )
    use_cache: bool = True


class Theme(BaseModel):
    theme: str
    count: int | None = None
    example_review_ids: list[str] = Field(default_factory=list)


class InsightResponse(BaseModel):
    summary: str
    themes: list[Theme] = Field(default_factory=list)
    sample_quotes: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    n_reviews: int = 0
    cached: bool = False


# --- /chat ----------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)


# --- /health --------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_db_connected: bool
    llm_provider: str
    llm_available: bool


# --- /stats ---------------------------------------------------------------

class StatsResponse(BaseModel):
    total: int
    by_emotion: dict[str, int]
    by_split: dict[str, dict[str, int]]
    negative_ratio: float
