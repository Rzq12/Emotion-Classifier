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
    note: str = Field("", description="Disclaimer: statistik dihitung dari sampel retrieval.")
    n_reviews: int = 0
    cached: bool = False


# --- /chat ----------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|bot|assistant)$")
    content: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN)
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=12,
        description="Riwayat percakapan (paling lama di depan), opsional.",
    )
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    cached: bool = False


# --- /health --------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vector_db_connected: bool
    llm_provider: str
    llm_available: bool
    telegram_configured: bool = Field(
        False,
        description="True jika TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID terbaca (bukan nilainya).",
    )


# --- /stats ---------------------------------------------------------------


class StatsResponse(BaseModel):
    total: int
    by_emotion: dict[str, int]
    by_split: dict[str, dict[str, int]]
    negative_ratio: float


# --- /agent ---------------------------------------------------------------


class AgentRunRequest(BaseModel):
    review_text: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LEN, description="Teks review yang diproses agent."
    )


class TicketResponse(BaseModel):
    """One ticket in the human-in-the-loop queue."""

    id: str
    review_text: str
    label: str = Field(..., description="Emosi: anger | happiness | sadness.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    branch: str = Field(..., description="Keputusan router: escalate | draft | archive.")
    priority: str = Field(..., description="high | normal | none.")
    draft: str = Field("", description="Draft balasan (kosong untuk branch archive).")
    grounding_ids: list[str] = Field(
        default_factory=list, description="ID review yang jadi grounding draft."
    )
    status: str = Field(..., description="pending | approved | rejected.")
    reason: str = ""
    created_at: str
    updated_at: str


class AgentRunResponse(BaseModel):
    ticket: TicketResponse
    notified: bool = Field(False, description="True jika notifikasi eskalasi terkirim.")
    notify_channel: str = Field("", description="Kanal notifikasi: telegram | file | ''.")


class QueueResponse(BaseModel):
    tickets: list[TicketResponse] = Field(default_factory=list)
    count: int = 0


class RejectRequest(BaseModel):
    reason: str = Field("", max_length=MAX_TEXT_LEN, description="Alasan penolakan draft.")


class AgentStatsResponse(BaseModel):
    total: int
    pending: int
    approved: int
    rejected: int
    escalations: int
    approval_rate: float = Field(..., ge=0.0, le=1.0)


class NotifyTestResponse(BaseModel):
    """Result of a Telegram delivery self-test (never leaks the token)."""

    telegram_configured: bool
    attempted: bool = Field(..., description="True jika sempat mencoba kirim ke Telegram.")
    ok: bool = Field(..., description="True jika Telegram membalas sukses.")
    detail: str = Field("", description="Alasan error Telegram/OS (ter-redaksi), kosong jika ok.")
