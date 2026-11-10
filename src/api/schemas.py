"""Pydantic request/response schemas for the API (contract per SYSTEM_DESIGN §4)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

MAX_TEXT_LEN = 2000
MAX_BATCH_SIZE = 100
MAX_LOOKUP_IDS = 50

ReviewText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LEN)]


# --- /classify ------------------------------------------------------------


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LEN, description="Teks review.")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Emosi: anger | happiness | sadness.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float] = Field(
        default_factory=dict, description="Probabilitas tiap kelas emosi."
    )


class ClassifyBatchRequest(BaseModel):
    texts: list[ReviewText] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE, description="Daftar teks review."
    )


class ClassifyBatchItem(ClassifyResponse):
    text: str


class ClassifyBatchResponse(BaseModel):
    results: list[ClassifyBatchItem] = Field(default_factory=list)
    count: int = 0
    by_emotion: dict[str, int] = Field(
        default_factory=dict, description="Jumlah prediksi per emosi dalam batch ini."
    )


# --- /reviews -------------------------------------------------------------


class ReviewLookupRequest(BaseModel):
    ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        ..., min_length=1, max_length=MAX_LOOKUP_IDS, description="ID review yang dicari."
    )


class ReviewItem(BaseModel):
    review_id: str
    text: str
    emotion: str


class ReviewLookupResponse(BaseModel):
    reviews: list[ReviewItem] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list, description="ID yang tidak ditemukan.")


# --- /insight -------------------------------------------------------------


class InsightRequest(BaseModel):
    query: str = Field(
        "keluhan utama pengguna",
        min_length=1,
        max_length=MAX_TEXT_LEN,
        description="Fokus insight (mis. 'masalah pembayaran').",
    )
    use_cache: bool = True
    emotions: list[Literal["anger", "sadness"]] | None = Field(
        None,
        min_length=1,
        max_length=2,
        description="Filter kategori emosi negatif. Kosong = semua emosi negatif.",
    )


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
    emotions: list[str] = Field(
        default_factory=list, description="Emosi yang dipakai sebagai filter retrieval."
    )
    degraded: bool = Field(
        False, description="True jika LLM gagal dan hasil hanya berisi kutipan retrieval."
    )


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
    degraded: bool = Field(
        False, description="True jika LLM gagal dan jawaban hanya berisi review relevan."
    )


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


# --- /monitoring ----------------------------------------------------------


class DailyCount(BaseModel):
    date: str = Field(..., description="Tanggal UTC (YYYY-MM-DD).")
    count: int


class DriftStatus(BaseModel):
    status: str = Field(..., description="ok | insufficient_data | no_reference.")
    min_samples: int
    label_psi: float | None = None
    label_verdict: str | None = Field(None, description="STABLE | MODERATE | DRIFT.")
    length_psi: float | None = None
    length_verdict: str | None = None


class MonitoringSummaryResponse(BaseModel):
    n_predictions: int
    label_counts: dict[str, int]
    label_distribution: dict[str, float]
    reference_distribution: dict[str, float] = Field(default_factory=dict)
    mean_confidence: float | None = None
    low_confidence_share: float | None = None
    low_confidence_threshold: float
    daily: list[DailyCount] = Field(default_factory=list)
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    drift: DriftStatus


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


class EditDraftRequest(BaseModel):
    draft: str = Field(
        ..., min_length=1, max_length=MAX_TEXT_LEN, description="Draft balasan hasil edit."
    )


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
