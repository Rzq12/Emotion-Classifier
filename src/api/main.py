"""FastAPI application for Indo Review Intelligence.

Endpoints (SYSTEM_DESIGN §4): /classify, /insight, /chat, /health.
LLM-calling endpoints are rate limited to control cost; CORS is restricted to
configured origins. LLM failures degrade gracefully to a 503 with a clear message.

Note: this module intentionally avoids ``from __future__ import annotations`` so
FastAPI can resolve body/dependency types through the slowapi rate-limit wrapper.
"""

from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api import schemas
from src.api.classifier import EmotionClassifier, ModelNotFoundError
from src.api.config import get_settings
from src.api.dependencies import (
    get_chat_responder,
    get_classifier,
    get_insight_generator,
    get_llm,
    get_prediction_logger,
    get_vector_store,
)
from src.llm.base import LLMClient, LLMError
from src.monitoring.prediction_log import PredictionLogger
from src.rag.chat import ChatResponder
from src.rag.insight import InsightGenerator
from src.rag.vector_store import ReviewVectorStore

settings = get_settings()
limiter = Limiter(key_func=get_remote_address)
_rate = f"{settings.rate_limit_per_minute}/minute"

app = FastAPI(title="Indo Review Intelligence API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency aliases (Annotated style avoids B008 and reads cleanly).
ClassifierDep = Annotated[EmotionClassifier, Depends(get_classifier)]
StoreDep = Annotated[ReviewVectorStore, Depends(get_vector_store)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
InsightDep = Annotated[InsightGenerator, Depends(get_insight_generator)]
ChatDep = Annotated[ChatResponder, Depends(get_chat_responder)]
PredictionLoggerDep = Annotated[PredictionLogger, Depends(get_prediction_logger)]


@app.exception_handler(LLMError)
async def _llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Layanan LLM sedang tidak tersedia. Coba lagi nanti."},
    )


@app.get("/health", response_model=schemas.HealthResponse)
def health(classifier: ClassifierDep, store: StoreDep, llm: LLMDep) -> schemas.HealthResponse:
    try:
        vector_ok = store.count() >= 0
    except Exception:  # noqa: BLE001 - health must never raise
        vector_ok = False
    return schemas.HealthResponse(
        status="ok",
        model_loaded=classifier.available(),
        vector_db_connected=vector_ok,
        llm_provider=getattr(llm, "provider", "unknown"),
        llm_available=llm.is_available(),
    )


@app.get("/stats", response_model=schemas.StatsResponse)
def stats() -> schemas.StatsResponse:
    from src.api.stats import compute_stats

    return schemas.StatsResponse(**compute_stats())


@app.post("/classify", response_model=schemas.ClassifyResponse)
def classify(
    body: schemas.ClassifyRequest,
    background_tasks: BackgroundTasks,
    classifier: ClassifierDep,
    prediction_logger: PredictionLoggerDep,
) -> schemas.ClassifyResponse:
    try:
        result = classifier.predict(body.text)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model classifier belum tersedia.") from exc
    # Monitoring log runs after the response is sent (non-blocking).
    background_tasks.add_task(
        prediction_logger.log, body.text, result["label"], result["confidence"]
    )
    return schemas.ClassifyResponse(**result)


@app.post("/insight", response_model=schemas.InsightResponse)
@limiter.limit(_rate)
def insight(
    request: Request, body: schemas.InsightRequest, generator: InsightDep
) -> schemas.InsightResponse:
    result = generator.generate(body.query, use_cache=body.use_cache)
    return schemas.InsightResponse(**result)


@app.post("/chat", response_model=schemas.ChatResponse)
@limiter.limit(_rate)
def chat(request: Request, body: schemas.ChatRequest, responder: ChatDep) -> schemas.ChatResponse:
    return schemas.ChatResponse(**responder.answer(body.question))
