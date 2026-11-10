"""FastAPI application for Indo Review Intelligence.

Endpoints (SYSTEM_DESIGN §4): /classify (+ /classify/batch), /insight, /chat,
/health, plus the /reviews, /monitoring and /agent routers.
LLM-calling endpoints are rate limited to control cost; CORS is restricted to
configured origins. When the LLM fails, /insight and /chat degrade to verbatim
retrieved reviews (``degraded: true``); any other LLMError becomes a clear 503.

Note: this module intentionally avoids ``from __future__ import annotations`` so
FastAPI can resolve body/dependency types through the slowapi rate-limit wrapper.
"""

import logging
from collections import Counter
from typing import Annotated

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.agent.notifier import EscalationNotifier
from src.api import schemas
from src.api.agent_routes import router as agent_router
from src.api.classifier import EmotionClassifier, ModelNotFoundError
from src.api.config import get_settings
from src.api.dependencies import (
    get_chat_responder,
    get_classifier,
    get_insight_generator,
    get_llm,
    get_notifier,
    get_prediction_logger,
    get_vector_store,
)
from src.api.limiter import RATE, limiter
from src.api.monitoring_routes import router as monitoring_router
from src.api.review_routes import router as review_router
from src.data.labels import LABELS
from src.llm.base import LLMClient, LLMError
from src.monitoring.prediction_log import PredictionLogger
from src.rag.chat import ChatResponder
from src.rag.insight import InsightGenerator
from src.rag.vector_store import ReviewVectorStore

logger = logging.getLogger(__name__)

settings = get_settings()
_rate = RATE

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
app.include_router(agent_router)
app.include_router(review_router)
app.include_router(monitoring_router)

# Dependency aliases (Annotated style avoids B008 and reads cleanly).
ClassifierDep = Annotated[EmotionClassifier, Depends(get_classifier)]
StoreDep = Annotated[ReviewVectorStore, Depends(get_vector_store)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
InsightDep = Annotated[InsightGenerator, Depends(get_insight_generator)]
ChatDep = Annotated[ChatResponder, Depends(get_chat_responder)]
PredictionLoggerDep = Annotated[PredictionLogger, Depends(get_prediction_logger)]
NotifierDep = Annotated[EscalationNotifier, Depends(get_notifier)]


@app.exception_handler(LLMError)
async def _llm_error_handler(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Layanan LLM sedang tidak tersedia. Coba lagi nanti."},
    )


@app.get("/health", response_model=schemas.HealthResponse)
def health(
    classifier: ClassifierDep, store: StoreDep, llm: LLMDep, notifier: NotifierDep
) -> schemas.HealthResponse:
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
        telegram_configured=notifier.telegram_configured,
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


@app.post(
    "/classify/batch",
    response_model=schemas.ClassifyBatchResponse,
    summary="Klasifikasi banyak review sekaligus",
)
@limiter.limit(_rate)
def classify_batch(
    request: Request,
    body: schemas.ClassifyBatchRequest,
    background_tasks: BackgroundTasks,
    classifier: ClassifierDep,
    prediction_logger: PredictionLoggerDep,
) -> schemas.ClassifyBatchResponse:
    """Classify up to ``MAX_BATCH_SIZE`` reviews; rate limited (CPU-heavy)."""
    try:
        predictions = classifier.predict_batch(body.texts)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model classifier belum tersedia.") from exc

    results = [
        schemas.ClassifyBatchItem(text=text, **prediction)
        for text, prediction in zip(body.texts, predictions, strict=True)
    ]
    for item in results:
        background_tasks.add_task(prediction_logger.log, item.text, item.label, item.confidence)

    counts = Counter(item.label for item in results)
    return schemas.ClassifyBatchResponse(
        results=results,
        count=len(results),
        by_emotion={label: counts.get(label, 0) for label in LABELS},
    )


@app.post("/insight", response_model=schemas.InsightResponse)
@limiter.limit(_rate)
def insight(
    request: Request, body: schemas.InsightRequest, generator: InsightDep
) -> schemas.InsightResponse:
    try:
        result = generator.generate(body.query, use_cache=body.use_cache, emotions=body.emotions)
    except LLMError as exc:
        # Degraded mode: real retrieved reviews instead of a 503 (PRD 6.4 v1.2).
        logger.warning("LLM unavailable for /insight, serving degraded result: %s", exc)
        result = generator.fallback(body.query, emotions=body.emotions)
    return schemas.InsightResponse(**result)


@app.post("/chat", response_model=schemas.ChatResponse)
@limiter.limit(_rate)
def chat(request: Request, body: schemas.ChatRequest, responder: ChatDep) -> schemas.ChatResponse:
    history = [m.model_dump() for m in body.history]
    try:
        result = responder.answer(body.question, history=history)
    except LLMError as exc:
        logger.warning("LLM unavailable for /chat, serving degraded answer: %s", exc)
        result = responder.fallback(body.question)
    return schemas.ChatResponse(**result)
