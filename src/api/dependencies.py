"""Lazy singletons wired as FastAPI dependencies (overridable in tests)."""

from __future__ import annotations

from functools import lru_cache

import yaml

from src.api.classifier import EmotionClassifier
from src.api.config import get_settings
from src.llm.base import LLMClient
from src.llm.factory import get_llm_client
from src.monitoring.prediction_log import PredictionLogger
from src.rag.chat import ChatConfig, ChatResponder
from src.rag.embeddings import Embedder
from src.rag.insight import InsightConfig, InsightGenerator
from src.rag.vector_store import ReviewVectorStore


@lru_cache
def _rag_cfg() -> dict:
    with open(get_settings().rag_config, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_classifier() -> EmotionClassifier:
    return EmotionClassifier(model_dir=get_settings().model_dir)


@lru_cache
def get_prediction_logger() -> PredictionLogger:
    return PredictionLogger(get_settings().prediction_log_path)


@lru_cache
def get_embedder() -> Embedder:
    cfg = _rag_cfg()["embedding"]
    return Embedder(model_name=cfg["model_name"], batch_size=cfg["batch_size"])


@lru_cache
def get_vector_store() -> ReviewVectorStore:
    cfg = _rag_cfg()["vector_store"]
    return ReviewVectorStore(persist_dir=cfg["persist_dir"], collection_name=cfg["collection_name"])


@lru_cache
def get_llm() -> LLMClient:
    return get_llm_client(get_settings().llm_provider)


@lru_cache
def get_insight_generator() -> InsightGenerator:
    ic = _rag_cfg()["insight"]
    config = InsightConfig(
        negative_emotions=ic["negative_emotions"],
        top_k=ic["top_k"],
        cache_ttl_seconds=ic["cache_ttl_seconds"],
        cache_max_size=ic["cache_max_size"],
    )
    return InsightGenerator(get_vector_store(), get_embedder(), get_llm(), config)


@lru_cache
def get_chat_responder() -> ChatResponder:
    cc = _rag_cfg()["chat"]
    return ChatResponder(get_vector_store(), get_embedder(), get_llm(), ChatConfig(top_k=cc["top_k"]))
