"""Lazy singletons wired as FastAPI dependencies (overridable in tests)."""

from __future__ import annotations

from functools import lru_cache

import yaml

from src.agent.config import AgentConfig
from src.agent.graph import AgentRuntime
from src.agent.nodes import AgentDeps
from src.agent.notifier import EscalationNotifier
from src.agent.queue_store import QueueStore
from src.api.classifier import EmotionClassifier
from src.api.config import get_settings
from src.llm.base import LLMClient
from src.llm.factory import get_llm_client
from src.monitoring.prediction_log import PredictionLogger
from src.rag.bm25 import BM25Index
from src.rag.chat import ChatConfig, ChatResponder
from src.rag.corpus import load_reviews
from src.rag.embeddings import Embedder
from src.rag.insight import InsightConfig, InsightGenerator
from src.rag.retriever import HybridRetriever
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
def get_bm25() -> BM25Index:
    reviews = load_reviews(_rag_cfg()["data"]["csvs"])
    index = BM25Index()
    index.fit(
        ids=reviews["review_id"].tolist(),
        texts=reviews["text"].tolist(),
        emotions=reviews["label"].tolist(),
    )
    return index


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(get_vector_store(), get_embedder(), get_bm25())


@lru_cache
def get_insight_generator() -> InsightGenerator:
    ic = _rag_cfg()["insight"]
    config = InsightConfig(
        negative_emotions=ic["negative_emotions"],
        top_k=ic["top_k"],
        cache_ttl_seconds=ic["cache_ttl_seconds"],
        cache_max_size=ic["cache_max_size"],
    )
    return InsightGenerator(get_retriever(), get_llm(), config)


@lru_cache
def get_chat_responder() -> ChatResponder:
    cc = _rag_cfg()["chat"]
    config = ChatConfig(
        top_k=cc["top_k"],
        min_score=cc.get("min_score", 0.35),
        min_bm25=cc.get("min_bm25", 1.0),
        cache_ttl_seconds=cc.get("cache_ttl_seconds", 1800),
        cache_max_size=cc.get("cache_max_size", 256),
    )
    return ChatResponder(get_retriever(), get_llm(), config)


# --- Agentic layer (Fase 6) -----------------------------------------------


@lru_cache
def get_agent_config() -> AgentConfig:
    return AgentConfig.load(get_settings().agent_config)


@lru_cache
def get_queue_store() -> QueueStore:
    return QueueStore(get_settings().agent_queue_db)


@lru_cache
def get_notifier() -> EscalationNotifier:
    cfg = get_agent_config().notifier
    return EscalationNotifier(fallback_log_path=cfg.fallback_log_path)


@lru_cache
def get_agent_runtime() -> AgentRuntime:
    cfg = get_agent_config()
    deps = AgentDeps(
        classifier=get_classifier(),
        retriever=get_retriever(),
        llm=get_llm(),
        draft_config=cfg.draft,
    )
    return AgentRuntime(deps, cfg)
