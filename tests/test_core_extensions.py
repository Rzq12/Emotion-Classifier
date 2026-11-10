"""Unit tests for v1.2 core additions: batch classify, degraded RAG, draft edit."""

from __future__ import annotations

import pytest
import torch
from src.agent.queue_store import InvalidTransitionError, QueueStore, TicketNotFoundError
from src.api.classifier import EmotionClassifier
from src.data.labels import LABEL2ID, LABELS
from src.llm.base import LLMClient, LLMError
from src.rag.chat import NO_CONTEXT_MESSAGE, ChatConfig, ChatResponder
from src.rag.insight import FALLBACK_QUOTES, InsightConfig, InsightGenerator
from src.rag.vector_store import RetrievedReview

# --- Classifier -------------------------------------------------------------


class _BatchTokenizer:
    def __init__(self):
        self.calls: list = []

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        n = len(text) if isinstance(text, list) else 1
        return {"input_ids": torch.zeros((n, kwargs["max_length"]), dtype=torch.long)}


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _BatchModel:
    def __init__(self, winning_label: str):
        self._row = torch.full((1, 3), -5.0)
        self._row[0, LABEL2ID[winning_label]] = 5.0

    def __call__(self, **inputs):
        return _Output(self._row.repeat(inputs["input_ids"].shape[0], 1))


def _loaded(winning_label: str = "anger") -> tuple[EmotionClassifier, _BatchTokenizer]:
    clf = EmotionClassifier(model_dir="unused", max_length=16)
    tok = _BatchTokenizer()
    clf._tokenizer, clf._model, clf._torch = tok, _BatchModel(winning_label), torch
    return clf, tok


def test_predict_returns_probabilities_for_every_label():
    clf, _ = _loaded("sadness")

    result = clf.predict("sedih sekali")

    assert set(result["probabilities"]) == set(LABELS)
    assert result["probabilities"]["sadness"] == result["confidence"]
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predict_batch_returns_one_result_per_text_in_chunks():
    clf, tok = _loaded("anger")

    results = clf.predict_batch([f"review {i}" for i in range(5)], batch_size=2)

    assert [r["label"] for r in results] == ["anger"] * 5
    assert [len(call) for call in tok.calls] == [2, 2, 1]


def test_predict_batch_cleans_each_text():
    clf, tok = _loaded()

    clf.predict_batch(["KESEL!!! https://contoh.com"])

    assert "https" not in tok.calls[0][0]
    assert tok.calls[0][0] == tok.calls[0][0].lower()


# --- RAG fakes --------------------------------------------------------------


class _LLM(LLMClient):
    provider = "fake"
    model = "fake-model"

    def __init__(self, response: str = "{}", fail: bool = False):
        self.response, self.fail, self.calls = response, fail, 0

    def generate(self, prompt, system=None, temperature=0.2, max_tokens=1024):
        self.calls += 1
        if self.fail:
            raise LLMError("down")
        return self.response

    def is_available(self) -> bool:
        return not self.fail


class _Retriever:
    def __init__(self, reviews):
        self._reviews = reviews
        self.emotions_seen: list = []

    def query(self, query_text, n_results=8, emotions=None):
        self.emotions_seen.append(emotions)
        return self._reviews[:n_results]


def _reviews(n: int, score: float = 0.8) -> list[RetrievedReview]:
    return [
        RetrievedReview(
            review_id=f"r{i}", text=f"review {i}", emotion="anger", score=score, bm25_score=0.0
        )
        for i in range(n)
    ]


# --- Insight ----------------------------------------------------------------


def test_insight_generate_emotion_filter_is_passed_to_retriever():
    retriever = _Retriever(_reviews(10))
    gen = InsightGenerator(retriever, _LLM('{"summary": "s"}'), InsightConfig())

    result = gen.generate("checkout", emotions=["anger"])

    assert retriever.emotions_seen == [["anger"]]
    assert result["emotions"] == ["anger"]
    assert result["degraded"] is False


def test_insight_generate_non_negative_emotion_falls_back_to_all_negative():
    retriever = _Retriever(_reviews(10))
    gen = InsightGenerator(retriever, _LLM('{"summary": "s"}'), InsightConfig())

    gen.generate("checkout", emotions=["happiness"])

    assert retriever.emotions_seen == [["anger", "sadness"]]


def test_insight_generate_caches_per_emotion_filter():
    llm = _LLM('{"summary": "s"}')
    gen = InsightGenerator(_Retriever(_reviews(10)), llm, InsightConfig())

    gen.generate("checkout", emotions=["anger"])
    gen.generate("checkout", emotions=["sadness"])
    assert llm.calls == 2  # different filter -> different cache entry

    assert gen.generate("checkout", emotions=["anger"])["cached"] is True
    assert llm.calls == 2


def test_insight_fallback_returns_verbatim_quotes_without_llm():
    llm = _LLM(fail=True)
    gen = InsightGenerator(_Retriever(_reviews(10)), llm, InsightConfig())

    result = gen.fallback("checkout", emotions=["sadness"])

    assert result["degraded"] is True
    assert result["sample_quotes"] == [f"review {i}" for i in range(FALLBACK_QUOTES)]
    assert result["themes"] == [] and result["recommendations"] == []
    assert result["emotions"] == ["sadness"]
    assert llm.calls == 0


def test_insight_fallback_without_reviews_says_so():
    result = InsightGenerator(_Retriever([]), _LLM(fail=True), InsightConfig()).fallback("x")

    assert result["sample_quotes"] == []
    assert "tidak ada review" in result["summary"]


# --- Chat -------------------------------------------------------------------


def test_chat_fallback_lists_relevant_reviews_with_sources():
    resp = ChatResponder(_Retriever(_reviews(5)), _LLM(fail=True), ChatConfig())

    out = resp.fallback("apa keluhan utama?")

    assert out["degraded"] is True
    assert out["sources"] == ["r0", "r1", "r2"]
    assert "review 0 [r0]" in out["answer"]


def test_chat_fallback_no_relevant_reviews_returns_no_context():
    resp = ChatResponder(_Retriever(_reviews(3, score=0.1)), _LLM(fail=True), ChatConfig())

    out = resp.fallback("di luar topik")

    assert out["answer"] == NO_CONTEXT_MESSAGE
    assert out["sources"] == []


# --- Queue ------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    return QueueStore(tmp_path / "queue.db")


def _ticket(store: QueueStore):
    return store.add("error terus", "anger", 0.9, "escalate", "high", "draft awal", ["r1"])


def test_update_draft_pending_ticket_replaces_draft(store):
    ticket = _ticket(store)

    updated = store.update_draft(ticket.id, "draft hasil edit")

    assert updated.draft == "draft hasil edit"
    assert updated.status == "pending"
    assert store.get(ticket.id).draft == "draft hasil edit"


def test_update_draft_resolved_ticket_raises(store):
    ticket = _ticket(store)
    store.approve(ticket.id)

    with pytest.raises(InvalidTransitionError):
        store.update_draft(ticket.id, "terlambat")


def test_update_draft_missing_ticket_raises(store):
    with pytest.raises(TicketNotFoundError):
        store.update_draft("nope", "x")
