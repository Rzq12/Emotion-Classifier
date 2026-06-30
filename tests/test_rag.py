"""Unit tests for RAG insight/chat pipelines using fakes (no network/models)."""

from __future__ import annotations

from src.llm.base import LLMClient
from src.rag.chat import NO_CONTEXT_MESSAGE, ChatConfig, ChatResponder
from src.rag.insight import InsightConfig, InsightGenerator, _parse_insight_json
from src.rag.vector_store import RetrievedReview


class FakeLLM(LLMClient):
    provider = "fake"

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate(self, prompt, system=None, temperature=0.2, max_tokens=1024):
        self.calls += 1
        return self.response

    def is_available(self) -> bool:
        return True


class FakeEmbedder:
    def encode_one(self, text):
        return [0.0, 0.1, 0.2]


class FakeStore:
    def __init__(self, reviews):
        self._reviews = reviews

    def query(self, query_embedding, n_results=8, where=None):
        return self._reviews[:n_results]


def _reviews(n, emotion="anger", score=0.8):
    return [
        RetrievedReview(review_id=f"r{i}", text=f"review {i}", emotion=emotion, score=score)
        for i in range(n)
    ]


# --- JSON parsing ---------------------------------------------------------

def test_parse_clean_json():
    data = _parse_insight_json('{"summary": "ok", "themes": [], "sample_quotes": [], "recommendations": []}')
    assert data["summary"] == "ok"


def test_parse_fenced_json():
    raw = '```json\n{"summary": "fenced"}\n```'
    data = _parse_insight_json(raw)
    assert data["summary"] == "fenced"
    # Missing keys are filled with defaults.
    assert data["themes"] == []


def test_parse_invalid_json_falls_back():
    data = _parse_insight_json("totally not json")
    assert "totally not json" in data["summary"]
    assert data["themes"] == []


# --- Insight generator ----------------------------------------------------

def test_insight_too_few_reviews():
    gen = InsightGenerator(FakeStore(_reviews(2)), FakeEmbedder(), FakeLLM("{}"), InsightConfig())
    result = gen.generate("keluhan")
    assert result["n_reviews"] == 2
    assert result["themes"] == []
    assert "belum cukup" in result["summary"].lower()


def test_insight_normal_and_cache():
    llm = FakeLLM('{"summary": "tema checkout", "themes": [{"theme": "checkout"}]}')
    gen = InsightGenerator(FakeStore(_reviews(10)), FakeEmbedder(), llm, InsightConfig())

    first = gen.generate("checkout")
    assert first["summary"] == "tema checkout"
    assert first["cached"] is False
    assert llm.calls == 1

    second = gen.generate("checkout")  # same query -> cache hit, no new LLM call
    assert second["cached"] is True
    assert llm.calls == 1


# --- Chat responder -------------------------------------------------------

def test_chat_empty_question():
    resp = ChatResponder(FakeStore(_reviews(5)), FakeEmbedder(), FakeLLM("x"))
    assert resp.answer("  ")["answer"] == NO_CONTEXT_MESSAGE


def test_chat_no_relevant_reviews():
    # All hits below min_score -> treated as no relevant context.
    low = _reviews(5, score=0.05)
    resp = ChatResponder(FakeStore(low), FakeEmbedder(), FakeLLM("x"), ChatConfig(min_score=0.15))
    out = resp.answer("apa keluhan?")
    assert out["answer"] == NO_CONTEXT_MESSAGE
    assert out["sources"] == []


def test_chat_grounded_answer_with_sources():
    resp = ChatResponder(FakeStore(_reviews(3, score=0.8)), FakeEmbedder(), FakeLLM("jawaban grounded"))
    out = resp.answer("apa keluhan utama?")
    assert out["answer"] == "jawaban grounded"
    assert out["sources"] == ["r0", "r1", "r2"]
