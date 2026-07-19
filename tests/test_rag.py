"""Unit tests for RAG retrieval/insight/chat pipelines using fakes (no network/models)."""

from __future__ import annotations

from src.llm.base import LLMClient
from src.rag.bm25 import BM25Index
from src.rag.chat import NO_CONTEXT_MESSAGE, ChatConfig, ChatResponder
from src.rag.insight import InsightConfig, InsightGenerator, _parse_insight_json
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import RetrievedReview


class FakeLLM(LLMClient):
    provider = "fake"
    model = "fake-model"

    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate(self, prompt, system=None, temperature=0.2, max_tokens=1024):
        self.calls += 1
        return self.response

    def is_available(self) -> bool:
        return True


class FakeRetriever:
    def __init__(self, reviews):
        self._reviews = reviews
        self.last_query = None

    def query(self, query_text, n_results=8, emotions=None):
        self.last_query = query_text
        return self._reviews[:n_results]


class FakeEmbedder:
    def encode_one(self, text):
        return [0.0, 0.1, 0.2]


class FakeStore:
    def __init__(self, reviews):
        self._reviews = reviews

    def query(self, query_embedding, n_results=8, where=None):
        return self._reviews[:n_results]


def _reviews(n, emotion="anger", score=0.8, bm25=0.0):
    return [
        RetrievedReview(
            review_id=f"r{i}", text=f"review {i}", emotion=emotion, score=score, bm25_score=bm25
        )
        for i in range(n)
    ]


# --- BM25 ------------------------------------------------------------------

def _bm25_corpus():
    index = BM25Index()
    index.fit(
        ids=["a1", "a2", "a3"],
        texts=[
            "pembayaran lewat dana selalu gagal",
            "dokter ramah dan membantu sekali",
            "aplikasi lambat saat dibuka pagi hari",
        ],
        emotions=["anger", "happiness", "sadness"],
    )
    return index


def test_bm25_exact_keyword_ranks_first():
    hits = _bm25_corpus().query("bayar pakai dana gagal", n_results=2)
    assert hits and hits[0].review_id == "a1"
    assert hits[0].score > 0


def test_bm25_emotion_filter():
    hits = _bm25_corpus().query("dana gagal", n_results=3, emotions=["happiness"])
    assert all(h.emotion == "happiness" for h in hits)


def test_bm25_no_token_overlap_returns_empty():
    assert _bm25_corpus().query("xyzabc qwerty", n_results=3) == []


# --- Hybrid retriever ------------------------------------------------------

def test_hybrid_fuses_vector_and_lexical():
    vector_hits = [
        RetrievedReview(review_id="v1", text="vector hit", emotion="anger", score=0.7),
    ]
    bm25 = _bm25_corpus()
    retriever = HybridRetriever(FakeStore(vector_hits), FakeEmbedder(), bm25)

    results = retriever.query("pembayaran dana gagal", n_results=4)
    ids = [r.review_id for r in results]
    assert "v1" in ids  # vector hit survives
    assert "a1" in ids  # lexical hit fused in
    lexical = next(r for r in results if r.review_id == "a1")
    assert lexical.bm25_score > 0


def test_hybrid_doc_in_both_rankings_gets_boosted():
    shared = RetrievedReview(
        review_id="a1", text="pembayaran lewat dana selalu gagal", emotion="anger", score=0.9
    )
    other = RetrievedReview(review_id="v2", text="lain", emotion="anger", score=0.95)
    retriever = HybridRetriever(FakeStore([other, shared]), FakeEmbedder(), _bm25_corpus())

    results = retriever.query("pembayaran dana gagal", n_results=2)
    # a1 appears in both rankings -> RRF puts it first despite lower cosine.
    assert results[0].review_id == "a1"
    assert results[0].score == 0.9
    assert results[0].bm25_score > 0


def test_hybrid_works_without_bm25():
    retriever = HybridRetriever(FakeStore(_reviews(3)), FakeEmbedder(), bm25=None)
    assert len(retriever.query("apa saja keluhan", n_results=3)) == 3


# --- JSON parsing ----------------------------------------------------------

def test_parse_clean_json():
    data = _parse_insight_json(
        '{"summary": "ok", "themes": [], "sample_quotes": [], "recommendations": []}'
    )
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


# --- Insight generator -----------------------------------------------------

def test_insight_too_few_reviews():
    gen = InsightGenerator(FakeRetriever(_reviews(2)), FakeLLM("{}"), InsightConfig())
    result = gen.generate("keluhan")
    assert result["n_reviews"] == 2
    assert result["themes"] == []
    assert "belum cukup" in result["summary"].lower()


def test_insight_normal_and_cache():
    llm = FakeLLM('{"summary": "tema checkout", "themes": [{"theme": "checkout"}]}')
    gen = InsightGenerator(FakeRetriever(_reviews(10)), llm, InsightConfig())

    first = gen.generate("checkout")
    assert first["summary"] == "tema checkout"
    assert first["cached"] is False
    assert "sampel" not in first["note"] or first["note"]  # note is present
    assert llm.calls == 1

    second = gen.generate("checkout")  # same query -> cache hit, no new LLM call
    assert second["cached"] is True
    assert llm.calls == 1


def test_insight_drops_hallucinated_example_ids():
    llm = FakeLLM(
        '{"summary": "s", "themes": [{"theme": "t", '
        '"example_review_ids": ["r0", "r1", "bukan_id_asli"]}]}'
    )
    gen = InsightGenerator(FakeRetriever(_reviews(10)), llm, InsightConfig())
    result = gen.generate("checkout")
    assert result["themes"][0]["example_review_ids"] == ["r0", "r1"]


# --- Chat responder --------------------------------------------------------

def test_chat_empty_question():
    resp = ChatResponder(FakeRetriever(_reviews(5)), FakeLLM("x"))
    assert resp.answer("  ")["answer"] == NO_CONTEXT_MESSAGE


def test_chat_no_relevant_reviews():
    # All hits below both gates -> treated as no relevant context.
    low = _reviews(5, score=0.05, bm25=0.0)
    resp = ChatResponder(
        FakeRetriever(low), FakeLLM("x"), ChatConfig(min_score=0.15, min_bm25=1.0)
    )
    out = resp.answer("apa keluhan?")
    assert out["answer"] == NO_CONTEXT_MESSAGE
    assert out["sources"] == []


def test_chat_lexical_only_hit_passes_gate():
    hits = _reviews(3, score=0.05, bm25=5.0)  # weak vector, strong lexical
    resp = ChatResponder(
        FakeRetriever(hits), FakeLLM("jawaban"), ChatConfig(min_score=0.35, min_bm25=1.0)
    )
    assert resp.answer("bayar pakai dana")["answer"] == "jawaban"


def test_chat_sources_are_cited_ids_only():
    resp = ChatResponder(
        FakeRetriever(_reviews(4, score=0.8)),
        FakeLLM("Keluhan utama di [r1] dan [r3]; id [tidak_ada] diabaikan."),
    )
    out = resp.answer("apa keluhan utama?")
    assert out["sources"] == ["r1", "r3"]


def test_chat_llm_decline_returns_no_sources():
    resp = ChatResponder(
        FakeRetriever(_reviews(3, score=0.8)),
        FakeLLM("Saya tidak menemukan review yang relevan dengan pertanyaan ini."),
    )
    assert resp.answer("pertanyaan di luar topik")["sources"] == []


def test_chat_sources_fall_back_to_all_relevant_when_uncited():
    resp = ChatResponder(FakeRetriever(_reviews(3, score=0.8)), FakeLLM("jawaban tanpa sitasi"))
    out = resp.answer("apa keluhan utama?")
    assert out["sources"] == ["r0", "r1", "r2"]


def test_chat_cache_hits_for_same_question_and_history():
    llm = FakeLLM("jawaban [r0]")
    resp = ChatResponder(FakeRetriever(_reviews(3, score=0.8)), llm)

    first = resp.answer("apa keluhan?", history=[{"role": "user", "content": "halo"}])
    assert first["cached"] is False
    assert llm.calls == 1

    second = resp.answer("apa keluhan?", history=[{"role": "user", "content": "halo"}])
    assert second["cached"] is True
    assert llm.calls == 1

    # Different history -> different cache entry.
    resp.answer("apa keluhan?", history=[{"role": "user", "content": "beda"}])
    assert llm.calls == 2


def test_chat_history_rendered_into_prompt():
    class PromptCapturingLLM(FakeLLM):
        def generate(self, prompt, system=None, temperature=0.2, max_tokens=1024):
            self.last_prompt = prompt
            return super().generate(prompt, system, temperature, max_tokens)

    llm = PromptCapturingLLM("ok")
    resp = ChatResponder(FakeRetriever(_reviews(3, score=0.8)), llm)
    resp.answer(
        "lalu contohnya apa?",
        history=[
            {"role": "user", "content": "apa keluhan soal bayar?"},
            {"role": "bot", "content": "banyak error saat transaksi."},
        ],
    )
    assert "Pengguna: apa keluhan soal bayar?" in llm.last_prompt
    assert "Asisten: banyak error saat transaksi." in llm.last_prompt
