"""Hybrid retrieval: vector similarity + BM25, fused with Reciprocal Rank Fusion.

The user query is cleaned with the same preprocessing as the indexed corpus
(slang normalization, emoji removal) so query and documents live in the same
text distribution. RRF combines the two rankings without fragile score
normalization; each hit keeps both raw scores so callers can apply relevance
gates (vector cosine and/or BM25).
"""

from __future__ import annotations

from src.data.preprocessing import clean_text
from src.rag.bm25 import BM25Index
from src.rag.embeddings import Embedder
from src.rag.vector_store import RetrievedReview, ReviewVectorStore

RRF_K = 60  # standard damping constant from the RRF paper


class HybridRetriever:
    """Query vector + lexical indexes and return one fused ranking."""

    def __init__(
        self,
        store: ReviewVectorStore,
        embedder: Embedder,
        bm25: BM25Index | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        emotions: list[str] | None = None,
    ) -> list[RetrievedReview]:
        """Return the top ``n_results`` reviews for ``query_text`` (fused ranking)."""
        cleaned = clean_text(query_text) or query_text
        candidate_pool = n_results * 2

        where = {"emotion": {"$in": emotions}} if emotions else None
        vector_hits = self.store.query(
            query_embedding=self.embedder.encode_one(cleaned),
            n_results=candidate_pool,
            where=where,
        )
        lexical_hits = (
            self.bm25.query(cleaned, n_results=candidate_pool, emotions=emotions)
            if self.bm25 is not None
            else []
        )

        # Reciprocal Rank Fusion over both rankings.
        fused: dict[str, dict] = {}
        for rank, hit in enumerate(vector_hits):
            entry = fused.setdefault(
                hit.review_id,
                {"text": hit.text, "emotion": hit.emotion, "score": 0.0, "bm25": 0.0, "rrf": 0.0},
            )
            entry["score"] = hit.score
            entry["rrf"] += 1.0 / (RRF_K + rank + 1)
        for rank, hit in enumerate(lexical_hits):
            entry = fused.setdefault(
                hit.review_id,
                {"text": hit.text, "emotion": hit.emotion, "score": 0.0, "bm25": 0.0, "rrf": 0.0},
            )
            entry["bm25"] = hit.score
            entry["rrf"] += 1.0 / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1]["rrf"], reverse=True)[:n_results]
        return [
            RetrievedReview(
                review_id=review_id,
                text=entry["text"],
                emotion=entry["emotion"],
                score=entry["score"],
                bm25_score=entry["bm25"],
            )
            for review_id, entry in ranked
        ]
