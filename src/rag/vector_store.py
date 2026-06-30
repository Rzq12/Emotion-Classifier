"""ChromaDB persistent vector store for review retrieval.

Stores one vector per review with metadata (``emotion``, ``split``) so insight
retrieval can filter by emotion and chat can do open similarity search. Cosine
space matches the normalized embeddings produced by :class:`~src.rag.embeddings.Embedder`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedReview:
    """A single retrieval hit."""

    review_id: str
    text: str
    emotion: str
    score: float  # cosine similarity in [0, 1]


class ReviewVectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, persist_dir: str = "chroma_db", collection_name: str = "reviews"):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Upsert reviews into the collection."""
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 8,
        where: dict | None = None,
    ) -> list[RetrievedReview]:
        """Return the top ``n_results`` reviews, optionally filtered by metadata."""
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
        return self._to_reviews(result)

    @staticmethod
    def _to_reviews(result: dict) -> list[RetrievedReview]:
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        reviews: list[RetrievedReview] = []
        for rid, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            reviews.append(
                RetrievedReview(
                    review_id=rid,
                    text=doc,
                    emotion=(meta or {}).get("emotion", "unknown"),
                    score=round(1.0 - float(dist), 4),  # cosine distance -> similarity
                )
            )
        return reviews
