"""Lightweight BM25 (Okapi) index for lexical retrieval — no external dependency.

Complements vector search in hybrid retrieval: exact keywords (feature names,
"dana", "ovo", error codes) that embeddings can blur are matched literally.
The corpus is small (~2.7k reviews) so an in-memory inverted index is plenty.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (assumes text is already cleaned/normalized)."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class BM25Hit:
    """One lexical retrieval hit."""

    review_id: str
    text: str
    emotion: str
    score: float


class BM25Index:
    """In-memory Okapi BM25 index over the review corpus."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._emotions: list[str] = []
        self._doc_freqs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._inverted: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self._ids)

    def fit(self, ids: list[str], texts: list[str], emotions: list[str]) -> None:
        """Index the corpus (replaces any previous fit)."""
        self._ids = list(ids)
        self._texts = list(texts)
        self._emotions = list(emotions)
        self._doc_freqs = []
        self._doc_lens = []
        self._inverted = {}

        df: Counter = Counter()
        for doc_idx, text in enumerate(self._texts):
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            for token in freqs:
                df[token] += 1
                self._inverted.setdefault(token, []).append(doc_idx)

        n = len(self._texts)
        self._avgdl = (sum(self._doc_lens) / n) if n else 0.0
        self._idf = {
            token: math.log(1 + (n - count + 0.5) / (count + 0.5)) for token, count in df.items()
        }

    def query(
        self,
        text: str,
        n_results: int = 8,
        emotions: list[str] | None = None,
    ) -> list[BM25Hit]:
        """Top ``n_results`` lexical matches, optionally filtered by emotion."""
        query_tokens = set(tokenize(text))
        if not query_tokens or not self._ids:
            return []

        allowed = set(emotions) if emotions else None
        scores: dict[int, float] = {}
        for token in query_tokens:
            idf = self._idf.get(token)
            if idf is None:
                continue
            for doc_idx in self._inverted[token]:
                if allowed is not None and self._emotions[doc_idx] not in allowed:
                    continue
                tf = self._doc_freqs[doc_idx][token]
                dl = self._doc_lens[doc_idx] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * tf * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n_results]
        return [
            BM25Hit(
                review_id=self._ids[i],
                text=self._texts[i],
                emotion=self._emotions[i],
                score=round(score, 4),
            )
            for i, score in ranked
        ]
