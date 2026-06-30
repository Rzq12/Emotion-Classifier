"""Sentence embedding wrapper (multilingual, CPU-friendly).

Lazily loads the SentenceTransformer model so importing the module is cheap and
tests can patch the encoder without downloading weights.
"""

from __future__ import annotations

from collections.abc import Sequence


class Embedder:
    """Encode text into dense vectors using a SentenceTransformer model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 64,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        model = self._get_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def encode_one(self, text: str) -> list[float]:
        """Encode a single string."""
        return self.encode([text])[0]
