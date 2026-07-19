"""IndoBERT emotion classifier wrapper for serving.

Loads the fine-tuned model + tokenizer lazily and exposes a simple ``predict``.
Cleaning mirrors training preprocessing so inference matches the training
distribution.
"""

from __future__ import annotations

from pathlib import Path

from src.data.labels import ID2LABEL
from src.data.preprocessing import clean_text


class ModelNotFoundError(RuntimeError):
    """Raised when the model directory is missing or incomplete."""


class EmotionClassifier:
    """Wraps a fine-tuned sequence-classification model."""

    def __init__(self, model_dir: str = "artifacts/indobert", max_length: int = 128):
        self.model_dir = model_dir
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def available(self) -> bool:
        """True if the model is loadable.

        Either a local directory with ``config.json`` exists, or ``model_dir`` is
        a Hugging Face Hub repo id (``org/name``) that transformers can download.
        """
        local = (Path(self.model_dir) / "config.json").exists()
        looks_like_repo_id = "/" in self.model_dir and not Path(self.model_dir).exists()
        return local or looks_like_repo_id

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.available():
            raise ModelNotFoundError(f"Model not found in '{self.model_dir}'.")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        except Exception as exc:  # noqa: BLE001 - HF raises many exception types here
            # A missing local dir that looks like a repo id (e.g. the default
            # "artifacts/indobert") must degrade to 503, not a raw 500.
            self._model = None
            self._tokenizer = None
            raise ModelNotFoundError(
                f"Model '{self.model_dir}' tidak bisa dimuat (path lokal tidak ada "
                f"atau repo HF tidak ditemukan): {exc}"
            ) from exc
        self._model.eval()

    def predict(self, text: str) -> dict:
        """Return ``{label, confidence}`` for a single review."""
        self._load()
        torch = self._torch

        cleaned = clean_text(text)
        inputs = self._tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        idx = int(torch.argmax(probs).item())
        return {"label": ID2LABEL[idx], "confidence": round(float(probs[idx].item()), 4)}
