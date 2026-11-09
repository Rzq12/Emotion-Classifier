"""Unit tests for the serving-time EmotionClassifier wrapper (no real weights)."""

from __future__ import annotations

import pytest
import torch
from src.api.classifier import EmotionClassifier, ModelNotFoundError
from src.data.labels import LABEL2ID


class _FakeTokenizer:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, text, **kwargs):
        self.calls.append(text)
        return {"input_ids": torch.zeros((1, kwargs["max_length"]), dtype=torch.long)}


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    def __init__(self, winning_label: str):
        self._logits = torch.full((1, 3), -5.0)
        self._logits[0, LABEL2ID[winning_label]] = 5.0

    def __call__(self, **inputs):
        return _FakeOutput(self._logits)


def _loaded(winning_label: str = "sadness") -> tuple[EmotionClassifier, _FakeTokenizer]:
    clf = EmotionClassifier(model_dir="unused", max_length=16)
    tok = _FakeTokenizer()
    clf._tokenizer, clf._model, clf._torch = tok, _FakeModel(winning_label), torch
    return clf, tok


def test_available_local_dir_with_config_returns_true(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert EmotionClassifier(model_dir=str(tmp_path)).available() is True


def test_available_existing_dir_without_config_returns_false(tmp_path):
    assert EmotionClassifier(model_dir=str(tmp_path)).available() is False


def test_available_hub_repo_id_returns_true():
    assert EmotionClassifier(model_dir="someone/indo-emotion-indobert").available() is True


def test_available_plain_missing_name_returns_false():
    assert EmotionClassifier(model_dir="no-such-model-dir").available() is False


def test_predict_unavailable_model_raises_model_not_found():
    with pytest.raises(ModelNotFoundError):
        EmotionClassifier(model_dir="no-such-model-dir").predict("halo")


def test_predict_load_failure_raises_model_not_found_and_stays_unloaded(monkeypatch):
    import transformers

    def boom(*args, **kwargs):
        raise OSError("repo not found")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    clf = EmotionClassifier(model_dir="someone/missing-repo")
    with pytest.raises(ModelNotFoundError, match="tidak bisa dimuat"):
        clf.predict("halo")
    assert clf.is_loaded is False


def test_predict_returns_argmax_label_and_rounded_confidence():
    clf, _ = _loaded("sadness")
    result = clf.predict("Aplikasinya bikin sedih")
    assert result["label"] == "sadness"
    assert 0.99 < result["confidence"] <= 1.0
    assert result["confidence"] == round(result["confidence"], 4)


def test_predict_cleans_text_before_tokenizing():
    clf, tok = _loaded("anger")
    clf.predict("KESEL BANGET!!! https://contoh.com")
    assert "https" not in tok.calls[0]
    assert tok.calls[0] == tok.calls[0].lower()


def test_is_loaded_reflects_injected_model():
    clf, _ = _loaded()
    assert clf.is_loaded is True
    assert EmotionClassifier(model_dir="x").is_loaded is False
