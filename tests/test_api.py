"""API contract tests using FastAPI TestClient with mocked dependencies."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.dependencies import (
    get_chat_responder,
    get_classifier,
    get_insight_generator,
    get_llm,
    get_prediction_logger,
    get_vector_store,
)
from src.api.main import app


class FakeClassifier:
    def available(self) -> bool:
        return True

    def predict(self, text: str) -> dict:
        return {"label": "anger", "confidence": 0.91}


class FakeStore:
    def count(self) -> int:
        return 2664


class FakeLLM:
    provider = "fake"

    def is_available(self) -> bool:
        return True


class FakeInsight:
    def generate(self, query: str, use_cache: bool = True) -> dict:
        return {
            "summary": "Keluhan dominan soal checkout.",
            "themes": [{"theme": "checkout error", "count": 3, "example_review_ids": ["train_1"]}],
            "sample_quotes": ["checkout selalu gagal"],
            "recommendations": ["audit flow pembayaran"],
            "n_reviews": 12,
            "cached": False,
        }


class FakeChat:
    def __init__(self):
        self.last_history = None

    def answer(self, question: str, history: list | None = None) -> dict:
        self.last_history = history
        return {
            "answer": "Keluhan utama soal pembayaran.",
            "sources": ["train_1", "val_2"],
            "cached": False,
        }


class FakePredictionLogger:
    def __init__(self):
        self.records = []

    def log(self, text: str, label: str, confidence: float) -> None:
        self.records.append({"text": text, "label": label, "confidence": confidence})


@pytest.fixture()
def prediction_logger():
    return FakePredictionLogger()


@pytest.fixture()
def fake_chat():
    return FakeChat()


@pytest.fixture()
def client(prediction_logger, fake_chat):
    app.dependency_overrides[get_classifier] = lambda: FakeClassifier()
    app.dependency_overrides[get_vector_store] = lambda: FakeStore()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    app.dependency_overrides[get_insight_generator] = lambda: FakeInsight()
    app.dependency_overrides[get_chat_responder] = lambda: fake_chat
    app.dependency_overrides[get_prediction_logger] = lambda: prediction_logger
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["vector_db_connected"] is True
    assert body["llm_provider"] == "fake"


def test_classify_ok(client):
    r = client.post("/classify", json={"text": "aplikasi error terus"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "anger"
    assert 0.0 <= body["confidence"] <= 1.0


def test_classify_logs_prediction(client, prediction_logger):
    r = client.post("/classify", json={"text": "aplikasi error terus"})
    assert r.status_code == 200
    assert prediction_logger.records == [
        {"text": "aplikasi error terus", "label": "anger", "confidence": 0.91}
    ]


def test_classify_empty_text_422(client):
    r = client.post("/classify", json={"text": ""})
    assert r.status_code == 422


def test_classify_missing_field_422(client):
    r = client.post("/classify", json={})
    assert r.status_code == 422


def test_insight_ok(client):
    r = client.post("/insight", json={"query": "pembayaran"})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]
    assert body["themes"][0]["theme"] == "checkout error"
    assert body["n_reviews"] == 12


def test_chat_ok(client):
    r = client.post("/chat", json={"question": "apa keluhan utama?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert body["sources"] == ["train_1", "val_2"]


def test_chat_empty_question_422(client):
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422


def test_chat_history_passed_to_responder(client, fake_chat):
    r = client.post(
        "/chat",
        json={
            "question": "lalu contohnya?",
            "history": [
                {"role": "user", "content": "apa keluhan?"},
                {"role": "bot", "content": "soal bayar."},
            ],
        },
    )
    assert r.status_code == 200
    assert fake_chat.last_history == [
        {"role": "user", "content": "apa keluhan?"},
        {"role": "bot", "content": "soal bayar."},
    ]


def test_chat_invalid_history_role_422(client):
    r = client.post(
        "/chat",
        json={"question": "x", "history": [{"role": "sistem", "content": "y"}]},
    )
    assert r.status_code == 422


def test_stats_ok(client):
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    assert set(body["by_emotion"].keys()) == {"anger", "happiness", "sadness"}
    assert set(body["by_split"].keys()) == {"train", "val", "test"}
    assert 0.0 <= body["negative_ratio"] <= 1.0
