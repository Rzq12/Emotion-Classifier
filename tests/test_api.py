"""API contract tests using FastAPI TestClient with mocked dependencies."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.api.dependencies import (
    get_chat_responder,
    get_classifier,
    get_insight_generator,
    get_llm,
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
    def answer(self, question: str) -> dict:
        return {"answer": "Keluhan utama soal pembayaran.", "sources": ["train_1", "val_2"]}


@pytest.fixture()
def client():
    app.dependency_overrides[get_classifier] = lambda: FakeClassifier()
    app.dependency_overrides[get_vector_store] = lambda: FakeStore()
    app.dependency_overrides[get_llm] = lambda: FakeLLM()
    app.dependency_overrides[get_insight_generator] = lambda: FakeInsight()
    app.dependency_overrides[get_chat_responder] = lambda: FakeChat()
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
