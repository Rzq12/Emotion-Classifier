"""API contract tests for the v1.2 endpoints (dependencies mocked)."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.agent.queue_store import QueueStore
from src.api.classifier import ModelNotFoundError
from src.api.dependencies import (
    get_chat_responder,
    get_classifier,
    get_drift_reference,
    get_insight_generator,
    get_prediction_logger,
    get_prediction_records,
    get_queue_store,
    get_review_catalog,
)
from src.api.limiter import limiter
from src.api.main import app
from src.api.schemas import MAX_BATCH_SIZE
from src.api.security import get_reviewer_key
from src.llm.base import LLMError

REVIEWER_KEY = "test-reviewer-key"
AUTH = {"X-API-Key": REVIEWER_KEY}
PROBS = {"anger": 0.9, "happiness": 0.04, "sadness": 0.06}


class FakeClassifier:
    def __init__(self, missing: bool = False):
        self.missing = missing

    def _one(self, text: str) -> dict:
        label = "anger" if "error" in text else "happiness"
        return {"label": label, "confidence": 0.9, "probabilities": PROBS}

    def predict(self, text: str) -> dict:
        return self._one(text)

    def predict_batch(self, texts: list[str]) -> list[dict]:
        if self.missing:
            raise ModelNotFoundError("no model")
        return [self._one(t) for t in texts]


class FakePredictionLogger:
    def __init__(self):
        self.records = []

    def log(self, text: str, label: str, confidence: float) -> None:
        self.records.append((text, label, confidence))


class FailingInsight:
    """LLM is down: ``generate`` raises, ``fallback`` serves retrieved reviews."""

    def __init__(self):
        self.seen_emotions = None

    def generate(self, query: str, use_cache: bool = True, emotions=None) -> dict:
        self.seen_emotions = emotions
        raise LLMError("down")

    def fallback(self, query: str, emotions=None) -> dict:
        return {
            "summary": "LLM tidak tersedia.",
            "themes": [],
            "sample_quotes": ["checkout selalu gagal"],
            "recommendations": [],
            "note": "Mode terdegradasi.",
            "cached": False,
            "n_reviews": 1,
            "emotions": emotions or ["anger", "sadness"],
            "degraded": True,
        }


class FailingChat:
    def answer(self, question: str, history: list | None = None) -> dict:
        raise LLMError("down")

    def fallback(self, question: str) -> dict:
        return {
            "answer": "LLM tidak tersedia.\n- checkout gagal [train_1]",
            "sources": ["train_1"],
            "cached": False,
            "degraded": True,
        }


def _records(n: int, label: str) -> list[dict]:
    return [
        {
            "timestamp": f"2026-11-01T10:00:{i % 60:02d}+00:00",
            "text": f"review {i}",
            "label": label,
            "confidence": 0.9,
        }
        for i in range(n)
    ]


@pytest.fixture()
def ctx(tmp_path):
    """Test client plus the fakes behind it, for assertions on side effects."""
    state = {
        "classifier": FakeClassifier(),
        "logger": FakePredictionLogger(),
        "insight": FailingInsight(),
        "queue": QueueStore(tmp_path / "queue.db"),
        "records": [],
    }
    reference = pd.DataFrame(
        {"text": [f"review {i}" for i in range(50)], "label": ["happiness"] * 50}
    )
    app.dependency_overrides[get_classifier] = lambda: state["classifier"]
    app.dependency_overrides[get_prediction_logger] = lambda: state["logger"]
    app.dependency_overrides[get_insight_generator] = lambda: state["insight"]
    app.dependency_overrides[get_chat_responder] = lambda: FailingChat()
    app.dependency_overrides[get_queue_store] = lambda: state["queue"]
    app.dependency_overrides[get_reviewer_key] = lambda: REVIEWER_KEY
    app.dependency_overrides[get_prediction_records] = lambda: state["records"]
    app.dependency_overrides[get_drift_reference] = lambda: reference
    app.dependency_overrides[get_review_catalog] = lambda: {
        "train_1": {"text": "checkout selalu gagal", "emotion": "anger"},
        "val_2": {"text": "dokter ramah", "emotion": "happiness"},
    }
    limiter.reset()
    state["client"] = TestClient(app)
    yield state
    app.dependency_overrides.clear()


# --- /classify probabilities + /classify/batch ------------------------------


def test_classify_response_includes_probabilities(ctx):
    r = ctx["client"].post("/classify", json={"text": "error terus"})

    assert r.status_code == 200
    assert r.json()["probabilities"] == PROBS


def test_classify_batch_valid_texts_returns_results_and_summary(ctx):
    r = ctx["client"].post("/classify/batch", json={"texts": ["error bayar", "dokter ramah"]})

    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert [item["label"] for item in body["results"]] == ["anger", "happiness"]
    assert body["results"][0]["text"] == "error bayar"
    assert body["by_emotion"] == {"anger": 1, "happiness": 1, "sadness": 0}


def test_classify_batch_logs_every_prediction(ctx):
    ctx["client"].post("/classify/batch", json={"texts": ["error bayar", "dokter ramah"]})

    assert ctx["logger"].records == [
        ("error bayar", "anger", 0.9),
        ("dokter ramah", "happiness", 0.9),
    ]


@pytest.mark.parametrize(
    "texts",
    [[], [""], ["ok"] * (MAX_BATCH_SIZE + 1)],
    ids=["empty-list", "blank-item", "too-many"],
)
def test_classify_batch_invalid_payload_returns_422(ctx, texts):
    assert ctx["client"].post("/classify/batch", json={"texts": texts}).status_code == 422


def test_classify_batch_missing_model_returns_503(ctx):
    ctx["classifier"] = FakeClassifier(missing=True)

    assert ctx["client"].post("/classify/batch", json={"texts": ["x"]}).status_code == 503


# --- /reviews/lookup --------------------------------------------------------


def test_reviews_lookup_returns_found_reviews_and_missing_ids(ctx):
    r = ctx["client"].post(
        "/reviews/lookup", json={"ids": ["val_2", "nope", "train_1", "val_2"]}
    )

    assert r.status_code == 200
    body = r.json()
    assert [rev["review_id"] for rev in body["reviews"]] == ["val_2", "train_1"]
    assert body["reviews"][1] == {
        "review_id": "train_1",
        "text": "checkout selalu gagal",
        "emotion": "anger",
    }
    assert body["missing"] == ["nope"]


def test_reviews_lookup_empty_ids_returns_422(ctx):
    assert ctx["client"].post("/reviews/lookup", json={"ids": []}).status_code == 422


# --- /monitoring/summary ----------------------------------------------------


def test_monitoring_summary_no_predictions_reports_insufficient_data(ctx):
    r = ctx["client"].get("/monitoring/summary")

    assert r.status_code == 200
    body = r.json()
    assert body["n_predictions"] == 0
    assert body["drift"]["status"] == "insufficient_data"


def test_monitoring_summary_shifted_predictions_reports_drift(ctx):
    ctx["records"] = _records(40, "anger")

    body = ctx["client"].get("/monitoring/summary").json()

    assert body["n_predictions"] == 40
    assert body["drift"]["status"] == "ok"
    assert body["drift"]["label_verdict"] == "DRIFT"
    assert body["daily"] == [{"date": "2026-11-01", "count": 40}]


# --- Degraded mode (LLM down) -----------------------------------------------


def test_insight_llm_error_returns_degraded_result_not_503(ctx):
    r = ctx["client"].post("/insight", json={"query": "pembayaran"})

    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["sample_quotes"] == ["checkout selalu gagal"]


def test_insight_emotion_filter_is_forwarded(ctx):
    r = ctx["client"].post("/insight", json={"query": "pembayaran", "emotions": ["anger"]})

    assert r.status_code == 200
    assert ctx["insight"].seen_emotions == ["anger"]
    assert r.json()["emotions"] == ["anger"]


def test_insight_non_negative_emotion_filter_returns_422(ctx):
    r = ctx["client"].post("/insight", json={"query": "x", "emotions": ["happiness"]})

    assert r.status_code == 422


def test_chat_llm_error_returns_degraded_answer_not_503(ctx):
    r = ctx["client"].post("/chat", json={"question": "apa keluhan utama?"})

    assert r.status_code == 200
    body = r.json()
    assert body["degraded"] is True
    assert body["sources"] == ["train_1"]


# --- /agent/edit ------------------------------------------------------------


def _ticket_id(ctx) -> str:
    return ctx["queue"].add("error terus", "anger", 0.9, "escalate", "high", "draft awal").id


def test_agent_edit_pending_ticket_updates_draft(ctx):
    tid = _ticket_id(ctx)

    r = ctx["client"].post(f"/agent/edit/{tid}", json={"draft": "  draft baru  "}, headers=AUTH)

    assert r.status_code == 200
    assert r.json()["draft"] == "draft baru"
    assert r.json()["status"] == "pending"


def test_agent_edit_without_key_returns_401_and_keeps_draft(ctx):
    tid = _ticket_id(ctx)

    r = ctx["client"].post(f"/agent/edit/{tid}", json={"draft": "x"})

    assert r.status_code == 401
    assert ctx["queue"].get(tid).draft == "draft awal"


def test_agent_edit_resolved_ticket_returns_409(ctx):
    tid = _ticket_id(ctx)
    ctx["queue"].approve(tid)

    r = ctx["client"].post(f"/agent/edit/{tid}", json={"draft": "x"}, headers=AUTH)

    assert r.status_code == 409


def test_agent_edit_missing_ticket_returns_404(ctx):
    r = ctx["client"].post("/agent/edit/nope", json={"draft": "x"}, headers=AUTH)

    assert r.status_code == 404


def test_agent_edit_whitespace_draft_returns_422(ctx):
    tid = _ticket_id(ctx)

    r = ctx["client"].post(f"/agent/edit/{tid}", json={"draft": "   "}, headers=AUTH)

    assert r.status_code == 422
