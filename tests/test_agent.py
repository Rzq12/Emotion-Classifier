"""Tests for the agentic layer (Fase 6): router, queue store, graph, notifier, API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from src.agent.config import AgentConfig, DraftConfig
from src.agent.graph import AgentRuntime
from src.agent.nodes import _FALLBACK_DRAFT, AgentDeps
from src.agent.notifier import EscalationNotifier, format_escalation
from src.agent.queue_store import (
    InvalidTransitionError,
    QueueStore,
    TicketNotFoundError,
)
from src.agent.router import RouterConfig, route_by_emotion
from src.llm.base import LLMError

# --- Fakes ----------------------------------------------------------------


class _Retrieved:
    def __init__(self, review_id: str) -> None:
        self.review_id = review_id
        self.text = "review contoh"
        self.emotion = "anger"


class FakeClassifier:
    def __init__(self, label: str = "anger", confidence: float = 0.9) -> None:
        self.label, self.confidence = label, confidence

    def predict(self, text: str) -> dict:
        return {"label": self.label, "confidence": self.confidence}


class FakeRetriever:
    def query(self, text: str, n_results: int = 5, emotions=None) -> list:
        return [_Retrieved("train_1"), _Retrieved("train_2")]


class FakeLLM:
    provider = "fake"
    model = "x"

    def __init__(self, raise_error: bool = False) -> None:
        self.raise_error = raise_error

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
        if self.raise_error:
            raise LLMError("boom")
        return "Mohon maaf atas kendalanya, tim kami menindaklanjuti."

    def is_available(self) -> bool:
        return True


def _deps(classifier=None, llm=None) -> AgentDeps:
    return AgentDeps(
        classifier=classifier or FakeClassifier(),
        retriever=FakeRetriever(),
        llm=llm or FakeLLM(),
        draft_config=DraftConfig(top_k=3),
    )


# --- Router ---------------------------------------------------------------


@pytest.mark.parametrize(
    "label,confidence,expected",
    [
        ("anger", 0.9, "escalate"),
        ("anger", 0.75, "escalate"),  # exactly at threshold -> inclusive
        ("anger", 0.7499, "draft"),  # just below -> draft, not escalate
        ("anger", 0.5, "draft"),
        ("sadness", 0.99, "draft"),
        ("happiness", 0.99, "archive"),
        ("happiness", 0.1, "archive"),
    ],
)
def test_route_by_emotion(label, confidence, expected):
    config = RouterConfig(escalate_min_confidence=0.75)
    branch = route_by_emotion({"label": label, "confidence": confidence}, config)
    assert branch == expected


def test_route_unknown_label_defaults_to_draft():
    # An unmapped label must reach a human, not be silently archived.
    assert route_by_emotion({"label": "netral", "confidence": 0.99}) == "draft"


def test_route_missing_fields_defaults_to_draft():
    assert route_by_emotion({}) == "draft"


def test_route_case_insensitive():
    assert route_by_emotion({"label": "ANGER", "confidence": 0.9}) == "escalate"


# --- Graph ----------------------------------------------------------------


def test_runtime_escalates_high_confidence_anger():
    runtime = AgentRuntime(_deps(FakeClassifier("anger", 0.95)), AgentConfig())
    state = runtime.run("aplikasi error terus, kesal!")
    assert state["branch"] == "escalate"
    assert state["priority"] == "high"
    assert state["escalated"] is True
    assert state["draft"]
    assert state["grounding_ids"] == ["train_1", "train_2"]


def test_runtime_drafts_for_sadness():
    runtime = AgentRuntime(_deps(FakeClassifier("sadness", 0.8)), AgentConfig())
    state = runtime.run("sedih sekali pesanan tidak sampai")
    assert state["branch"] == "draft"
    assert state["priority"] == "normal"
    assert state["escalated"] is False


def test_runtime_archives_happiness_without_llm():
    runtime = AgentRuntime(_deps(FakeClassifier("happiness", 0.99)), AgentConfig())
    state = runtime.run("pelayanan memuaskan, terima kasih")
    assert state["branch"] == "archive"
    assert state["draft"] == ""
    assert state["grounding_ids"] == []


def test_runtime_draft_falls_back_on_llm_error():
    runtime = AgentRuntime(
        _deps(FakeClassifier("anger", 0.95), FakeLLM(raise_error=True)), AgentConfig()
    )
    state = runtime.run("error terus")
    # Escalate still succeeds with a safe canned draft (a human approves it).
    assert state["branch"] == "escalate"
    assert state["draft"] == _FALLBACK_DRAFT


# --- Queue store ----------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    return QueueStore(tmp_path / "queue.db")


def test_queue_add_and_get(store):
    ticket = store.add("teks", "anger", 0.9, "escalate", "high", "draft", ["train_1"])
    fetched = store.get(ticket.id)
    assert fetched.id == ticket.id
    assert fetched.status == "pending"
    assert fetched.grounding_ids == ["train_1"]


def test_queue_list_filters_by_status(store):
    a = store.add("a", "anger", 0.9, "escalate", "high", "d")
    store.add("b", "sadness", 0.8, "draft", "normal", "d")
    store.approve(a.id)
    assert len(store.list()) == 2
    assert len(store.list(status="pending")) == 1
    assert len(store.list(status="approved")) == 1


def test_queue_approve_then_reject_raises(store):
    ticket = store.add("a", "anger", 0.9, "escalate", "high", "d")
    store.approve(ticket.id)
    with pytest.raises(InvalidTransitionError):
        store.reject(ticket.id, reason="late")


def test_queue_reject_records_reason(store):
    ticket = store.add("a", "anger", 0.9, "draft", "normal", "d")
    rejected = store.reject(ticket.id, reason="nada terlalu kaku")
    assert rejected.status == "rejected"
    assert rejected.reason == "nada terlalu kaku"


def test_queue_get_missing_raises(store):
    with pytest.raises(TicketNotFoundError):
        store.get("nope")


def test_queue_stats(store):
    a = store.add("a", "anger", 0.9, "escalate", "high", "d")
    b = store.add("b", "sadness", 0.8, "draft", "normal", "d")
    store.approve(a.id)
    store.reject(b.id, reason="x")
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["escalations"] == 1
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["approval_rate"] == 0.5


def test_queue_stats_empty(store):
    assert store.stats() == {
        "total": 0,
        "pending": 0,
        "approved": 0,
        "rejected": 0,
        "escalations": 0,
        "approval_rate": 0.0,
    }


# --- Notifier -------------------------------------------------------------


def test_notifier_falls_back_to_file(tmp_path):
    log = tmp_path / "esc.log"
    notifier = EscalationNotifier(bot_token="", chat_id="", fallback_log_path=log)
    channel = notifier.notify(format_escalation("abc", "anger", 0.9, "review\nmulti-line"))
    assert channel == "file"
    # One record per line despite the multi-line message.
    assert len(log.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_notifier_telegram_configured_flag():
    assert EscalationNotifier(bot_token="t", chat_id="c").telegram_configured is True
    assert EscalationNotifier(bot_token="", chat_id="c").telegram_configured is False


# --- API ------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    from src.api.dependencies import get_agent_runtime, get_notifier, get_queue_store
    from src.api.main import app

    queue = QueueStore(tmp_path / "api_queue.db")
    # Force empty creds so the notifier always uses the file fallback: tests must
    # never hit Telegram, even when a real token is present in the environment.
    notifier = EscalationNotifier(bot_token="", chat_id="", fallback_log_path=tmp_path / "esc.log")
    runtime = AgentRuntime(_deps(FakeClassifier("anger", 0.95)), AgentConfig())

    app.dependency_overrides[get_agent_runtime] = lambda: runtime
    app.dependency_overrides[get_queue_store] = lambda: queue
    app.dependency_overrides[get_notifier] = lambda: notifier
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_agent_run_escalates_and_notifies(client):
    r = client.post("/agent/run", json={"review_text": "error terus saat bayar!"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticket"]["branch"] == "escalate"
    assert body["ticket"]["status"] == "pending"
    assert body["notified"] is True
    assert body["notify_channel"] == "file"


def test_agent_run_empty_text_422(client):
    assert client.post("/agent/run", json={"review_text": ""}).status_code == 422


def test_agent_queue_lists_created_ticket(client):
    client.post("/agent/run", json={"review_text": "error terus"})
    r = client.get("/agent/queue")
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_agent_queue_bad_status_422(client):
    assert client.get("/agent/queue?status=bogus").status_code == 422


def test_agent_approve_flow(client):
    tid = client.post("/agent/run", json={"review_text": "error terus"}).json()["ticket"]["id"]
    r = client.post(f"/agent/approve/{tid}")
    assert r.status_code == 200
    assert r.json()["ticket"]["status"] == "approved"
    # Re-approving a resolved ticket is a conflict.
    assert client.post(f"/agent/approve/{tid}").status_code == 409


def test_agent_reject_flow(client):
    tid = client.post("/agent/run", json={"review_text": "error terus"}).json()["ticket"]["id"]
    r = client.post(f"/agent/reject/{tid}", json={"reason": "nada kaku"})
    assert r.status_code == 200
    assert r.json()["ticket"]["status"] == "rejected"
    assert r.json()["ticket"]["reason"] == "nada kaku"


def test_agent_approve_missing_404(client):
    assert client.post("/agent/approve/nope").status_code == 404


def test_agent_stats_endpoint(client):
    client.post("/agent/run", json={"review_text": "error terus"})
    r = client.get("/agent/stats")
    assert r.status_code == 200
    assert r.json()["total"] == 1
    assert r.json()["escalations"] == 1
