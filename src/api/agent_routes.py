"""Agentic layer endpoints (Fase 6): run graph + human-in-the-loop queue.

Grouped under an ``APIRouter`` (prefix ``/agent``) rather than inflating
``main``. ``/agent/run`` calls the LLM so it is rate limited like the other
LLM endpoints; the queue/approve/reject endpoints are cheap DB operations.

This module intentionally avoids ``from __future__ import annotations`` so
FastAPI can resolve body/dependency types through the slowapi wrapper (same
reason as ``main``).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from src.agent.graph import AgentRuntime
from src.agent.notifier import EscalationNotifier, format_escalation
from src.agent.queue_store import (
    VALID_STATUSES,
    InvalidTransitionError,
    QueueStore,
    Ticket,
    TicketNotFoundError,
)
from src.api import schemas
from src.api.dependencies import get_agent_runtime, get_notifier, get_queue_store
from src.api.limiter import RATE, limiter

router = APIRouter(prefix="/agent", tags=["agent"])

RuntimeDep = Annotated[AgentRuntime, Depends(get_agent_runtime)]
QueueDep = Annotated[QueueStore, Depends(get_queue_store)]
NotifierDep = Annotated[EscalationNotifier, Depends(get_notifier)]


def _to_response(ticket: Ticket) -> schemas.TicketResponse:
    return schemas.TicketResponse(**ticket.to_dict())


@router.post("/run", response_model=schemas.AgentRunResponse, summary="Jalankan agent")
@limiter.limit(RATE)
def agent_run(
    request: Request,
    body: schemas.AgentRunRequest,
    runtime: RuntimeDep,
    queue: QueueDep,
    notifier: NotifierDep,
) -> schemas.AgentRunResponse:
    """Classify a review, route it through the graph, and queue the result.

    Escalated tickets trigger an escalation notification (Telegram, or the
    file fallback) so a reviewer is alerted immediately.
    """
    state = runtime.run(body.review_text)
    ticket = queue.add(
        review_text=body.review_text,
        label=state["label"],
        confidence=state["confidence"],
        branch=state["branch"],
        priority=state.get("priority", "none"),
        draft=state.get("draft", ""),
        grounding_ids=state.get("grounding_ids", []),
    )

    notified, channel = False, ""
    if ticket.branch == "escalate":
        channel = notifier.notify(
            format_escalation(ticket.id, ticket.label, ticket.confidence, ticket.review_text)
        )
        notified = True

    return schemas.AgentRunResponse(
        ticket=_to_response(ticket), notified=notified, notify_channel=channel
    )


@router.get("/queue", response_model=schemas.QueueResponse, summary="List antrean tiket")
def agent_queue(
    queue: QueueDep,
    status: str | None = None,
    limit: int = 100,
) -> schemas.QueueResponse:
    """List queued tickets (newest first), optionally filtered by ``status``."""
    if status is not None and status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail="status harus pending|approved|rejected.")
    tickets = queue.list(status=status, limit=max(1, min(limit, 500)))
    return schemas.QueueResponse(tickets=[_to_response(t) for t in tickets], count=len(tickets))


@router.post(
    "/approve/{ticket_id}", response_model=schemas.AgentRunResponse, summary="Setujui draft"
)
def agent_approve(
    ticket_id: str, queue: QueueDep, notifier: NotifierDep
) -> schemas.AgentRunResponse:
    """Approve a pending draft, then notify that it is ready to send."""
    ticket = _resolve_ticket(lambda: queue.approve(ticket_id))
    channel = notifier.notify(
        f"✅ Draft tiket [{ticket.id}] disetujui reviewer dan siap dikirim "
        f"(emosi: {ticket.label})."
    )
    return schemas.AgentRunResponse(
        ticket=_to_response(ticket), notified=True, notify_channel=channel
    )


@router.post("/reject/{ticket_id}", response_model=schemas.AgentRunResponse, summary="Tolak draft")
def agent_reject(
    ticket_id: str, body: schemas.RejectRequest, queue: QueueDep
) -> schemas.AgentRunResponse:
    """Reject a pending draft, recording the reviewer's reason."""
    ticket = _resolve_ticket(lambda: queue.reject(ticket_id, reason=body.reason))
    return schemas.AgentRunResponse(ticket=_to_response(ticket), notified=False)


@router.get("/stats", response_model=schemas.AgentStatsResponse, summary="Metrik agent")
def agent_stats(queue: QueueDep) -> schemas.AgentStatsResponse:
    """Queue metrics for the monitoring dashboard (escalations, approval rate)."""
    return schemas.AgentStatsResponse(**queue.stats())


@router.get("/notify-test", response_model=schemas.NotifyTestResponse, summary="Uji notifikasi")
def agent_notify_test(notifier: NotifierDep) -> schemas.NotifyTestResponse:
    """Send a test Telegram message and surface the exact delivery outcome.

    Debug aid: returns whether the token is read, whether a send was attempted,
    success, and the redacted error reason — so delivery can be diagnosed
    without container logs.
    """
    return schemas.NotifyTestResponse(**notifier.diagnose())


def _resolve_ticket(action) -> Ticket:
    """Run a queue transition, mapping domain errors to HTTP status codes."""
    try:
        return action()
    except TicketNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Tiket tidak ditemukan.") from exc
    except InvalidTransitionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
