"""SQLite-backed ticket queue for the human-in-the-loop step.

Every graph run produces a ticket persisted here; reviewers then approve or
reject drafts through the API. Deliberately a single-file SQLite database
(``agent_queue.db``), matching the lightweight storage style already used for
``mlflow.db`` — no separate service to run for a portfolio demo.

Status lifecycle: ``pending`` -> ``approved`` | ``rejected`` (terminal).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

VALID_STATUSES = ("pending", "approved", "rejected")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Ticket:
    """One review processed by the agent, awaiting or past human review."""

    id: str
    review_text: str
    label: str
    confidence: float
    branch: str
    priority: str
    draft: str
    grounding_ids: list[str] = field(default_factory=list)
    status: str = "pending"
    reason: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)


class TicketNotFoundError(KeyError):
    """Raised when a ticket id does not exist."""


class InvalidTransitionError(RuntimeError):
    """Raised when approving/rejecting a ticket that is no longer pending."""


class QueueStore:
    """Thread-safe CRUD over the agent ticket queue."""

    def __init__(self, db_path: str | Path = "agent_queue.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        parent = Path(self.db_path).parent
        if parent and str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    id            TEXT PRIMARY KEY,
                    review_text   TEXT NOT NULL,
                    label         TEXT NOT NULL,
                    confidence    REAL NOT NULL,
                    branch        TEXT NOT NULL,
                    priority      TEXT NOT NULL,
                    draft         TEXT NOT NULL,
                    grounding_ids TEXT NOT NULL,
                    status        TEXT NOT NULL,
                    reason        TEXT NOT NULL DEFAULT '',
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                )
                """)
            # Index the columns the queue is filtered/sorted by.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at)")

    def add(
        self,
        review_text: str,
        label: str,
        confidence: float,
        branch: str,
        priority: str,
        draft: str,
        grounding_ids: list[str] | None = None,
    ) -> Ticket:
        """Insert a new pending ticket and return it."""
        ticket = Ticket(
            id=uuid.uuid4().hex[:12],
            review_text=review_text,
            label=label,
            confidence=round(float(confidence), 4),
            branch=branch,
            priority=priority,
            draft=draft,
            grounding_ids=grounding_ids or [],
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tickets (id, review_text, label, confidence, branch,
                    priority, draft, grounding_ids, status, reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket.id,
                    ticket.review_text,
                    ticket.label,
                    ticket.confidence,
                    ticket.branch,
                    ticket.priority,
                    ticket.draft,
                    json.dumps(ticket.grounding_ids, ensure_ascii=False),
                    ticket.status,
                    ticket.reason,
                    ticket.created_at,
                    ticket.updated_at,
                ),
            )
        return ticket

    def get(self, ticket_id: str) -> Ticket:
        """Return one ticket or raise :class:`TicketNotFoundError`."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        if row is None:
            raise TicketNotFoundError(ticket_id)
        return _row_to_ticket(row)

    def list(self, status: str | None = None, limit: int = 100) -> list[Ticket]:
        """List tickets (newest first), optionally filtered by status."""
        query = "SELECT * FROM tickets"
        params: list = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_ticket(r) for r in rows]

    def approve(self, ticket_id: str) -> Ticket:
        """Mark a pending ticket approved. Raises if not pending."""
        return self._resolve(ticket_id, "approved", reason="")

    def reject(self, ticket_id: str, reason: str = "") -> Ticket:
        """Mark a pending ticket rejected, recording the reason."""
        return self._resolve(ticket_id, "rejected", reason=reason)

    def _resolve(self, ticket_id: str, status: str, reason: str) -> Ticket:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if row is None:
                raise TicketNotFoundError(ticket_id)
            if row["status"] != "pending":
                raise InvalidTransitionError(
                    f"Ticket {ticket_id} is '{row['status']}', cannot set to '{status}'."
                )
            updated_at = _now()
            conn.execute(
                "UPDATE tickets SET status = ?, reason = ?, updated_at = ? WHERE id = ?",
                (status, reason, updated_at, ticket_id),
            )
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
        return _row_to_ticket(row)

    def stats(self) -> dict:
        """Aggregate counts for the monitoring dashboard.

        Returns totals per status plus escalation count and approval rate
        (approved / resolved), the proxy for draft quality named in the plan.
        """
        with self._connect() as conn:
            by_status = {
                r["status"]: r["n"]
                for r in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM tickets GROUP BY status"
                ).fetchall()
            }
            escalations = conn.execute(
                "SELECT COUNT(*) AS n FROM tickets WHERE branch = 'escalate'"
            ).fetchone()["n"]
        approved = by_status.get("approved", 0)
        rejected = by_status.get("rejected", 0)
        resolved = approved + rejected
        return {
            "total": sum(by_status.values()),
            "pending": by_status.get("pending", 0),
            "approved": approved,
            "rejected": rejected,
            "escalations": escalations,
            "approval_rate": round(approved / resolved, 4) if resolved else 0.0,
        }


def _row_to_ticket(row: sqlite3.Row) -> Ticket:
    return Ticket(
        id=row["id"],
        review_text=row["review_text"],
        label=row["label"],
        confidence=row["confidence"],
        branch=row["branch"],
        priority=row["priority"],
        draft=row["draft"],
        grounding_ids=json.loads(row["grounding_ids"]),
        status=row["status"],
        reason=row["reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
