"""Reviewer authentication for state-changing agent endpoints.

``/agent/approve``, ``/agent/reject`` and ``/agent/notify-test`` act on behalf
of a human reviewer (they change ticket state or send a real Telegram message),
so they require the ``X-API-Key`` header to match ``AGENT_API_KEY``.

The check fails closed: when ``AGENT_API_KEY`` is not configured the endpoints
return 503 instead of silently becoming public. Read-only endpoints
(``/agent/queue``, ``/agent/stats``) and the rate-limited ``/agent/run`` stay
public so the demo works without a key.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException

from src.api.config import get_settings


def get_reviewer_key() -> str:
    """Return the configured reviewer key (overridable in tests)."""
    return get_settings().agent_api_key


def require_reviewer(
    expected: Annotated[str, Depends(get_reviewer_key)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Reject the request unless ``X-API-Key`` matches the reviewer key.

    Raises:
        HTTPException: 503 if no key is configured, 401 if the header is
            missing or wrong.
    """
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Aksi reviewer dinonaktifkan: AGENT_API_KEY belum dikonfigurasi di server.",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="API key reviewer tidak valid atau kosong.")
