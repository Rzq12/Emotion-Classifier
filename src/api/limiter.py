"""Shared slowapi rate limiter.

Kept in its own module so both ``main`` and the agent router decorate their
LLM-calling endpoints with the *same* limiter instance (the one registered on
``app.state.limiter``), avoiding a circular import through ``main``.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from src.api.config import get_settings

limiter = Limiter(key_func=get_remote_address)

#: Rate string (e.g. "10/minute") applied to endpoints that call the LLM.
RATE = f"{get_settings().rate_limit_per_minute}/minute"
