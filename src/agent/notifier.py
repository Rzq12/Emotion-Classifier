"""Escalation notifier: Telegram bot with a file/console fallback.

When ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` are set, escalations are
pushed to Telegram (chosen for the lowest setup friction — no business
approval). Otherwise the message is appended to a log file and logged, so the
pipeline never blocks on notification setup (PLAN risk mitigation).

Uses only stdlib ``urllib`` to avoid adding an HTTP dependency. Notification
failures are logged, never raised: they must not fail an approved action.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT_SECONDS = 5


class EscalationNotifier:
    """Send short escalation messages to Telegram, falling back to a log file."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        fallback_log_path: str | Path = "data/monitoring/escalations.log",
    ) -> None:
        # Only fall back to the environment when an arg is omitted (None). An
        # explicit "" means "no credential" and must not read env — otherwise a
        # token in the environment would leak into tests meant to be offline.
        # ``.strip()`` guards against a trailing space/newline in a pasted secret.
        raw_token = os.getenv("TELEGRAM_BOT_TOKEN", "") if bot_token is None else bot_token
        raw_chat = os.getenv("TELEGRAM_CHAT_ID", "") if chat_id is None else chat_id
        self.bot_token = raw_token.strip()
        self.chat_id = raw_chat.strip()
        self.fallback_log_path = Path(fallback_log_path)
        self._lock = threading.Lock()

    @property
    def telegram_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def notify(self, message: str) -> str:
        """Deliver ``message``; return the channel used (``telegram`` | ``file``).

        Never raises: a failed Telegram call falls back to the log file so the
        caller (an approved/escalated action) always completes.
        """
        if self.telegram_configured and self._send_telegram(message):
            return "telegram"
        self._append_log(message)
        return "file"

    def _send_telegram(self, message: str) -> bool:
        url = _TELEGRAM_API.format(token=self.bot_token)
        payload = json.dumps({"chat_id": self.chat_id, "text": message}).encode("utf-8")
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("Telegram notify failed, falling back to file: %s", exc)
            return False

    def _append_log(self, message: str) -> None:
        # Collapse the multi-line Telegram message into one record per line.
        flattened = message.replace("\n", " | ")
        line = f"{datetime.now(timezone.utc).isoformat()}\t{flattened}\n"
        try:
            with self._lock:
                self.fallback_log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.fallback_log_path.open("a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as exc:
            logger.warning("Failed to write escalation log: %s", exc)
        logger.info("escalation_notice: %s", message)


def format_escalation(ticket_id: str, label: str, confidence: float, review_text: str) -> str:
    """Build a compact, PII-light escalation message (truncated review)."""
    snippet = review_text.strip().replace("\n", " ")
    if len(snippet) > 160:
        snippet = snippet[:157] + "..."
    return (
        f"🚨 Tiket escalate baru [{ticket_id}]\n"
        f"Emosi: {label} ({confidence:.0%})\n"
        f"Review: {snippet}\n"
        f"Menunggu approval reviewer."
    )
