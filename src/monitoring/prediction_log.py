"""Append-only JSONL logger for /classify predictions.

Deliberately simple (PLAN Fase 6): one JSON object per line with timestamp,
input text, predicted label, and confidence — enough input for the drift check
script without pulling in a database or external service.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Thread-safe append-only JSONL prediction log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(self, text: str, label: str, confidence: float) -> None:
        """Append one prediction record.

        Failures are logged as warnings, never raised: monitoring must not
        break the serving path.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "label": label,
            "confidence": round(float(confidence), 4),
        }
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write prediction log to %s: %s", self.path, exc)


def read_predictions(path: str | Path) -> list[dict]:
    """Read a JSONL prediction log; skips malformed lines with a warning."""
    records: list[dict] = []
    log_path = Path(path)
    if not log_path.exists():
        return records
    with log_path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", lineno, log_path)
    return records
