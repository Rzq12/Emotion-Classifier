"""Load agent configuration from ``configs/agent.yaml`` (no hardcoded thresholds).

Mirrors the pattern used for the RAG layer: a small dataclass tree parsed once
from YAML, so behaviour (routing thresholds, grounding depth, queue location)
is reproducible and reviewable in one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.agent.router import RouterConfig


@dataclass(frozen=True)
class DraftConfig:
    """Controls grounded draft generation for escalate/draft branches."""

    top_k: int = 5
    temperature: float = 0.3
    max_tokens: int = 400


@dataclass(frozen=True)
class NotifierConfig:
    provider: str = "telegram"
    fallback_log_path: str = "data/monitoring/escalations.log"


@dataclass(frozen=True)
class AgentConfig:
    """Full agent configuration tree."""

    router: RouterConfig = field(default_factory=RouterConfig)
    draft: DraftConfig = field(default_factory=DraftConfig)
    queue_db: str = "agent_queue.db"
    notifier: NotifierConfig = field(default_factory=NotifierConfig)

    @classmethod
    def load(cls, path: str | Path) -> AgentConfig:
        """Parse ``agent.yaml`` into an :class:`AgentConfig`."""
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        draft = raw.get("draft", {})
        notifier = raw.get("notifier", {})
        return cls(
            router=RouterConfig.from_dict(raw.get("router", {})),
            draft=DraftConfig(
                top_k=int(draft.get("top_k", 5)),
                temperature=float(draft.get("temperature", 0.3)),
                max_tokens=int(draft.get("max_tokens", 400)),
            ),
            queue_db=raw.get("queue", {}).get("db_path", "agent_queue.db"),
            notifier=NotifierConfig(
                provider=notifier.get("provider", "telegram"),
                fallback_log_path=notifier.get(
                    "fallback_log_path", "data/monitoring/escalations.log"
                ),
            ),
        )
