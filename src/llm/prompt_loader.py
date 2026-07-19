"""Load and render prompt templates stored as files in ``src/llm/prompts/``."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _read_template(name: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **kwargs: str) -> str:
    """Load template ``name`` and substitute ``{placeholder}`` fields."""
    template = _read_template(name)
    return template.format(**kwargs)
