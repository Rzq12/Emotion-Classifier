"""Text cleaning and normalization for Indonesian app reviews.

Pure-Python implementation (stdlib + regex only) so the pipeline stays light and
reproducible. Each transformation is a small, independently testable function;
`clean_text` composes them according to a :class:`PreprocessConfig`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.data.slang_dict import SLANG_DICT

# --- Precompiled patterns -------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # 3+ same chars -> keep 2

# Emoji / pictographic Unicode ranges. Covers the blocks that actually appear in
# review text (emoticons, symbols, transport, supplemental, dingbats, flags).
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001faff"
    "\U00002600-\U000026ff"
    "\U00002700-\U000027bf"
    "\U0001f1e6-\U0001f1ff"
    "\U0000fe0f"
    "]+",
    flags=re.UNICODE,
)

# Keep letters, digits and basic sentence punctuation; drop the rest.
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s.,!?]")


@dataclass(frozen=True)
class PreprocessConfig:
    """Toggles for the cleaning pipeline (mirrors ``configs/data.yaml``)."""

    lowercase: bool = True
    remove_urls: bool = True
    remove_mentions: bool = True
    reduce_repeated_chars: bool = True
    normalize_slang: bool = True
    emoji_strategy: str = "remove"  # "remove" | "keep"
    min_chars: int = 3


def remove_urls(text: str) -> str:
    """Strip http(s) and www URLs."""
    return _URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    """Strip @username mentions."""
    return _MENTION_RE.sub(" ", text)


def handle_emoji(text: str, strategy: str = "remove") -> str:
    """Apply the configured emoji strategy.

    ``remove`` deletes emoji; any other value leaves them untouched.
    """
    if strategy == "remove":
        return _EMOJI_RE.sub(" ", text)
    return text


def reduce_repeated_chars(text: str) -> str:
    """Collapse 3+ repeated characters to 2 ("baguuusss" -> "baguus")."""
    return _REPEATED_CHAR_RE.sub(r"\1\1", text)


def normalize_slang(text: str) -> str:
    """Replace known slang/abbreviation tokens with their standard form."""
    tokens = text.split()
    return " ".join(SLANG_DICT.get(tok, tok) for tok in tokens)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and trim."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str, config: PreprocessConfig | None = None) -> str:
    """Run the full cleaning pipeline on a single review string.

    Returns an empty string for null-like or non-string input so callers can
    filter rows uniformly.
    """
    config = config or PreprocessConfig()

    if not isinstance(text, str):
        return ""

    if config.lowercase:
        text = text.lower()
    if config.remove_urls:
        text = remove_urls(text)
    if config.remove_mentions:
        text = remove_mentions(text)

    text = handle_emoji(text, config.emoji_strategy)

    if config.reduce_repeated_chars:
        text = reduce_repeated_chars(text)

    # Drop leftover symbols before slang lookup so tokens match dictionary keys.
    text = _NON_TEXT_RE.sub(" ", text)
    text = normalize_whitespace(text)

    if config.normalize_slang:
        text = normalize_slang(text)

    text = normalize_whitespace(text)
    return text
