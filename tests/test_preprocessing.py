"""Unit tests for text cleaning functions."""

from __future__ import annotations

from src.data.preprocessing import (
    PreprocessConfig,
    clean_text,
    handle_emoji,
    normalize_slang,
    reduce_repeated_chars,
    remove_mentions,
    remove_urls,
)


def test_remove_urls():
    result = remove_urls("cek https://halodoc.com sekarang")
    assert "halodoc.com" not in result
    assert "cek" in result and "sekarang" in result
    assert "www" not in remove_urls("buka www.gojek.com ya")


def test_remove_mentions():
    assert "@halodoc" not in remove_mentions("halo @halodoc tolong dong")


def test_handle_emoji_remove():
    assert handle_emoji("bagus banget 😍🔥", "remove").strip() == "bagus banget"


def test_handle_emoji_keep():
    text = "bagus 😍"
    assert handle_emoji(text, "keep") == text


def test_reduce_repeated_chars():
    assert reduce_repeated_chars("baguuusss") == "baguuss"
    assert reduce_repeated_chars("mantappp") == "mantapp"
    # Two or fewer repeats are left untouched.
    assert reduce_repeated_chars("good") == "good"


def test_normalize_slang():
    assert normalize_slang("aplikasi ga bisa pake") == "aplikasi tidak bisa pakai"
    assert normalize_slang("bgt mantul") == "banget mantap"


def test_clean_text_full_pipeline():
    raw = "Aplikasinya ERROR terusss @cs https://x.co 😡😡"
    cleaned = clean_text(raw)
    # "terusss" -> "teruss" (repeated chars reduced to 2; not a slang token)
    assert cleaned == "aplikasi error teruss"


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_clean_text_strips_symbols_but_keeps_basic_punctuation():
    cleaned = clean_text("Bagus!! tapi mahal#$%")
    assert "#" not in cleaned and "$" not in cleaned
    assert "bagus" in cleaned and "tapi" in cleaned


def test_clean_text_removes_emoji_by_default():
    # Full pipeline strips non-text characters, so emoji never survive cleaning
    # regardless of strategy; the per-function `keep` option is covered above.
    cfg = PreprocessConfig(emoji_strategy="remove", normalize_slang=False)
    assert clean_text("oke 🔥", cfg) == "oke"
