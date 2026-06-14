"""Tests for `app.ingestion.chunking.chunk_text`."""

from __future__ import annotations

import pytest

from app.ingestion.chunking import chunk_text


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []
    assert chunk_text("   \n\t  ", chunk_size=100, chunk_overlap=10) == []


def test_short_text_returns_single_chunk() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert chunks == [text]


def test_long_text_is_split_into_multiple_chunks() -> None:
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50


def test_consecutive_chunks_overlap() -> None:
    words = [f"word{i:03d}" for i in range(50)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=40, chunk_overlap=15)

    # The tail of each chunk should reappear at the start of the next chunk.
    for first, second in zip(chunks, chunks[1:], strict=False):
        first_words = first.split(" ")
        second_words = second.split(" ")
        assert first_words[-1] in second_words


def test_whitespace_is_normalized() -> None:
    text = "alpha   beta\n\ncharlie\tdelta"
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert chunks == ["alpha beta charlie delta"]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-10, 0), (100, -1), (100, 100), (100, 150)],
)
def test_invalid_configuration_raises(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
