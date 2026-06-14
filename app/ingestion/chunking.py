"""Text chunking with configurable size and overlap."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks of approximately `chunk_size` characters.

    Splitting happens on whitespace boundaries so words are never broken
    mid-token. Consecutive whitespace is collapsed to single spaces before
    chunking, which keeps offsets simple and chunks clean for embedding.

    Args:
        text: The full document text.
        chunk_size: Target maximum number of characters per chunk.
        chunk_overlap: Number of characters of overlap between consecutive chunks.

    Returns:
        A list of non-empty chunk strings. Empty input returns an empty list.

    Raises:
        ValueError: If `chunk_size` is not positive or `chunk_overlap` is
            negative or not smaller than `chunk_size`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = _WHITESPACE_RE.sub(" ", text).strip()
    if not normalized:
        return []

    words = normalized.split(" ")

    chunks: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        added_len = len(word) + (1 if current_words else 0)
        if current_words and current_len + added_len > chunk_size:
            chunks.append(" ".join(current_words))

            # Build the overlap tail (in characters) for the next chunk.
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current_words):
                w_len = len(w) + (1 if overlap_words else 0)
                if overlap_len + w_len > chunk_overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += w_len

            current_words = overlap_words
            current_len = overlap_len

            added_len = len(word) + (1 if current_words else 0)

        current_words.append(word)
        current_len += added_len

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks
