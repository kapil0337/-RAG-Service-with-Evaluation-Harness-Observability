"""Tests for `app.generation.prompts`."""

from __future__ import annotations

from app.generation.prompts import build_context_block, build_messages
from app.retrieval.search import SearchResult


def _result(chunk_id: int, filename: str, chunk_index: int, content: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, document_id=1, filename=filename, chunk_index=chunk_index, content=content, score=0.9
    )


def test_build_context_block_numbers_passages_from_one() -> None:
    results = [
        _result(1, "a.txt", 0, "Content A"),
        _result(2, "b.txt", 3, "Content B"),
    ]

    block = build_context_block(results)

    assert "[1] source=a.txt chunk=0" in block
    assert "Content A" in block
    assert "[2] source=b.txt chunk=3" in block
    assert "Content B" in block


def test_build_context_block_empty() -> None:
    assert build_context_block([]) == ""


def test_build_messages_includes_system_and_user_roles() -> None:
    results = [_result(1, "a.txt", 0, "Relevant content")]

    messages = build_messages("What is X?", results)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is X?" in messages[1]["content"]
    assert "Relevant content" in messages[1]["content"]
    assert "[1]" in messages[1]["content"]
