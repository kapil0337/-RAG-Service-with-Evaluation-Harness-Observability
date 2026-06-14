"""Tests for `app.retrieval.search` and `app.retrieval.reranker`."""

from __future__ import annotations

from typing import Any

import pytest

import app.retrieval.reranker as reranker_module
from app.models import Chunk
from app.retrieval.reranker import rerank
from app.retrieval.search import SearchResult, semantic_search


class _FakeSearchQuery:
    def __init__(self, rows: list[tuple[Chunk, str, float]]) -> None:
        self._rows = rows

    def join(self, *_args: Any, **_kwargs: Any) -> "_FakeSearchQuery":
        return self

    def order_by(self, *_args: Any, **_kwargs: Any) -> "_FakeSearchQuery":
        return self

    def limit(self, _n: int) -> "_FakeSearchQuery":
        return self

    def filter(self, *_args: Any, **_kwargs: Any) -> "_FakeSearchQuery":
        return self

    def all(self) -> list[tuple[Chunk, str, float]]:
        return self._rows


class _FakeSession:
    def __init__(self, rows: list[tuple[Chunk, str, float]]) -> None:
        self._rows = rows

    def query(self, *_args: Any, **_kwargs: Any) -> _FakeSearchQuery:
        return _FakeSearchQuery(self._rows)


def _make_chunk(chunk_id: int, document_id: int, chunk_index: int, content: str) -> Chunk:
    chunk = Chunk(document_id=document_id, chunk_index=chunk_index, content=content, embedding=[0.0] * 384)
    chunk.id = chunk_id
    return chunk


def test_semantic_search_converts_distance_to_similarity_score() -> None:
    rows = [
        (_make_chunk(1, 1, 0, "first chunk"), "doc.txt", 0.1),
        (_make_chunk(2, 1, 1, "second chunk"), "doc.txt", 0.3),
    ]
    session = _FakeSession(rows)

    results = semantic_search(session, "some query", top_k=2)

    assert [r.chunk_id for r in results] == [1, 2]
    assert results[0].score == pytest.approx(0.9)
    assert results[1].score == pytest.approx(0.7)
    assert results[0].filename == "doc.txt"
    assert results[0].content == "first chunk"


def test_semantic_search_returns_empty_for_no_rows() -> None:
    session = _FakeSession([])

    assert semantic_search(session, "anything", top_k=5) == []


def test_rerank_reorders_by_cross_encoder_score(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        SearchResult(chunk_id=1, document_id=1, filename="doc.txt", chunk_index=0, content="low relevance", score=0.9),
        SearchResult(chunk_id=2, document_id=1, filename="doc.txt", chunk_index=1, content="high relevance", score=0.5),
    ]

    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [0.1, 0.95]

    monkeypatch.setattr(reranker_module, "get_reranker", lambda: FakeCrossEncoder())

    reranked = rerank("query", results, top_k=2)

    assert [r.chunk_id for r in reranked] == [2, 1]
    assert reranked[0].score == pytest.approx(0.95)


def test_rerank_respects_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        SearchResult(chunk_id=i, document_id=1, filename="doc.txt", chunk_index=i, content=f"chunk {i}", score=0.5)
        for i in range(5)
    ]

    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            return [float(i) for i in range(len(pairs))]

    monkeypatch.setattr(reranker_module, "get_reranker", lambda: FakeCrossEncoder())

    reranked = rerank("query", results, top_k=2)

    assert len(reranked) == 2
    assert [r.chunk_id for r in reranked] == [4, 3]


def test_rerank_empty_results_returns_empty() -> None:
    assert rerank("query", [], top_k=3) == []
