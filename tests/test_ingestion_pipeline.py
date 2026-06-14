"""Tests for `app.ingestion.pipeline`, using an in-memory fake session.

A real `pgvector`-backed Postgres database is not required for these tests:
the fake session below records `add`/`commit`/`delete` calls so we can
assert on the objects the pipeline builds without needing a live database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import settings
from app.ingestion.pipeline import _hash_text, ingest_document, reindex_document
from app.models import Chunk, Document


class _FakeQuery:
    def __init__(self, session: "_FakeSession", model: type) -> None:
        self._session = session
        self._model = model
        self._filters: list[Any] = []

    def filter(self, *_args: Any, **_kwargs: Any) -> "_FakeQuery":
        return self

    def one_or_none(self) -> Document | None:
        if self._model is Document:
            return self._session.existing_document
        return None

    def delete(self) -> int:
        if self._model is Chunk:
            removed = len(self._session.chunks)
            self._session.chunks.clear()
            return removed
        return 0


class _FakeSession:
    """Minimal stand-in for a SQLAlchemy `Session`."""

    def __init__(self, existing_document: Document | None = None) -> None:
        self.existing_document = existing_document
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self.committed = False
        self._next_id = 1

    def query(self, model: type) -> _FakeQuery:
        return _FakeQuery(self, model)

    def add(self, obj: Document | Chunk) -> None:
        if isinstance(obj, Document):
            obj.id = self._next_id
            self._next_id += 1
            self.documents.append(obj)
        elif isinstance(obj, Chunk):
            self.chunks.append(obj)

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.committed = True

    def refresh(self, _obj: Document) -> None:
        pass


@pytest.fixture
def long_document(tmp_path: Path) -> Path:
    path = tmp_path / "doc.txt"
    words = " ".join(f"word{i}" for i in range(300))
    path.write_text(words, encoding="utf-8")
    return path


def test_ingest_document_creates_chunks(long_document: Path) -> None:
    session = _FakeSession()

    document, chunks_created, was_duplicate = ingest_document(
        session, long_document, "doc.txt", chunk_size=200, chunk_overlap=20
    )

    assert was_duplicate is False
    assert chunks_created > 1
    assert session.committed is True
    assert len(session.chunks) == chunks_created
    assert document.filename == "doc.txt"
    assert document.chunk_size == 200
    assert document.chunk_overlap == 20

    for chunk in session.chunks:
        assert chunk.document_id == document.id
        assert len(chunk.embedding) == settings.embedding_dim


def test_ingest_document_deduplicates_by_content_hash(long_document: Path) -> None:
    raw_text = long_document.read_text(encoding="utf-8")
    existing = Document(
        id=99,
        filename="doc.txt",
        content_hash=_hash_text(raw_text),
        raw_text=raw_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    session = _FakeSession(existing_document=existing)

    document, chunks_created, was_duplicate = ingest_document(session, long_document, "doc.txt")

    assert was_duplicate is True
    assert chunks_created == 0
    assert document is existing
    assert session.chunks == []


def test_reindex_document_replaces_chunks(long_document: Path) -> None:
    session = _FakeSession()
    document, _, _ = ingest_document(session, long_document, "doc.txt", chunk_size=200, chunk_overlap=20)
    first_chunk_count = len(session.chunks)

    chunks_created = reindex_document(session, document, chunk_size=50, chunk_overlap=10)

    assert chunks_created > first_chunk_count
    assert len(session.chunks) == chunks_created
    assert document.chunk_size == 50
    assert document.chunk_overlap == 10
