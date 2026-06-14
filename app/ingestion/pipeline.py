"""Orchestrates document ingestion: load -> hash -> chunk -> embed -> persist."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion.chunking import chunk_text
from app.ingestion.embedder import embed_passages
from app.ingestion.loaders import load_document
from app.models import Chunk, Document


def _hash_text(text: str) -> str:
    """Return a stable SHA-256 hex digest of `text`, used for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_and_store_chunks(
    db: Session, document: Document, chunk_texts: list[str]
) -> int:
    """Embed `chunk_texts` and persist them as `Chunk` rows for `document`.

    Returns the number of chunks created.
    """
    if not chunk_texts:
        return 0

    embeddings = embed_passages(chunk_texts)
    for index, (content, embedding) in enumerate(zip(chunk_texts, embeddings, strict=True)):
        db.add(
            Chunk(
                document_id=document.id,
                chunk_index=index,
                content=content,
                embedding=embedding,
            )
        )
    return len(chunk_texts)


def ingest_document(
    db: Session,
    file_path: Path,
    filename: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> tuple[Document, int, bool]:
    """Ingest a single document: extract text, chunk it, embed, and store.

    If a document with identical content has already been ingested, the
    existing record is returned unchanged (deduplication by content hash).

    Args:
        db: Active database session.
        file_path: Path to the uploaded file on disk.
        filename: Original filename (used for display and citations).
        chunk_size: Override for the configured chunk size, in characters.
        chunk_overlap: Override for the configured chunk overlap, in characters.

    Returns:
        A tuple of `(document, chunks_created, was_duplicate)`.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    raw_text = load_document(file_path)
    content_hash = _hash_text(raw_text)

    existing = db.query(Document).filter(Document.content_hash == content_hash).one_or_none()
    if existing is not None:
        return existing, 0, True

    document = Document(
        filename=filename,
        content_hash=content_hash,
        raw_text=raw_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    db.add(document)
    db.flush()  # assign document.id before creating chunks

    chunks = chunk_text(raw_text, chunk_size, chunk_overlap)
    chunks_created = _embed_and_store_chunks(db, document, chunks)

    db.commit()
    db.refresh(document)
    return document, chunks_created, False


def reindex_document(
    db: Session,
    document: Document,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Re-chunk and re-embed an existing document's stored raw text.

    Useful after changing chunking parameters or the embedding model.
    Existing chunks for the document are deleted and replaced.

    Args:
        db: Active database session.
        document: The document to re-index (must already be persisted).
        chunk_size: New chunk size; defaults to the document's current value.
        chunk_overlap: New chunk overlap; defaults to the document's current value.

    Returns:
        The number of chunks created.
    """
    chunk_size = chunk_size or document.chunk_size
    chunk_overlap = chunk_overlap or document.chunk_overlap

    db.query(Chunk).filter(Chunk.document_id == document.id).delete()

    document.chunk_size = chunk_size
    document.chunk_overlap = chunk_overlap

    chunks = chunk_text(document.raw_text, chunk_size, chunk_overlap)
    chunks_created = _embed_and_store_chunks(db, document, chunks)

    db.commit()
    db.refresh(document)
    return chunks_created


def reindex_all_documents(
    db: Session, chunk_size: int | None = None, chunk_overlap: int | None = None
) -> dict[int, int]:
    """Re-index every document currently stored in the database.

    Args:
        db: Active database session.
        chunk_size: New chunk size applied to all documents.
        chunk_overlap: New chunk overlap applied to all documents.

    Returns:
        A mapping of `document_id` to the number of chunks created.
    """
    results: dict[int, int] = {}
    for document in db.query(Document).all():
        results[document.id] = reindex_document(db, document, chunk_size, chunk_overlap)
    return results
