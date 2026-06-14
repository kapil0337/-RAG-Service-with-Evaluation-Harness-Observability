"""FastAPI route handlers for ingestion, querying, and health checks."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.generation.llm_client import generate_answer
from app.generation.prompts import build_messages
from app.ingestion.loaders import SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ingest_document, reindex_all_documents, reindex_document
from app.models import Chunk, Document
from app.observability.tracing import QueryTrace
from app.retrieval.reranker import rerank
from app.retrieval.search import semantic_search
from app.schemas import (
    Citation,
    DocumentOut,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TokenUsage,
)

router = APIRouter()


def _document_to_out(db: Session, document: Document) -> DocumentOut:
    chunk_count = db.query(Chunk).filter(Chunk.document_id == document.id).count()
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        chunk_count=chunk_count,
        chunk_size=document.chunk_size,
        chunk_overlap=document.chunk_overlap,
        created_at=document.created_at,
    )


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    """Report service health, including database connectivity."""
    try:
        db.execute(text("SELECT 1"))
        database_ok = True
    except Exception:  # noqa: BLE001 - health check must never raise
        database_ok = False

    return HealthResponse(
        status="ok" if database_ok else "degraded",
        database=database_ok,
        embedding_model=settings.embedding_model,
        llm_model=settings.groq_model,
    )


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Ingest an uploaded document: extract, chunk, embed, and store its chunks.

    Args:
        file: The uploaded PDF, Markdown, or text file.
        chunk_size: Optional override for the chunk size (characters).
        chunk_overlap: Optional override for the chunk overlap (characters).
    """
    filename = file.filename or "document"
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        document, chunks_created, _was_duplicate = ingest_document(
            db, tmp_path, filename, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        document=_document_to_out(db, document),
        chunks_created=chunks_created,
        reindexed=False,
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentOut]:
    """List all ingested documents with their chunk counts."""
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return [_document_to_out(db, doc) for doc in documents]


@router.post("/documents/{document_id}/reindex", response_model=IngestResponse)
def reindex_single_document(
    document_id: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    db: Session = Depends(get_db),
) -> IngestResponse:
    """Re-chunk and re-embed a previously ingested document.

    Args:
        document_id: ID of the document to re-index.
        chunk_size: New chunk size (characters); defaults to the document's current value.
        chunk_overlap: New chunk overlap (characters); defaults to the document's current value.
    """
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    chunks_created = reindex_document(db, document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return IngestResponse(document=_document_to_out(db, document), chunks_created=chunks_created, reindexed=True)


@router.post("/reindex", response_model=list[IngestResponse])
def reindex_all(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    db: Session = Depends(get_db),
) -> list[IngestResponse]:
    """Re-chunk and re-embed every document in the corpus."""
    results = reindex_all_documents(db, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [
        IngestResponse(
            document=_document_to_out(db, db.get(Document, doc_id)),
            chunks_created=chunks_created,
            reindexed=True,
        )
        for doc_id, chunks_created in results.items()
    ]


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    """Answer a question using retrieval-augmented generation.

    Performs top-k semantic search over the ingested corpus, optionally
    reranks the candidates with a cross-encoder, then asks the LLM to generate
    a grounded answer that cites the retrieved chunks.
    """
    top_k = request.top_k or settings.top_k
    use_rerank = request.rerank if request.rerank is not None else settings.enable_reranking

    trace = QueryTrace(request.question)
    total_start = time.perf_counter()

    with trace.span("retrieval", top_k=top_k, document_id=request.document_id) as span_out:
        results = semantic_search(db, request.question, top_k, document_id=request.document_id)
        span_out["result_count"] = len(results)
    retrieval_latency_ms = span_out["latency_ms"]

    if use_rerank and results:
        with trace.span("rerank", candidates=len(results), rerank_top_k=settings.rerank_top_k) as span_out:
            results = rerank(request.question, results, settings.rerank_top_k)
            span_out["result_count"] = len(results)
        retrieval_latency_ms += span_out["latency_ms"]

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No indexed content found. Ingest at least one document first.",
        )

    with trace.span("generation", model=settings.groq_model) as span_out:
        gen_result = generate_answer(request.question, results)
        span_out["total_tokens"] = gen_result.total_tokens
    generation_latency_ms = span_out["latency_ms"]

    trace.log_generation("llm_generation", build_messages(request.question, results), gen_result)

    citations = [
        Citation(
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            filename=r.filename,
            chunk_index=r.chunk_index,
            content=r.content,
            score=r.score,
        )
        for r in results
    ]

    total_latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
    trace.finalize(
        answer=gen_result.answer,
        citations_count=len(citations),
        total_tokens=gen_result.total_tokens,
        total_latency_ms=total_latency_ms,
    )

    return QueryResponse(
        answer=gen_result.answer,
        citations=citations,
        model=gen_result.model,
        usage=TokenUsage(
            prompt_tokens=gen_result.prompt_tokens,
            completion_tokens=gen_result.completion_tokens,
            total_tokens=gen_result.total_tokens,
        ),
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
    )
