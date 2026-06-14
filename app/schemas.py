"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """Summary of an ingested document."""

    id: int
    filename: str
    chunk_count: int
    chunk_size: int
    chunk_overlap: int
    created_at: dt.datetime

    model_config = {"from_attributes": True}


class IngestResponse(BaseModel):
    """Result of an ingestion or re-indexing request."""

    document: DocumentOut
    chunks_created: int
    reindexed: bool = False


class QueryRequest(BaseModel):
    """A natural-language question to answer over the ingested corpus."""

    question: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Number of chunks to retrieve.")
    rerank: bool | None = Field(default=None, description="Whether to apply cross-encoder reranking.")
    document_id: int | None = Field(default=None, description="Restrict retrieval to a single document.")


class Citation(BaseModel):
    """A source chunk cited in the generated answer."""

    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


class TokenUsage(BaseModel):
    """Token accounting reported by the LLM provider."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    """Answer generated from the retrieved context, with citations and timing."""

    answer: str
    citations: list[Citation]
    model: str
    usage: TokenUsage
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Service health check result."""

    status: str
    database: bool
    embedding_model: str
    llm_model: str
