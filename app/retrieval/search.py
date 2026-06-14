"""Top-k semantic search over chunk embeddings stored in pgvector."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.ingestion.embedder import embed_query
from app.models import Chunk, Document


@dataclass(slots=True)
class SearchResult:
    """A single retrieved chunk with its similarity score."""

    chunk_id: int
    document_id: int
    filename: str
    chunk_index: int
    content: str
    score: float


def semantic_search(
    db: Session,
    query: str,
    top_k: int,
    document_id: int | None = None,
) -> list[SearchResult]:
    """Embed `query` and return the `top_k` most similar chunks.

    Similarity is computed as cosine similarity (`1 - cosine_distance`)
    against the pgvector `embedding` column. Both stored and query
    embeddings are L2-normalized, so cosine similarity ranges from -1 to 1
    (in practice close to 0..1 for natural-language text).

    Args:
        db: Active database session.
        query: The natural-language question to search for.
        top_k: Maximum number of results to return.
        document_id: If set, restrict the search to a single document.

    Returns:
        A list of `SearchResult` ordered by descending similarity score.
    """
    query_embedding = embed_query(query)
    distance = Chunk.embedding.cosine_distance(query_embedding)

    stmt = (
        db.query(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(top_k)
    )
    if document_id is not None:
        stmt = stmt.filter(Chunk.document_id == document_id)

    results: list[SearchResult] = []
    for chunk, filename, distance_value in stmt.all():
        results.append(
            SearchResult(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=filename,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                score=1.0 - float(distance_value),
            )
        )
    return results
