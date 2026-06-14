"""Optional cross-encoder reranking of semantic search results."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.retrieval.search import SearchResult


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranking model."""
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
    """Re-score `results` against `query` using a cross-encoder and return the top `top_k`.

    Cross-encoders jointly encode (query, passage) pairs and are typically
    more accurate than embedding similarity alone, at the cost of being
    much slower - hence they are applied only to the small candidate set
    returned by the initial vector search.

    Args:
        query: The original natural-language question.
        results: Candidate results from `semantic_search`.
        top_k: Number of top-scoring results to keep.

    Returns:
        A new list of up to `top_k` `SearchResult` objects, sorted by
        descending cross-encoder score, with `score` replaced by the
        cross-encoder relevance score.
    """
    if not results:
        return []

    model = get_reranker()
    pairs = [(query, result.content) for result in results]
    scores = model.predict(pairs)

    reranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    output: list[SearchResult] = []
    for result, score in reranked[:top_k]:
        output.append(
            SearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                filename=result.filename,
                chunk_index=result.chunk_index,
                content=result.content,
                score=float(score),
            )
        )
    return output
