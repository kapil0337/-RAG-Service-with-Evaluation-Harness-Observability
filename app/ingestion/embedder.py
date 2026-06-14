"""Local embedding generation using sentence-transformers (BAAI/bge-small-en-v1.5)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings

# BGE models are trained to expect this instruction prefix on *queries* (but
# not on the documents/passages being indexed) to improve retrieval quality.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers embedding model."""
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed a batch of document chunks for indexing.

    Args:
        texts: Chunk texts to embed.

    Returns:
        A list of L2-normalized embedding vectors, one per input text.
    """
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user query for similarity search.

    Applies the BGE query instruction prefix, which is required for this
    model family to align query and passage embedding spaces.

    Args:
        text: The raw user question.

    Returns:
        An L2-normalized embedding vector.
    """
    model = get_embedding_model()
    vector = model.encode(_QUERY_INSTRUCTION + text, normalize_embeddings=True, convert_to_numpy=True)
    return vector.tolist()
