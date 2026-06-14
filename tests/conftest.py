"""Shared pytest fixtures.

All tests run against a fake embedding "model" so they execute quickly and
without downloading `BAAI/bge-small-en-v1.5` or any other network resource.
"""

from __future__ import annotations

import numpy as np
import pytest

import app.ingestion.embedder as embedder_module
from app.config import settings


class FakeEmbeddingModel:
    """Deterministic stand-in for `SentenceTransformer.encode`.

    Produces unit-norm vectors of `settings.embedding_dim`, derived from a
    hash of the input text so that identical inputs always map to the same
    vector and different inputs map to different vectors.
    """

    def encode(self, texts, normalize_embeddings: bool = True, convert_to_numpy: bool = True):
        single = isinstance(texts, str)
        texts_list = [texts] if single else list(texts)

        vectors = []
        for text in texts_list:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            vector = rng.random(settings.embedding_dim)
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)

        array = np.array(vectors)
        return array[0] if single else array


@pytest.fixture(autouse=True)
def fake_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the real embedding model with `FakeEmbeddingModel` for every test."""
    monkeypatch.setattr(embedder_module, "get_embedding_model", lambda: FakeEmbeddingModel())
