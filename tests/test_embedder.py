"""Tests for `app.ingestion.embedder`."""

from __future__ import annotations

import pytest

import app.ingestion.embedder as embedder_module
from app.config import settings
from app.ingestion.embedder import embed_passages, embed_query


def test_embed_passages_returns_correct_shape() -> None:
    vectors = embed_passages(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    for vector in vectors:
        assert len(vector) == settings.embedding_dim


def test_embed_passages_empty_input_returns_empty_list() -> None:
    assert embed_passages([]) == []


def test_embed_query_returns_correct_shape() -> None:
    vector = embed_query("What is the durability target?")

    assert len(vector) == settings.embedding_dim


def test_embed_query_applies_instruction_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    class CapturingModel:
        def encode(self, text, normalize_embeddings: bool = True, convert_to_numpy: bool = True):
            captured["text"] = text
            import numpy as np

            return np.ones(settings.embedding_dim)

    monkeypatch.setattr(embedder_module, "get_embedding_model", lambda: CapturingModel())

    embed_query("What is the price of Archive storage?")

    assert captured["text"].startswith("Represent this sentence for searching relevant passages: ")
    assert captured["text"].endswith("What is the price of Archive storage?")


def test_identical_text_produces_identical_embeddings() -> None:
    a = embed_passages(["the quick brown fox"])[0]
    b = embed_passages(["the quick brown fox"])[0]

    assert a == b
