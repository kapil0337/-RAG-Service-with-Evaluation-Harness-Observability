"""Client for Groq's hosted models via its OpenAI-compatible API."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.config import settings
from app.generation.prompts import build_messages
from app.retrieval.search import SearchResult


@dataclass(slots=True)
class GenerationResult:
    """The generated answer plus model/token accounting."""

    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Build and cache an OpenAI-compatible client pointed at the Groq API.

    Note:
        The OpenAI SDK is used purely as an HTTP client because Groq exposes
        an OpenAI-compatible `/v1/chat/completions` endpoint.
    """
    return OpenAI(base_url=settings.groq_base_url, api_key=settings.groq_api_key)


def generate_answer(question: str, results: list[SearchResult]) -> GenerationResult:
    """Generate a grounded, citation-aware answer from retrieved context.

    Args:
        question: The user's natural-language question.
        results: Retrieved (and optionally reranked) context passages.

    Returns:
        A `GenerationResult` containing the answer text and token usage.
    """
    client = get_client()
    messages = build_messages(question, results)

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.2,
    )

    answer = response.choices[0].message.content or ""
    usage = response.usage

    return GenerationResult(
        answer=answer,
        model=response.model,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
    )
