"""Prompt construction for grounded, citation-aware answer generation."""

from __future__ import annotations

from app.retrieval.search import SearchResult

SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. Answer the user's question "
    "using ONLY the information in the numbered context passages below. "
    "Every factual claim in your answer must be followed by a citation marker "
    "like [1] or [2] referencing the passage(s) it came from. "
    "If the passages do not contain enough information to answer the question, "
    "say so explicitly instead of guessing."
)


def build_context_block(results: list[SearchResult]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt.

    Args:
        results: Retrieved (and optionally reranked) search results, in the
            order they should be presented to the model. The 1-based
            position in this list is the citation number used in the prompt.

    Returns:
        A string with one `[n] source=<filename> chunk=<index>` header per
        passage, followed by its content.
    """
    blocks = []
    for i, result in enumerate(results, start=1):
        header = f"[{i}] source={result.filename} chunk={result.chunk_index}"
        blocks.append(f"{header}\n{result.content}")
    return "\n\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict[str, str]]:
    """Build the chat messages sent to the LLM for grounded generation.

    Args:
        question: The user's natural-language question.
        results: Retrieved context passages, numbered for citation.

    Returns:
        A list of `{"role": ..., "content": ...}` messages suitable for the
        OpenAI-compatible chat completions API.
    """
    context_block = build_context_block(results)
    user_content = (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the context passages above, citing sources "
        "with [n] markers."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
