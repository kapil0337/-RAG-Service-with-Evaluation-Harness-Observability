"""Tracing for retrieval + generation requests.

Every request is logged as structured JSON (always on, zero configuration).
If `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are configured, the same
spans are additionally sent to Langfuse for richer trace visualization,
latency breakdowns, and token-usage dashboards.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from app.config import settings
from app.generation.llm_client import GenerationResult

logger = logging.getLogger("rag_service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _langfuse_enabled() -> bool:
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


@lru_cache(maxsize=1)
def get_langfuse_client() -> Any:
    """Build and cache the Langfuse client (only called when configured)."""
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


class QueryTrace:
    """Aggregates structured logs (and an optional Langfuse trace) for one query."""

    def __init__(self, question: str) -> None:
        self.question = question
        self._trace: Any | None = None
        if _langfuse_enabled():
            self._trace = get_langfuse_client().trace(
                name="rag_query", input={"question": question}
            )

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[dict[str, Any]]:
        """Time a block of work, logging latency and any extra fields written to the yielded dict.

        Args:
            name: Span name, e.g. `"retrieval"` or `"rerank"`.
            **metadata: Extra input fields to attach to the log record / span.

        Yields:
            A mutable dict. Anything written to it before the block exits is
            included in the final log record and Langfuse span output.
        """
        start = time.perf_counter()
        output: dict[str, Any] = {}
        lf_span = None
        if self._trace is not None:
            lf_span = self._trace.span(name=name, input=metadata)
        try:
            yield output
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            output["latency_ms"] = latency_ms
            record = {"event": name, "question": self.question, **metadata, **output}
            logger.info(json.dumps(record, default=str))
            if lf_span is not None:
                lf_span.end(output=output)

    def log_generation(self, name: str, messages: list[dict[str, str]], result: GenerationResult) -> None:
        """Record an LLM generation call with its prompt, output, and token usage."""
        record = {
            "event": name,
            "question": self.question,
            "model": result.model,
            "answer": result.answer,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }
        logger.info(json.dumps(record, default=str))
        if self._trace is not None:
            self._trace.generation(
                name=name,
                model=result.model,
                input=messages,
                output=result.answer,
                usage={
                    "input": result.prompt_tokens,
                    "output": result.completion_tokens,
                    "total": result.total_tokens,
                    "unit": "TOKENS",
                },
            )

    def finalize(self, **output: Any) -> None:
        """Attach final output fields to the trace and flush to Langfuse if enabled."""
        record = {"event": "rag_query_complete", "question": self.question, **output}
        logger.info(json.dumps(record, default=str))
        if self._trace is not None:
            self._trace.update(output=output)
            get_langfuse_client().flush()
