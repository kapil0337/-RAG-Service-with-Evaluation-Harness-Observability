"""RAGAS LLM and embeddings backends, wired to Groq and the local embedding model.

RAGAS metrics need an LLM (for judging faithfulness/relevancy/precision) and
an embeddings model (for similarity-based metrics). Per project requirements,
both are provided locally / via Groq - no OpenAI models are used anywhere.
"""

from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

from app.config import settings


def get_ragas_llm() -> LangchainLLMWrapper:
    """Return a RAGAS-compatible LLM wrapper backed by Groq.

    Uses the same configuration as the main application (`GROQ_API_KEY`,
    `GROQ_MODEL`).
    """
    chat_model = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        # The `groq` SDK's request paths already include `/openai/v1/...`, so
        # its base URL must be the bare host - not `settings.groq_base_url`
        # (which is `https://api.groq.com/openai/v1`, for the OpenAI-compatible
        # client used elsewhere). Passing it explicitly avoids the SDK picking
        # up the `GROQ_BASE_URL` env var and doubling the `/openai/v1` segment.
        base_url="https://api.groq.com",
        temperature=0.0,
    )
    return LangchainLLMWrapper(chat_model)


def get_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """Return a RAGAS-compatible embeddings wrapper using the same local model as ingestion."""
    hf_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainEmbeddingsWrapper(hf_embeddings)
