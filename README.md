# RAG Service with Evaluation Harness & Observability

A production-grade Retrieval-Augmented Generation (RAG) service built with
FastAPI, Postgres + pgvector, and [Groq](https://console.groq.com)
(`llama-3.3-70b-versatile`). Upload documents, ask questions, and get
grounded, citation-backed answers - with a built-in web UI, structured
observability, and a RAGAS-based evaluation harness.

```mermaid
flowchart LR
    U[User] -->|upload| ING[Ingestion]
    ING -->|chunk + embed| DB[(Postgres + pgvector)]
    U -->|question| API[FastAPI]
    API --> SEARCH[Semantic Search]
    SEARCH --> DB
    SEARCH --> RERANK[Cross-Encoder Reranker]
    RERANK --> GROQ[Groq LLM]
    GROQ -->|grounded answer + citations| U
    API --> TRACE[Observability:\nstructured logs + Langfuse]
```

## Features

- **Ingestion**: upload PDF, Markdown, or plain-text documents. Files are
  chunked (configurable size/overlap), embedded with
  `BAAI/bge-small-en-v1.5`, and stored in Postgres via `pgvector`.
- **Retrieval**: pgvector cosine-similarity search, with optional
  cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for higher
  precision.
- **Generation**: grounded, citation-aware answers via **Groq**
  (`llama-3.3-70b-versatile`), using an OpenAI-compatible client.
- **Web UI**: a lightweight built-in UI (served at `/`) for asking questions
  and managing documents - no separate frontend needed.
- **Observability**: every request is logged as structured JSON (retrieval,
  rerank, and generation spans with latency and token usage). Optionally
  forwards the same traces to [Langfuse](https://langfuse.com) if configured.
- **Evaluation**: a RAGAS-based harness (`evals/`) measuring faithfulness,
  answer relevancy, context precision, and abstention accuracy against a
  labeled QA set, with results written to `evals/RESULTS.md`.

## Project Structure

```
app/
├── main.py                          # FastAPI app, lifespan, static UI mount
├── config.py                        # Settings loaded from env / .env
├── db.py                            # SQLAlchemy engine, session, init_db
├── models.py                        # Document / Chunk ORM models
├── schemas.py                       # Pydantic request/response models
├── static/                          # Built-in web UI (HTML/CSS/JS)
├── api/
│   └── routes.py                    # /health, /documents, /query, /reindex
├── ingestion/
│   ├── loaders.py                   # PDF / Markdown / text extraction
│   ├── chunking.py                  # Configurable chunking with overlap
│   ├── embedder.py                  # Sentence-transformers embeddings
│   └── pipeline.py                  # Ingest / reindex orchestration
├── retrieval/
│   ├── search.py                    # pgvector cosine similarity search
│   └── reranker.py                  # Optional cross-encoder reranking
├── generation/
│   ├── prompts.py                   # Citation-aware prompt templates
│   └── llm_client.py                # Groq client
└── observability/
    └── tracing.py                   # Structured logging + Langfuse

evals/
├── corpus/                 # Legacy sample documents (unused)
├── dataset.json             # Labeled QA set (includes 2 abstention cases)
├── ragas_backends.py         # Groq LLM + local embeddings for RAGAS
├── run_eval.py                 # Evaluation harness entrypoint
└── RESULTS.md                   # Generated evaluation report

sample_docs/              # Sample documents for evaluation/ingestion
tests/                  # pytest suite (ingestion + retrieval)
scripts/init_db.sql       # Creates the pgvector extension
```

## Setup

### 1. Configure environment variables

Copy `.env.example` to `.env` and set your Groq API key:

```bash
cp .env.example .env
```

```ini
# --- Database ---
DATABASE_URL=postgresql+psycopg2://rag:rag@db:5432/ragdb

# --- Groq (OpenAI-compatible API) ---
# Required. Get a key from https://console.groq.com
GROQ_API_KEY=gsk_your_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile

# --- Embeddings ---
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384

# --- Reranking ---
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENABLE_RERANKING=true

# --- Chunking / retrieval defaults ---
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=5
RERANK_TOP_K=3

# --- Storage ---
UPLOAD_DIR=./data/uploads

# --- Observability: Langfuse (optional) ---
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 2. Run with Docker Compose (recommended)

```bash
make docker-up     # builds the app image and starts app + Postgres/pgvector
```

The service will be available at http://localhost:8000, with the web UI at
http://localhost:8000/ and interactive API docs at
http://localhost:8000/docs.

### 3. Or run locally

```bash
make install            # install runtime dependencies
make db-init             # create the pgvector extension and tables
make dev                  # run with auto-reload
```

This requires a local Postgres instance with the `pgvector` extension
available (see `scripts/init_db.sql`) and `DATABASE_URL` pointed at it.

## Using the Service

### Web UI

Open http://localhost:8000/ to upload documents and ask questions through the
built-in UI - no separate frontend setup required.

### API

| Method | Path                          | Description |
|--------|-------------------------------|-------------|
| GET    | `/health`                     | Service health check (DB connectivity, configured models). |
| POST   | `/documents`                  | Upload and ingest a document (PDF/Markdown/text). |
| GET    | `/documents`                  | List ingested documents and their chunk counts. |
| POST   | `/documents/{id}/reindex`     | Re-chunk and re-embed a single document. |
| POST   | `/reindex`                    | Re-chunk and re-embed every document. |
| POST   | `/query`                      | Ask a question; returns a grounded, cited answer. |

### Ingest a document

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_docs/atlas_company_faq.txt" \
  -F "chunk_size=800" \
  -F "chunk_overlap=100"
```

### Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period after probation?", "top_k": 5, "rerank": true}'
```

Example response:

```json
{
  "answer": "The notice period after probation is 60 days [1].",
  "citations": [
    {
      "chunk_id": 12,
      "document_id": 1,
      "filename": "atlas_employee_handbook.md",
      "chunk_index": 3,
      "content": "...Probation and Notice... 60-day notice period after confirmation...",
      "score": 0.87
    }
  ],
  "model": "llama-3.3-70b-versatile",
  "usage": {"prompt_tokens": 612, "completion_tokens": 24, "total_tokens": 636},
  "retrieval_latency_ms": 45.2,
  "generation_latency_ms": 540.1,
  "total_latency_ms": 588.9
}
```

## Observability

Every `/query` request emits structured JSON log lines for each stage
(`retrieval`, `rerank`, `generation`, `llm_generation`, `rag_query_complete`),
including latency and token usage:

```json
{"event": "retrieval", "question": "What is the notice period after probation?", "top_k": 5, "document_id": null, "result_count": 5, "latency_ms": 18.4}
{"event": "rerank", "question": "What is the notice period after probation?", "candidates": 5, "rerank_top_k": 3, "result_count": 3, "latency_ms": 12.1}
{"event": "llm_generation", "question": "What is the notice period after probation?", "model": "llama-3.3-70b-versatile", "answer": "The notice period after probation is 60 days [1].", "prompt_tokens": 612, "completion_tokens": 24, "total_tokens": 636}
{"event": "rag_query_complete", "question": "What is the notice period after probation?", "answer": "The notice period after probation is 60 days [1].", "citations_count": 1, "total_tokens": 636, "total_latency_ms": 588.9}
```

If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, the same spans are
also sent to [Langfuse](https://langfuse.com) for trace visualization,
latency breakdowns, and token-usage dashboards. If left blank, the service
falls back to structured JSON logging only - no setup required.

## Testing

```bash
make install-dev
make test
```

Tests cover chunking edge cases, document loaders, embedding shape/
determinism, the ingestion pipeline (dedup + re-indexing), semantic search
scoring, and reranking - all using a fake embedding model so no network
access or GPU is required.

## Evaluation Results

The evaluation harness in `evals/` ingests the sample corpus in
`sample_docs/` (if not already ingested), answers each question in
`evals/dataset.json` using the live RAG pipeline, and scores the results with
[RAGAS](https://github.com/explodinggradients/ragas) using
**Groq (`llama-3.3-70b-versatile`)** as the judge LLM and
**`BAAI/bge-small-en-v1.5`** for embedding-based metrics. It also reports
**abstention accuracy**: whether the model correctly declines to answer
questions that the corpus does not cover.

```bash
make install-eval
make eval        # writes evals/RESULTS.md
```

See [`evals/RESULTS.md`](evals/RESULTS.md) for the full per-question breakdown.

- **Faithfulness** measures whether claims in the answer are supported by the retrieved context.
- **Answer relevancy** measures how directly the answer addresses the question.
- **Context precision** measures whether the retrieved chunks ranked highly are actually relevant to the ground-truth answer.
- **Abstention accuracy** measures whether the model correctly says "I don't know" for questions the corpus doesn't cover, instead of hallucinating an answer.

## Configuration Reference

| Variable               | Default                                  | Description |
|------------------------|-------------------------------------------|-------------|
| `DATABASE_URL`          | `postgresql+psycopg2://rag:rag@localhost:5432/ragdb` | SQLAlchemy connection string. |
| `GROQ_API_KEY`          | _(required)_                               | API key from https://console.groq.com. |
| `GROQ_BASE_URL`         | `https://api.groq.com/openai/v1`           | Groq's OpenAI-compatible base URL. |
| `GROQ_MODEL`            | `llama-3.3-70b-versatile`                  | Chat model used for generation. |
| `EMBEDDING_MODEL`       | `BAAI/bge-small-en-v1.5`                   | Sentence-transformers embedding model. |
| `EMBEDDING_DIM`         | `384`                                       | Embedding vector dimension (must match `EMBEDDING_MODEL`). |
| `RERANKER_MODEL`        | `cross-encoder/ms-marco-MiniLM-L-6-v2`     | Cross-encoder used for reranking. |
| `ENABLE_RERANKING`      | `true`                                      | Whether reranking is applied by default. |
| `CHUNK_SIZE`            | `800`                                       | Default chunk size in characters. |
| `CHUNK_OVERLAP`         | `100`                                       | Default chunk overlap in characters. |
| `TOP_K`                 | `5`                                         | Default number of chunks retrieved. |
| `RERANK_TOP_K`          | `3`                                         | Number of chunks kept after reranking. |
| `UPLOAD_DIR`            | `./data/uploads`                           | Directory for temporarily storing uploads. |
| `LANGFUSE_PUBLIC_KEY`   | _(empty - disabled)_                       | Optional Langfuse public key. |
| `LANGFUSE_SECRET_KEY`   | _(empty - disabled)_                       | Optional Langfuse secret key. |
| `LANGFUSE_HOST`         | `https://cloud.langfuse.com`               | Langfuse host. |
