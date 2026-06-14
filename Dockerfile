FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies required by psycopg2 and torch/sentence-transformers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-eval.txt ./
# Install CPU-only torch first so sentence-transformers doesn't pull the
# CUDA-enabled build (saves several GB of unused GPU deps).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# Includes the eval extras (ragas, langchain-groq, etc.) so `make eval` works
# inside the container without a separate install step.
RUN pip install -r requirements-eval.txt

# RAGAS's faithfulness metric uses nltk's sentence tokenizer.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY app ./app
COPY evals ./evals
COPY scripts ./scripts
COPY sample_docs ./sample_docs

# Pre-download the embedding and reranker models so the first request isn't slow.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
