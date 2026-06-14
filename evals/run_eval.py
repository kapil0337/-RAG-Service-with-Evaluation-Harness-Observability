"""Evaluation harness: runs the RAG pipeline over a labeled QA set and scores it with RAGAS.

Usage:
    python -m evals.run_eval

Requires a running Postgres + pgvector instance (see `DATABASE_URL`) and a
valid `GROQ_API_KEY`. The script ingests `sample_docs/` (if the corpus is
empty), answers each question in `evals/dataset.json` using the live RAG
pipeline, scores the results with RAGAS (faithfulness, answer relevancy,
context precision), additionally scores abstention accuracy on questions
that are intentionally unanswerable from the corpus, and writes a markdown
report to `evals/RESULTS.md`.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from openai import RateLimitError
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
from ragas.run_config import RunConfig
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, init_db
from app.generation.llm_client import generate_answer
from app.ingestion.pipeline import ingest_document
from app.models import Document
from app.retrieval.reranker import rerank
from app.retrieval.search import semantic_search
from evals.ragas_backends import get_ragas_embeddings, get_ragas_llm

CORPUS_DIR = Path(__file__).resolve().parent.parent / "sample_docs"
DATASET_PATH = Path(__file__).parent / "dataset.json"
REPORT_PATH = Path(__file__).parent / "RESULTS.md"

UNANSWERABLE_MARKER = "UNANSWERABLE"

# Phrases that indicate the model correctly declined to answer from the
# retrieved context rather than hallucinating.
ABSTENTION_PHRASES = (
    "do not provide",
    "does not provide",
    "doesn't provide",
    "don't provide",
    "no information",
    "not mention",
    "doesn't mention",
    "do not mention",
    "not specified",
    "not specify",
    "cannot find",
    "can't find",
    "don't know",
    "do not know",
    "not contain",
    "doesn't contain",
    "unable to find",
    "not addressed",
    "not covered",
    "not available in",
    "no mention",
)


def load_qa_dataset() -> list[dict[str, str]]:
    """Load the labeled question/ground-truth pairs."""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def ingest_corpus(db: Session) -> None:
    """Ingest every supported file in `sample_docs/` (idempotent - dedupes by content hash)."""
    for path in sorted(CORPUS_DIR.iterdir()):
        if path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
            ingest_document(db, path, path.name)


def is_abstention(answer: str) -> bool:
    """Heuristic: does the answer indicate the corpus doesn't contain the information?"""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


RETRY_SECONDS_RE = re.compile(r"try again in ([\d.]+)s")


def _generate_with_retry(question: str, results, max_retries: int = 5):
    """Call `generate_answer`, retrying with backoff on Groq TPM rate limits."""
    for attempt in range(max_retries):
        try:
            return generate_answer(question, results)
        except RateLimitError as exc:
            if attempt == max_retries - 1:
                raise
            match = RETRY_SECONDS_RE.search(str(exc))
            wait = float(match.group(1)) + 1.0 if match else 10.0
            print(f"Rate limited by Groq, retrying in {wait:.1f}s...", file=sys.stderr)
            time.sleep(wait)


def answer_question(db: Session, question: str) -> tuple[str, list[str]]:
    """Run the full retrieval + generation pipeline for one question.

    Returns:
        A tuple of `(answer_text, retrieved_context_texts)`.
    """
    results = semantic_search(db, question, settings.top_k)
    if settings.enable_reranking and results:
        results = rerank(question, results, settings.rerank_top_k)

    gen_result = _generate_with_retry(question, results)
    contexts = [r.content for r in results]
    return gen_result.answer, contexts


def build_ragas_dataset(qa_pairs: list[dict[str, str]], answers: list[str], contexts: list[list[str]]) -> EvaluationDataset:
    """Assemble a RAGAS `EvaluationDataset` from questions, generated answers, and retrieved contexts."""
    samples = [
        SingleTurnSample(
            user_input=qa["question"],
            response=answer,
            retrieved_contexts=context_list,
            reference=qa["ground_truth"],
        )
        for qa, answer, context_list in zip(qa_pairs, answers, contexts, strict=True)
    ]
    return EvaluationDataset(samples=samples)


def write_markdown_report(
    qa_pairs: list[dict[str, str]],
    answers: list[str],
    contexts: list[list[str]],
    scores_df,
) -> None:
    """Write a markdown evaluation report with per-question and aggregate scores."""
    metric_columns = ["faithfulness", "answer_relevancy", "context_precision"]
    aggregates = {col: scores_df[col].mean() for col in metric_columns if col in scores_df.columns}

    unanswerable_indices = [i for i, qa in enumerate(qa_pairs) if UNANSWERABLE_MARKER in qa.get("source", "")]
    abstentions = [is_abstention(answers[i]) for i in unanswerable_indices]
    abstention_accuracy = sum(abstentions) / len(abstentions) if abstentions else None

    lines = ["# RAG Evaluation Results", ""]
    lines.append(f"Evaluated **{len(qa_pairs)}** questions against the corpus in `sample_docs/`.")
    lines.append(f"LLM: `{settings.groq_model}` | Embeddings: `{settings.embedding_model}`")
    lines.append("")
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    for metric, score in aggregates.items():
        lines.append(f"| {metric} | {score:.3f} |")
    if abstention_accuracy is not None:
        lines.append(f"| abstention_accuracy | {abstention_accuracy:.3f} |")
    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| # | Question | Answer | " + " | ".join(metric_columns) + " | abstained |")
    lines.append("|---|---|---|" + "---|" * len(metric_columns) + "---|")
    for i, (qa, answer) in enumerate(zip(qa_pairs, answers, strict=True)):
        row_scores = [f"{scores_df.iloc[i][col]:.3f}" if col in scores_df.columns else "n/a" for col in metric_columns]
        question = qa["question"].replace("|", "\\|")
        short_answer = answer.replace("\n", " ").replace("|", "\\|")
        if len(short_answer) > 200:
            short_answer = short_answer[:200] + "..."
        abstained = "yes" if is_abstention(answer) else "no"
        if i not in unanswerable_indices:
            abstained += " (n/a)"
        lines.append(f"| {i + 1} | {question} | {short_answer} | " + " | ".join(row_scores) + f" | {abstained} |")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set. Set it in your environment or .env file before running evals.", file=sys.stderr)
        return 1

    init_db()
    qa_pairs = load_qa_dataset()

    db = SessionLocal()
    try:
        if db.query(Document).count() == 0:
            ingest_corpus(db)

        answers: list[str] = []
        contexts: list[list[str]] = []
        for qa in qa_pairs:
            answer, context_list = answer_question(db, qa["question"])
            answers.append(answer)
            contexts.append(context_list)
    finally:
        db.close()

    dataset = build_ragas_dataset(qa_pairs, answers, contexts)

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        run_config=RunConfig(max_workers=1),
    )

    scores_df = result.to_pandas()
    write_markdown_report(qa_pairs, answers, contexts, scores_df)
    print(f"Wrote evaluation report to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
