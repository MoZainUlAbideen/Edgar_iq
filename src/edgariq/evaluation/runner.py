"""Runs every golden-set case through the real agent pipeline, times it,
grades it, and aggregates into one EvalReport."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from edgariq.agents import answer_question
from edgariq.evaluation.grading import grade_case
from edgariq.evaluation.models import EvalCase, EvalCaseResult, EvalReport
from edgariq.indexing import OllamaEmbedder, VectorStore
from edgariq.llm import GroqClient


def run_eval(
    cases: list[EvalCase],
    ticker: str,
    store: VectorStore,
    embedder: OllamaEmbedder,
    llm: GroqClient,
) -> EvalReport:
    results: list[EvalCaseResult] = []

    for case in cases:
        start = time.monotonic()
        answer = answer_question(case.question, store, embedder, llm)
        latency = time.monotonic() - start

        grade = grade_case(llm, case, answer.answer)

        results.append(
            EvalCaseResult(case=case, answer=answer, grade=grade, latency_seconds=latency)
        )

    n = len(results) or 1  # avoid division by zero on an empty case list
    pass_rate = sum(r.grade.passed for r in results) / n
    avg_latency = sum(r.latency_seconds for r in results) / n
    warning_rate = sum(bool(r.answer.warnings) for r in results) / n

    return EvalReport(
        run_at=datetime.now(timezone.utc).isoformat(),
        ticker=ticker,
        results=results,
        pass_rate=pass_rate,
        avg_latency_seconds=avg_latency,
        warning_rate=warning_rate,
    )
