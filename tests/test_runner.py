from tests.fakes import FakeLLM

from edgariq.evaluation.models import EvalCase
from edgariq.evaluation.runner import run_eval
from edgariq.indexing import Chunk, VectorStore


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _make_store() -> VectorStore:
    store = VectorStore()
    store.add(
        chunks=[
            Chunk(
                text="Data Center revenue was $75.2 billion, up 92% year over year.",
                chunk_type="text",
                metadata={"form_type": "10-Q", "filing_date": "2026-08-26", "source_url": "u"},
            )
        ],
        vectors=[[1.0, 0.0]],
    )
    return store


def test_run_eval_aggregates_pass_rate_and_latency_across_mixed_case_types():
    quant_case = EvalCase(
        id="q1", ticker="NVDA", question="q1?", category="quantitative",
        expected_facts=["$75.2 billion"],
    )
    qual_case = EvalCase(
        id="q2", ticker="NVDA", question="q2?", category="qualitative",
        expected_topics=["regulatory risk"],
    )
    llm = FakeLLM(
        [
            # quant_case: planner, drafter, critic
            "search query",
            "Revenue was $75.2 billion this quarter.",
            "APPROVED",
            # qual_case: planner, drafter, critic, then the judge call
            "search query 2",
            "The main risk discussed is regulatory scrutiny impacting operations.",
            "APPROVED",
            '{"score": 0.8, "reasoning": "covers the regulatory risk topic"}',
        ]
    )

    report = run_eval([quant_case, qual_case], ticker="NVDA", store=_make_store(),
                       embedder=FakeEmbedder(), llm=llm)

    assert report.ticker == "NVDA"
    assert len(report.results) == 2
    assert report.pass_rate == 1.0
    assert report.warning_rate == 0.0
    assert report.avg_latency_seconds >= 0.0
    assert report.results[0].grade.case_id == "q1"
    assert report.results[1].grade.score == 0.8


def test_run_eval_reflects_a_failing_case_in_pass_rate():
    failing_case = EvalCase(
        id="q1", ticker="NVDA", question="q1?", category="quantitative",
        expected_facts=["$999 billion"],  # will never appear in the drafted answer below
    )
    llm = FakeLLM(["search query", "Revenue was $75.2 billion this quarter.", "APPROVED"])

    report = run_eval([failing_case], ticker="NVDA", store=_make_store(),
                       embedder=FakeEmbedder(), llm=llm)

    assert report.pass_rate == 0.0
    assert report.results[0].grade.passed is False
