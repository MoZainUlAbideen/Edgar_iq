from tests.fakes import FakeLLM

from edgariq.evaluation.grading import (
    fact_present,
    grade_qualitative,
    grade_quantitative,
)
from edgariq.evaluation.models import EvalCase


def test_fact_present_matches_numeric_fact_despite_formatting():
    assert fact_present("$75.2 billion", "Revenue was $75.2 billion this quarter.") is True
    assert fact_present("75.2", "Revenue was $75.2 billion this quarter.") is True


def test_fact_present_false_when_number_missing():
    assert fact_present("$9 billion", "Revenue was $75.2 billion this quarter.") is False


def test_fact_present_matches_text_case_insensitively():
    assert fact_present("export control", "We discussed Export Control risks.") is True
    assert fact_present("supply chain risk", "We discussed export control risks.") is False


def test_grade_quantitative_passes_when_all_facts_present():
    case = EvalCase(
        id="c1", ticker="NVDA", question="q", category="quantitative",
        expected_facts=["$75.2 billion", "92%"],
    )
    grade = grade_quantitative(case, "Data center revenue was $75.2 billion, up 92% YoY.")

    assert grade.passed is True
    assert grade.score == 1.0


def test_grade_quantitative_fails_and_reports_missing_facts():
    case = EvalCase(
        id="c1", ticker="NVDA", question="q", category="quantitative",
        expected_facts=["$75.2 billion", "92%"],
    )
    grade = grade_quantitative(case, "Data center revenue was $75.2 billion.")

    assert grade.passed is False
    assert grade.score == 0.0
    assert "92%" in grade.detail


def test_grade_qualitative_parses_judge_json_response():
    llm = FakeLLM(['{"score": 0.9, "reasoning": "covers all three topics clearly"}'])
    case = EvalCase(
        id="c2", ticker="NVDA", question="q", category="qualitative",
        expected_topics=["topic a", "topic b"],
    )

    grade = grade_qualitative(llm, case, "some generated answer")

    assert grade.passed is True
    assert grade.score == 0.9
    assert "covers all three" in grade.detail


def test_grade_qualitative_handles_markdown_fenced_json():
    llm = FakeLLM(['```json\n{"score": 0.4, "reasoning": "missing one topic"}\n```'])
    case = EvalCase(
        id="c3", ticker="NVDA", question="q", category="qualitative",
        expected_topics=["topic a"],
    )

    grade = grade_qualitative(llm, case, "answer")

    assert grade.score == 0.4
    assert grade.passed is False  # below the 0.7 threshold


def test_grade_qualitative_defaults_to_zero_on_unparseable_response():
    llm = FakeLLM(["I think this answer is pretty good, no JSON here"])
    case = EvalCase(
        id="c4", ticker="NVDA", question="q", category="qualitative",
        expected_topics=["topic a"],
    )

    grade = grade_qualitative(llm, case, "answer")

    assert grade.score == 0.0
    assert grade.passed is False
    assert "could not be parsed" in grade.detail
