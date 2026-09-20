from edgariq.agents import AgentAnswer
from edgariq.evaluation.models import EvalCase, EvalCaseResult, EvalReport
from edgariq.evaluation.models import GradeResult
from edgariq.evaluation.report_html import generate_html_report


def _make_report() -> EvalReport:
    case = EvalCase(
        id="q1", ticker="NVDA", question="What was revenue?",
        category="quantitative", expected_facts=["$75.2 billion"],
    )
    result = EvalCaseResult(
        case=case,
        answer=AgentAnswer(answer="Revenue was $75.2 billion <script>alert(1)</script>"),
        grade=GradeResult(case_id="q1", passed=True, score=1.0, detail="all facts present"),
        latency_seconds=1.23,
    )
    return EvalReport(
        run_at="2026-01-01T00:00:00Z", ticker="NVDA", results=[result],
        pass_rate=1.0, avg_latency_seconds=1.23, warning_rate=0.0,
    )


def test_report_includes_summary_stats():
    html_out = generate_html_report(_make_report())

    assert "NVDA" in html_out
    assert "100%" in html_out  # pass rate
    assert "1.2s" in html_out  # avg latency


def test_report_includes_case_details():
    html_out = generate_html_report(_make_report())

    assert "q1" in html_out
    assert "What was revenue?" in html_out
    assert "PASS" in html_out


def test_report_escapes_untrusted_answer_content():
    html_out = generate_html_report(_make_report())

    # The answer contains a literal <script> tag — it must be escaped, not
    # injected raw, since a real LLM answer is untrusted content once it's
    # going into an HTML page.
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out
