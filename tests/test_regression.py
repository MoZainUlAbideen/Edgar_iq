from edgariq.agents import AgentAnswer
from edgariq.evaluation.models import EvalCase, EvalCaseResult, EvalReport, GradeResult
from edgariq.evaluation.regression import compare_reports


def _result(case_id: str, passed: bool) -> EvalCaseResult:
    case = EvalCase(id=case_id, ticker="NVDA", question="q", category="quantitative", expected_facts=["x"])
    return EvalCaseResult(
        case=case,
        answer=AgentAnswer(answer="a"),
        grade=GradeResult(case_id=case_id, passed=passed, score=1.0 if passed else 0.0, detail="d"),
        latency_seconds=0.1,
    )


def _report(pass_rate: float, results: list[EvalCaseResult]) -> EvalReport:
    return EvalReport(
        run_at="2026-01-01T00:00:00Z", ticker="NVDA", results=results,
        pass_rate=pass_rate, avg_latency_seconds=1.0, warning_rate=0.0,
    )


def test_detects_a_regression_where_a_passing_case_now_fails():
    previous = _report(1.0, [_result("q1", passed=True)])
    current = _report(0.0, [_result("q1", passed=False)])

    messages = compare_reports(previous, current)

    assert any("REGRESSION" in m and "q1" in m for m in messages)


def test_detects_an_improvement_where_a_failing_case_now_passes():
    previous = _report(0.0, [_result("q1", passed=False)])
    current = _report(1.0, [_result("q1", passed=True)])

    messages = compare_reports(previous, current)

    assert any("IMPROVEMENT" in m and "q1" in m for m in messages)


def test_no_messages_when_nothing_changed():
    previous = _report(1.0, [_result("q1", passed=True)])
    current = _report(1.0, [_result("q1", passed=True)])

    assert compare_reports(previous, current) == []


def test_flags_overall_pass_rate_drop_even_with_different_cases():
    previous = _report(1.0, [_result("q1", passed=True)])
    current = _report(0.5, [_result("q1", passed=True), _result("q2", passed=False)])

    messages = compare_reports(previous, current)

    assert any("pass rate dropped" in m for m in messages)


def test_ignores_a_case_that_only_exists_in_the_current_run():
    previous = _report(1.0, [])
    current = _report(1.0, [_result("new_case", passed=True)])

    assert compare_reports(previous, current) == []
