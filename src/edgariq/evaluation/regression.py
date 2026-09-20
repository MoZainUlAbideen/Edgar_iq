"""Compares two EvalReports so a prompt/retrieval/model change gets
checked against real history, not just eyeballed on a single run."""

from __future__ import annotations

from edgariq.evaluation.models import EvalReport


def compare_reports(previous: EvalReport, current: EvalReport) -> list[str]:
    messages: list[str] = []
    previous_by_id = {r.case.id: r for r in previous.results}

    for result in current.results:
        prior = previous_by_id.get(result.case.id)
        if prior is None:
            continue  # new case, nothing to compare against
        if prior.grade.passed and not result.grade.passed:
            messages.append(
                f"REGRESSION: '{result.case.id}' passed previously but now FAILS "
                f"({result.grade.detail})"
            )
        elif not prior.grade.passed and result.grade.passed:
            messages.append(f"IMPROVEMENT: '{result.case.id}' was failing, now passes")

    if current.pass_rate < previous.pass_rate:
        messages.append(
            f"Overall pass rate dropped: {previous.pass_rate:.0%} -> {current.pass_rate:.0%}"
        )

    return messages
