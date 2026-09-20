"""Two different grading strategies, deliberately kept separate:

- Quantitative cases (a specific number should appear) are graded with
  plain code — no LLM involved. There's no reason to ask an LLM whether
  "$75.2 billion" appears in a string; that's a deterministic check, and
  using an LLM for it would just add cost, latency, and a new way to be
  wrong.

- Qualitative cases (did the answer adequately cover a set of concepts)
  genuinely need judgment — "did it explain the export control risk well"
  isn't a substring match. This uses an LLM judge, which is itself
  fallible, so its score is treated as a signal to review, not gospel.
"""

from __future__ import annotations

import json

from edgariq.agents.numeric_check import extract_numbers, normalize_number
from edgariq.evaluation.models import EvalCase, GradeResult
from edgariq.llm import GroqClient

_JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a generated answer adequately covers a list of "
    "expected topics for a financial research question. Score from 0.0 "
    "(covers none of them) to 1.0 (clearly and accurately covers all of "
    "them). Partial coverage should get a partial score. "
    'Respond with ONLY a JSON object in this exact shape: {"score": <number '
    'between 0 and 1>, "reasoning": "<one sentence>"}. No markdown fences, '
    "no extra text before or after the JSON."
)

_QUALITATIVE_PASS_THRESHOLD = 0.7


def fact_present(fact: str, answer: str) -> bool:
    """True if `fact` shows up in `answer` — numeric facts are compared via
    the same normalized-number matching the grounding checker uses (so
    "$75.2 billion" and "75.2" are recognized as the same fact); textual
    facts are a simple case-insensitive substring check."""
    if any(ch.isdigit() for ch in fact):
        return normalize_number(fact) in extract_numbers(answer)
    return fact.lower() in answer.lower()


def grade_quantitative(case: EvalCase, answer_text: str) -> GradeResult:
    missing = [f for f in case.expected_facts if not fact_present(f, answer_text)]
    passed = not missing
    detail = "all expected facts present" if passed else f"missing facts: {missing}"
    return GradeResult(case_id=case.id, passed=passed, score=1.0 if passed else 0.0, detail=detail)


def _parse_judge_response(raw: str) -> tuple[float, str]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned.strip())
        return float(data["score"]), str(data.get("reasoning", ""))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return 0.0, f"Judge response could not be parsed as expected JSON: {raw!r}"


def grade_qualitative(llm: GroqClient, case: EvalCase, answer_text: str) -> GradeResult:
    topics = "\n".join(f"- {t}" for t in case.expected_topics)
    user_prompt = (
        f"Question: {case.question}\n\nExpected topics to cover:\n{topics}\n\n"
        f"Generated answer:\n{answer_text}"
    )
    raw = llm.complete(system=_JUDGE_SYSTEM_PROMPT, user=user_prompt)
    score, reasoning = _parse_judge_response(raw)
    return GradeResult(
        case_id=case.id, passed=score >= _QUALITATIVE_PASS_THRESHOLD, score=score, detail=reasoning
    )


def grade_case(llm: GroqClient, case: EvalCase, answer_text: str) -> GradeResult:
    if case.category == "quantitative":
        return grade_quantitative(case, answer_text)
    return grade_qualitative(llm, case, answer_text)
