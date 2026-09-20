"""Data models for the eval harness.

Mirrors the shape used for GPU Scout's eval harness (golden set + judge +
aggregate report) but adapted to this pipeline's AgentAnswer output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from edgariq.agents import AgentAnswer


class EvalCase(BaseModel):
    id: str
    ticker: str
    question: str
    category: Literal["quantitative", "qualitative"]
    # Quantitative cases: exact facts (numbers or short phrases) that MUST
    # appear in the answer — checked deterministically, no LLM judge needed.
    expected_facts: list[str] = Field(default_factory=list)
    # Qualitative cases: concepts the answer should cover — graded by an
    # LLM judge, since "did it discuss export control risk adequately" has
    # no single deterministic substring to check for.
    expected_topics: list[str] = Field(default_factory=list)


class GradeResult(BaseModel):
    case_id: str
    passed: bool
    score: float  # 0.0-1.0; quantitative cases are 0 or 1, qualitative is graded
    detail: str


class EvalCaseResult(BaseModel):
    case: EvalCase
    answer: AgentAnswer
    grade: GradeResult
    latency_seconds: float


class EvalReport(BaseModel):
    run_at: str  # ISO timestamp
    ticker: str
    results: list[EvalCaseResult]
    pass_rate: float
    avg_latency_seconds: float
    warning_rate: float  # fraction of cases where the agent pipeline itself raised a warning
