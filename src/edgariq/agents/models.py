"""Output models for the agent pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    form_type: str
    filing_date: str
    source_url: str


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description="Grounding/critic concerns surfaced about this answer — "
        "shown to the user rather than silently suppressed.",
    )
