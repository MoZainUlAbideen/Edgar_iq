"""Critic agent: a second LLM pass whose only job is to find claims in the
draft that AREN'T clearly backed by the retrieved excerpts — a semantic
check complementary to the numeric grounding check (which only catches
missing/wrong numbers, not overstated or misattributed claims).

Deliberately a separate call from the drafter, using a different framing
("find problems with this answer" vs "write this answer") — asking the
same completion to both write and grade its own work tends to just
rubber-stamp itself.

Uses the SAME format_context() as the drafter (see context_formatting.py)
so the critic is judging the answer against exactly what it was written
from — an earlier version built its own stripped-down context here, which
caused the critic to flag correct filing-date citations as "unsupported"
simply because it had never been shown the dates in the first place.
"""

from __future__ import annotations

from edgariq.agents.context_formatting import format_context
from edgariq.indexing import Chunk
from edgariq.llm import GroqClient

_SYSTEM_PROMPT = (
    "You are a strict fact-checker reviewing a draft financial research "
    "answer against its source excerpts. Identify any claim in the draft "
    "that is NOT clearly supported by the excerpts — including numbers "
    "that don't appear, claims attributed to the wrong period, or "
    "conclusions the excerpts don't actually support. "
    "If every claim is well-supported, respond with EXACTLY the single "
    "word: APPROVED. Otherwise, list each unsupported claim on its own "
    "line, each starting with 'UNSUPPORTED: '."
)


def critique_answer(
    llm: GroqClient, question: str, draft: str, chunks: list[tuple[Chunk, float]]
) -> str:
    context = format_context(chunks)
    user_prompt = (
        f"Question: {question}\n\nDraft answer:\n{draft}\n\nSource excerpts:\n{context}"
    )
    return llm.complete(system=_SYSTEM_PROMPT, user=user_prompt)


def is_approved(critique: str) -> bool:
    return critique.strip().upper() == "APPROVED"