"""Drafter agent: writes the answer, constrained to only the retrieved
filing excerpts, with instructions to cite the source filing for every
figure and to say so explicitly when the excerpts don't cover something —
rather than filling the gap from general knowledge.
"""

from __future__ import annotations

from edgariq.agents.context_formatting import format_context
from edgariq.indexing import Chunk
from edgariq.llm import GroqClient

_SYSTEM_PROMPT = (
    "You are a financial research assistant. Answer the user's question "
    "using ONLY the provided filing excerpts below — do not use outside "
    "knowledge or guess at figures. For every number or specific claim you "
    "state, mention which filing it came from (e.g. '10-Q filed 2026-08-26'). "
    "If the excerpts do not contain enough information to fully answer the "
    "question, say so explicitly rather than filling the gap yourself."
)


def draft_answer(llm: GroqClient, question: str, chunks: list[tuple[Chunk, float]]) -> str:
    context = format_context(chunks)
    user_prompt = f"Question: {question}\n\nFiling excerpts:\n\n{context}"
    return llm.complete(system=_SYSTEM_PROMPT, user=user_prompt)