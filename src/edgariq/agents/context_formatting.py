"""Shared context formatting, used by BOTH the drafter and the critic.

This exists as its own module specifically to prevent a real bug found
during live testing: the drafter was building context with source labels
([Source: 10-Q filed 2026-08-26]) attached to each chunk, while the critic
was independently building a stripped-down version with no labels at all.
The critic then correctly-but-unfairly flagged the drafter's date
citations as "unsupported by the excerpts" — because it had literally
never been shown the dates. A critic can only fairly judge an answer
against the SAME evidence the answer was written from; sharing one
function is what guarantees that, rather than trusting two independent
implementations to stay in sync.
"""

from __future__ import annotations

from edgariq.indexing import Chunk


def format_context(chunks: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, _ in chunks:
        form = chunk.metadata.get("form_type", "unknown filing")
        date = chunk.metadata.get("filing_date", "unknown date")
        blocks.append(f"[Source: {form} filed {date}]\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)