"""Data model for a retrieval-sized chunk of a filing.

Every chunk carries provenance metadata (which filing, which form type,
filed when, source URL) — this is what eventually lets the final answer
cite "NVIDIA 10-Q filed 2026-08-26" instead of just asserting a number with
no way to check it. Grounding starts here, not at the answer stage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    chunk_type: Literal["text", "table"]
    metadata: dict[str, str] = Field(default_factory=dict)
