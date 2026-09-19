"""Turns a ParsedFiling into retrieval-sized Chunks.

Two deliberate design choices here:

1. Text sections are greedily grouped up to a target size (~1000 chars)
   rather than embedded one-per-line. 455 one-line sections (what our
   parser extracts from a real 10-Q) are too fine-grained to embed well —
   a single line rarely contains a complete thought, so its embedding is a
   poor search target. Grouping into paragraph/section-sized chunks gives
   the embedding model actual context to work with.

2. Tables are NEVER merged with surrounding text or split across chunks —
   each table becomes exactly one chunk (its markdown + the heading we
   guessed as its context). Splitting a table mid-row would silently
   corrupt the one piece of content where exact values matter most.
"""

from __future__ import annotations

from edgariq.indexing.models import Chunk
from edgariq.ingestion.models import FilingMetadata
from edgariq.parsing.models import ParsedFiling

_TARGET_CHUNK_CHARS = 1000
_MAX_CHUNK_CHARS = 1500


def _group_text_sections(sections: list[str]) -> list[str]:
    """Greedily accumulate sections into chunks near the target size,
    never exceeding the max (except when a single section alone is
    already bigger than max — kept whole rather than split mid-sentence)."""
    grouped: list[str] = []
    current: list[str] = []
    current_len = 0

    for section in sections:
        projected_len = current_len + len(section) + 1
        if current and projected_len > _MAX_CHUNK_CHARS:
            grouped.append(" ".join(current))
            current, current_len = [section], len(section)
        else:
            current.append(section)
            current_len = projected_len
            if current_len >= _TARGET_CHUNK_CHARS:
                grouped.append(" ".join(current))
                current, current_len = [], 0

    if current:
        grouped.append(" ".join(current))

    return grouped


def chunk_filing(parsed: ParsedFiling, metadata: FilingMetadata, source_url: str) -> list[Chunk]:
    """Convert a parsed filing into Chunks ready for embedding."""
    base_metadata = {
        "cik": metadata.cik,
        "form_type": metadata.form_type.value,
        "filing_date": str(metadata.filing_date),
        "accession_number": metadata.accession_number,
        "source_url": source_url,
    }

    chunks = [
        Chunk(text=text, chunk_type="text", metadata=base_metadata)
        for text in _group_text_sections(parsed.text_sections)
    ]

    for table in parsed.tables:
        table_text = f"{table.context}\n\n{table.to_markdown()}".strip()
        chunks.append(Chunk(text=table_text, chunk_type="table", metadata=base_metadata))

    return chunks
