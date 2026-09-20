"""Retriever: runs each planned query against the vector store, merges and
dedupes results across queries, and returns the best overall set.

Pure code, no LLM call — the planner already did the reasoning about what
to search for. Deduping matters because overlapping sub-queries often
retrieve the same chunk more than once; when that happens we keep the
higher of the two scores.

Also detects an explicit filing date in the ORIGINAL question (e.g. "the
10-Q filed 2026-05-20") and, when present, restricts the vector search to
that filing's chunks before ranking. This exists because of a real
eval-harness finding: a chunk's text never states its own filing date (the
date is metadata, not chunk content), so a question that names a specific
filing has almost no semantic overlap with the actual chunk text. Pure
embedding similarity works well for "how did revenue grow" (any matching
quarter is a reasonable hit) but has no mechanism at all for "specifically
THIS quarter" — several quarters phrase the same disclosure almost
identically, so semantic search alone can't tell them apart. A date
mentioned in the question is an unambiguous, cheap signal that pure vector
similarity can't use on its own; filtering on it first is a simple form of
the hybrid (semantic + metadata) retrieval real systems need once multiple
similar-but-distinct documents are in play.

When that filter is active, retrieval also looks deeper (more results per
query, more kept overall) before cutting off. Diagnosed via
debug_retrieval.py against a real failing eval case: the chunk with the
actual figure ("Data Center revenue was up 112% from a year ago") was
correctly in the filtered candidate pool, but ranked 6th — a narrative
chunk using similar keywords ("revenue growth ... driven by data center
compute and networking") scored higher despite not containing the number
at all. A figure-dense sentence can embed less distinctively than a
topic sentence using similar words, even when the figure is exactly what
the question needs. Once a date filter has already narrowed the pool to
one filing (~100 chunks, not thousands), looking deeper before cutting off
is nearly free and meaningfully more likely to include the right chunk.
"""

from __future__ import annotations

import re

from edgariq.indexing import Chunk, OllamaEmbedder, VectorStore

_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

# Applied only when a filing-date filter is active — see the module
# docstring for why a filtered (small) candidate pool can afford to look
# deeper than the unfiltered default.
_FILTERED_TOP_K_PER_QUERY = 8
_FILTERED_MAX_TOTAL = 12


def extract_filing_date_hint(question: str) -> str | None:
    """Returns an ISO date (YYYY-MM-DD) found in the question, if any —
    e.g. "the 10-Q filed 2026-05-20" -> "2026-05-20". None if no date-like
    substring is present, in which case retrieval stays purely semantic."""
    match = _DATE_RE.search(question)
    return match.group(1) if match else None


def _chunk_key(chunk: Chunk) -> tuple[str, str]:
    """Identity for dedup purposes: same filing + same starting text."""
    return (chunk.metadata.get("accession_number", ""), chunk.text[:80])


def retrieve(
    store: VectorStore,
    embedder: OllamaEmbedder,
    queries: list[str],
    top_k_per_query: int = 4,
    max_total: int = 8,
    original_question: str | None = None,
) -> list[tuple[Chunk, float]]:
    filter_fn = None
    target_date = extract_filing_date_hint(original_question) if original_question else None
    if target_date:
        filter_fn = lambda chunk: chunk.metadata.get("filing_date") == target_date  # noqa: E731
        # A date filter already narrows the pool a lot — afford to look
        # deeper within it rather than applying the same shallow top-k used
        # for an unfiltered, thousands-of-chunks search.
        top_k_per_query = max(top_k_per_query, _FILTERED_TOP_K_PER_QUERY)
        max_total = max(max_total, _FILTERED_MAX_TOTAL)

    best_by_key: dict[tuple[str, str], tuple[Chunk, float]] = {}

    for query in queries:
        query_vector = embedder.embed(query)
        for chunk, score in store.search(query_vector, top_k=top_k_per_query, filter_fn=filter_fn):
            key = _chunk_key(chunk)
            if key not in best_by_key or score > best_by_key[key][1]:
                best_by_key[key] = (chunk, score)

    ranked = sorted(best_by_key.values(), key=lambda pair: -pair[1])
    return ranked[:max_total]