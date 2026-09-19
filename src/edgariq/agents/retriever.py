"""Retriever: runs each planned query against the vector store, merges and
dedupes results across queries, and returns the best overall set.

Pure code, no LLM call — the planner already did the reasoning about what
to search for. Deduping matters because overlapping sub-queries often
retrieve the same chunk more than once; when that happens we keep the
higher of the two scores.
"""

from __future__ import annotations

from edgariq.indexing import Chunk, OllamaEmbedder, VectorStore


def _chunk_key(chunk: Chunk) -> tuple[str, str]:
    """Identity for dedup purposes: same filing + same starting text."""
    return (chunk.metadata.get("accession_number", ""), chunk.text[:80])


def retrieve(
    store: VectorStore,
    embedder: OllamaEmbedder,
    queries: list[str],
    top_k_per_query: int = 4,
    max_total: int = 8,
) -> list[tuple[Chunk, float]]:
    best_by_key: dict[tuple[str, str], tuple[Chunk, float]] = {}

    for query in queries:
        query_vector = embedder.embed(query)
        for chunk, score in store.search(query_vector, top_k=top_k_per_query):
            key = _chunk_key(chunk)
            if key not in best_by_key or score > best_by_key[key][1]:
                best_by_key[key] = (chunk, score)

    ranked = sorted(best_by_key.values(), key=lambda pair: -pair[1])
    return ranked[:max_total]
