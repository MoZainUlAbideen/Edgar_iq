"""A minimal local vector store — flat in-memory list + numpy cosine
similarity, with JSON persistence to disk.

No need for a full vector database (Qdrant, Chroma, etc.) at this scale —
a single company's filing history is a few hundred to a couple thousand
chunks, which a flat numpy search handles in milliseconds. Swapping this
for a real vector DB later is a drop-in replacement of this one class,
not a rewrite of anything upstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from edgariq.indexing.models import Chunk


class VectorStore:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None  # shape: (n_chunks, embedding_dim)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors — must match 1:1"
            )
        new_vectors = np.array(vectors, dtype=np.float32)
        self._chunks.extend(chunks)
        self._vectors = (
            new_vectors if self._vectors is None else np.vstack([self._vectors, new_vectors])
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_fn: Callable[[Chunk], bool] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Cosine-similarity search, optionally restricted to chunks where
        filter_fn(chunk) is True. Filtering happens BEFORE ranking, not
        after — e.g. restrict to one filing's chunks first, then rank
        those by similarity, rather than ranking everything and hoping the
        right filing's chunk happens to score in the top-k. Pure semantic
        similarity has no way to distinguish "the passage about data center
        revenue" from "the passage about data center revenue IN THIS
        SPECIFIC QUARTER" when multiple quarters phrase it almost
        identically — filter_fn is what lets a caller (see retriever.py)
        use metadata like filing_date to disambiguate when the question
        actually specifies a period."""
        if self._vectors is None or len(self._chunks) == 0:
            return []

        if filter_fn is not None:
            candidate_indices = [i for i, c in enumerate(self._chunks) if filter_fn(c)]
            if not candidate_indices:
                return []
            candidate_vectors = self._vectors[candidate_indices]
        else:
            candidate_indices = list(range(len(self._chunks)))
            candidate_vectors = self._vectors

        query = np.array(query_vector, dtype=np.float32)
        norms = np.linalg.norm(candidate_vectors, axis=1) * np.linalg.norm(query) + 1e-10
        similarities = (candidate_vectors @ query) / norms

        top_k = min(top_k, len(candidate_indices))
        top_local_indices = np.argsort(-similarities)[:top_k]

        return [
            (self._chunks[candidate_indices[i]], float(similarities[i]))
            for i in top_local_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks)

    def chunks_matching(self, filter_fn: Callable[[Chunk], bool]) -> list[Chunk]:
        """All stored chunks where filter_fn(chunk) is True — for
        diagnostics/debugging retrieval issues (e.g. "how many chunks do we
        even have for this filing date, and what do they say") without
        reaching into the store's private internals."""
        return [c for c in self._chunks if filter_fn(c)]    

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "chunks": [c.model_dump() for c in self._chunks],
            "vectors": self._vectors.tolist() if self._vectors is not None else [],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        chunks = [Chunk(**c) for c in payload["chunks"]]
        if chunks:
            store.add(chunks, payload["vectors"])
        return store