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

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self._vectors is None or len(self._chunks) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)
        # Cosine similarity against every stored vector at once.
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query) + 1e-10
        similarities = (self._vectors @ query) / norms

        top_k = min(top_k, len(self._chunks))
        top_indices = np.argsort(-similarities)[:top_k]

        return [(self._chunks[i], float(similarities[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {
            "chunks": [c.model_dump() for c in self._chunks],
            "vectors": self._vectors.tolist() if self._vectors is not None else [],
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        payload = json.loads(Path(path).read_text())
        store = cls()
        chunks = [Chunk(**c) for c in payload["chunks"]]
        if chunks:
            store.add(chunks, payload["vectors"])
        return store
