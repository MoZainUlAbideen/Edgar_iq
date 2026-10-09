"""Resumable document embeddings.

Embedding ~1,100 chunks through a rate-limited API can fail halfway (quota,
network). Without a cache, every retry starts from zero and burns quota again.
CachingEmbedder appends each finished batch to a JSONL file keyed by a hash of
the text, so re-running the same build skips everything already done.

Only document embedding (embed_batch) is cached; query embedding passes
straight through. The cache file is named after the embedder's model_id, so
vectors from different models can never be mixed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class CachingEmbedder:
    def __init__(self, inner, cache_dir: str | Path, chunk_size: int = 20):
        self._inner = inner
        self._chunk_size = chunk_size
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", inner.model_id)
        self._path = Path(cache_dir) / f"{safe}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        if self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self._cache[row["k"]] = row["v"]
                except (ValueError, KeyError):
                    continue  # a half-written last line from a crash; just re-embed it

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def cached_count(self) -> int:
        return len(self._cache)

    def embed(self, text: str) -> list[float]:
        return self._inner.embed(text)

    def embed_batch(self, texts: list[str], on_progress=None) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if _key(t) not in self._cache))
        done = len(texts) - sum(1 for t in texts if _key(t) not in self._cache)
        if on_progress and done:
            on_progress(done, len(texts))
        for start in range(0, len(missing), self._chunk_size):
            piece = missing[start : start + self._chunk_size]
            vectors = self._inner.embed_batch(piece)
            with self._path.open("a", encoding="utf-8") as f:
                for text, vec in zip(piece, vectors):
                    self._cache[_key(text)] = vec
                    f.write(json.dumps({"k": _key(text), "v": vec}) + "\n")
            if on_progress:
                on_progress(min(len(texts), done + start + len(piece)), len(texts))
        if on_progress:
            on_progress(len(texts), len(texts))  # duplicates are embedded once, so finish at 100%
        return [self._cache[_key(t)] for t in texts]
