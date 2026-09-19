"""Query an already-built index (see build_index.py). No SEC or bulk
embedding calls here — just loads the saved index and embeds your one query.

Usage:
    uv run python scripts/query_index.py NVDA "how did data center revenue grow"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgariq.config import settings
from edgariq.indexing import OllamaEmbedder, VectorStore

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    index_path = INDEX_DIR / f"{args.ticker.upper()}.json"
    if not index_path.exists():
        raise SystemExit(
            f"No index found at {index_path}. Build one first with:\n"
            f"  uv run python scripts/build_index.py {args.ticker.upper()}"
        )

    store = VectorStore.load(index_path)
    print(f"Loaded {len(store)} chunks for {args.ticker.upper()}\n")

    embedder = OllamaEmbedder(
        base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBEDDING_MODEL
    )
    query_vector = embedder.embed(args.question)
    results = store.search(query_vector, top_k=args.top_k)

    for i, (chunk, score) in enumerate(results, 1):
        source = chunk.metadata.get("form_type", "?")
        date = chunk.metadata.get("filing_date", "?")
        print(f"[{i}] score={score:.3f} type={chunk.chunk_type} source={source} ({date})")
        print(f"    {chunk.text[:250].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
