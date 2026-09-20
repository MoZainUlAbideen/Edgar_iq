"""Diagnostic tool for debugging retrieval quality on a real question,
without guessing. Shows: the extracted date filter (if any), the planner's
generated sub-queries, exactly which chunks got retrieved and their
scores, and — when a filing date is detected — EVERY chunk in the index
for that filing, so you can see whether the fact you expected even made it
into the candidate pool, or whether it's there but just didn't rank high
enough to be picked.

Usage:
    uv run python scripts/debug_retrieval.py NVDA "What was NVIDIA's data center revenue year-over-year growth in the 10-Q filed 2024-11-20?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgariq.agents.planner import plan_search_queries
from edgariq.agents.retriever import extract_filing_date_hint, retrieve
from edgariq.config import settings
from edgariq.indexing import OllamaEmbedder, VectorStore
from edgariq.llm import GroqClient

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("question")
    args = parser.parse_args()

    index_path = INDEX_DIR / f"{args.ticker.upper()}.json"
    if not index_path.exists():
        raise SystemExit(f"No index found at {index_path}.")

    store = VectorStore.load(index_path)
    embedder = OllamaEmbedder(
        base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBEDDING_MODEL
    )
    llm = GroqClient(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        min_request_interval=settings.GROQ_MIN_REQUEST_INTERVAL_SECONDS,
    )

    date_hint = extract_filing_date_hint(args.question)
    print(f"Detected filing-date filter: {date_hint or 'none'}\n")

    queries = plan_search_queries(llm, args.question)
    print(f"Planner generated sub-queries: {queries}\n")

    results = retrieve(store, embedder, queries, original_question=args.question)
    print(f"Retrieved {len(results)} chunks (these are exactly what the drafter would see):\n")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"[{i}] score={score:.3f} type={chunk.chunk_type} filing_date={chunk.metadata.get('filing_date')}")
        print(f"    {chunk.text[:300].replace(chr(10), ' ')}...\n")

    if date_hint:
        all_for_filing = store.chunks_matching(lambda c: c.metadata.get("filing_date") == date_hint)
        print(f"\n{'=' * 60}")
        print(f"ALL {len(all_for_filing)} chunks in the index for filing_date={date_hint}")
        print("(so you can see whether the fact exists at all, and if so, why it wasn't retrieved)")
        print(f"{'=' * 60}")
        for i, chunk in enumerate(all_for_filing, 1):
            print(f"[{i}] type={chunk.chunk_type}")
            print(f"    {chunk.text[:200].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
