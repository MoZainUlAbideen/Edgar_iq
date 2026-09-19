"""Ask a real question and get a grounded, cited, critic-checked answer.

Requires an index already built with build_index.py, and a valid
GROQ_API_KEY in your .env.

Usage:
    uv run python scripts/ask.py NVDA "How did data center revenue grow YoY, and what export control risks did they flag?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgariq.agents import answer_question
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
        raise SystemExit(
            f"No index found at {index_path}. Build one first with:\n"
            f"  uv run python scripts/build_index.py {args.ticker.upper()}"
        )

    store = VectorStore.load(index_path)
    embedder = OllamaEmbedder(
        base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBEDDING_MODEL
    )
    llm = GroqClient(api_key=settings.GROQ_API_KEY, model=settings.GROQ_MODEL)

    print(f"Thinking about: {args.question!r}\n(planning searches, retrieving, drafting, "
          f"checking, critiquing — a few Groq calls, should take a few seconds)\n")

    result = answer_question(args.question, store, embedder, llm)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.answer)

    if result.citations:
        print("\n" + "-" * 70)
        print("SOURCES")
        print("-" * 70)
        for c in result.citations:
            print(f"  - {c.form_type} filed {c.filing_date}: {c.source_url}")

    if result.warnings:
        print("\n" + "!" * 70)
        print("WARNINGS")
        print("!" * 70)
        for w in result.warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
