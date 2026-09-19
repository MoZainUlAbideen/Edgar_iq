"""Build a persisted vector index for a company across several filings.

Usage (run on YOUR machine, not the sandbox — needs real internet + Ollama):

    uv run python scripts/build_index.py NVDA
    uv run python scripts/build_index.py NVDA --forms 10-K 10-Q --limit 4

Saves to data/indexes/<TICKER>.json. A later query/agent step just does
VectorStore.load(...) instead of re-fetching and re-embedding every time —
indexing is a separate, occasional step from answering questions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgariq.config import settings
from edgariq.indexing import OllamaEmbedder, VectorStore, chunk_filing
from edgariq.ingestion import EdgarClient, FilingType
from edgariq.parsing import parse_filing_html

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"

_FORM_TYPE_MAP = {"10-K": FilingType.ANNUAL_REPORT, "10-Q": FilingType.QUARTERLY_REPORT}


def build_index(ticker: str, forms: list[str], limit_per_form: int) -> VectorStore:
    if not settings.SEC_USER_AGENT:
        raise SystemExit(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and fill it in."
        )

    client = EdgarClient(user_agent=settings.SEC_USER_AGENT)
    embedder = OllamaEmbedder(
        base_url=settings.OLLAMA_BASE_URL, model=settings.OLLAMA_EMBEDDING_MODEL
    )
    store = VectorStore()

    profile = client.resolve_ticker(ticker)
    print(f"Building index for {profile.name} (CIK {profile.cik})")

    form_types = [_FORM_TYPE_MAP[f] for f in forms]
    filings = client.get_filing_history(
        cik=profile.cik, form_types=form_types, limit=limit_per_form * len(forms)
    )
    print(f"Found {len(filings)} filings to index: "
          f"{[f'{f.form_type.value} ({f.filing_date})' for f in filings]}")

    for filing in filings:
        print(f"\n  Processing {filing.form_type.value} filed {filing.filing_date}...")
        doc = client.download_filing(filing)
        parsed = parse_filing_html(doc.raw_html)
        chunks = chunk_filing(parsed, filing, doc.source_url)
        print(f"    {len(chunks)} chunks, embedding...")

        vectors = embedder.embed_batch(
            [c.text for c in chunks],
            on_progress=lambda i, total: print(f"    embedded {i}/{total}", end="\r"),
        )
        print()
        store.add(chunks, vectors)

    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a persisted filing index for a ticker.")
    parser.add_argument("ticker", help="e.g. NVDA")
    parser.add_argument(
        "--forms", nargs="+", default=["10-K", "10-Q"], choices=list(_FORM_TYPE_MAP)
    )
    parser.add_argument(
        "--limit", type=int, default=4, help="max filings per form type (default: 4)"
    )
    args = parser.parse_args()

    store = build_index(args.ticker, args.forms, args.limit)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INDEX_DIR / f"{args.ticker.upper()}.json"
    store.save(out_path)
    print(f"\nSaved index with {len(store)} chunks to {out_path}")


if __name__ == "__main__":
    main()
