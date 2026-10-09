"""Build a persisted vector index for a company across several filings.

Usage (run on YOUR machine, not the sandbox — needs real internet + Ollama):

    uv run python scripts/build_index.py NVDA
    uv run python scripts/build_index.py NVDA --forms 10-K 10-Q --limit 4

Saves to data/indexes/<TICKER>.json (or --out PATH). The embedder comes from
EMBEDDING_PROVIDER in .env ("ollama" locally, "gemini" for the deployed API) and
is recorded inside the index so a mismatched server refuses to load it. A later query/agent step just does
VectorStore.load(...) instead of re-fetching and re-embedding every time —
indexing is a separate, occasional step from answering questions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgariq.config import settings
from edgariq.indexing import CachingEmbedder, VectorStore, build_embedder, chunk_filing
from edgariq.ingestion import EdgarClient, FilingType
from edgariq.parsing import parse_filing_html

INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "indexes"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embedding_cache"

_FORM_TYPE_MAP = {"10-K": FilingType.ANNUAL_REPORT, "10-Q": FilingType.QUARTERLY_REPORT}


def build_index(
    ticker: str, forms: list[str], limit_per_form: int, provider: str | None = None
) -> VectorStore:
    if not settings.SEC_USER_AGENT:
        raise SystemExit(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and fill it in."
        )

    client = EdgarClient(user_agent=settings.SEC_USER_AGENT)
    # Cache finished embeddings so an interrupted/rate-limited build resumes
    # instead of starting over (and re-spending API quota).
    embedder = CachingEmbedder(build_embedder(settings, provider), CACHE_DIR)
    print(f"Embedding with {embedder.model_id} ({embedder.cached_count} chunks already cached)")
    store = VectorStore(embedder_id=embedder.model_id)

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
    parser.add_argument(
        "--provider", choices=["ollama", "gemini"],
        help="embedding provider (default: EMBEDDING_PROVIDER from .env, i.e. ollama)",
    )
    parser.add_argument(
        "--out", help="output path (default: data/indexes/<TICKER>.json)"
    )
    args = parser.parse_args()

    store = build_index(args.ticker, args.forms, args.limit, args.provider)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else INDEX_DIR / f"{args.ticker.upper()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(out_path)
    print(f"\nSaved index with {len(store)} chunks to {out_path}")


if __name__ == "__main__":
    main()
