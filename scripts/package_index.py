"""Package an index for deployment: gzip it into deploy/indexes/.

data/indexes/ is gitignored (big, regenerable). The deployed API needs one
index file at build time, so we commit a gzip-compressed copy under
deploy/indexes/ instead.

Usage:
    uv run python scripts/package_index.py data/indexes/NVDA_gemini.json
    -> deploy/indexes/NVDA.json.gz

Refuses an index built with a local-only embedder (Ollama): the server can't
embed queries with it, so shipping it would only fail at startup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edgariq.indexing import VectorStore

DEPLOY_DIR = Path(__file__).resolve().parent.parent / "deploy" / "indexes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--allow-ollama", action="store_true")
    args = parser.parse_args()

    store = VectorStore.load(args.index_path)
    local_only = store.embedder_id is None or store.embedder_id.startswith("ollama")
    if local_only and not args.allow_ollama:
        raise SystemExit(
            f"This index's embedder is {store.embedder_id!r}, which a hosted server can't use. "
            "Rebuild with EMBEDDING_PROVIDER=gemini (see README), or pass --allow-ollama "
            "if you really mean it."
        )

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    out = DEPLOY_DIR / f"{args.ticker.upper()}.json.gz"
    store.save(out)
    print(f"{len(store)} chunks, embedder {store.embedder_id} -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
