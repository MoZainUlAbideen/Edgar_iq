"""Wires the real index, embedder and Groq client into the FastAPI app.

Fails fast, with a clear message, if the index is missing or was built with
a different embedding model than the one this server will use for queries --
a mismatch would not crash, it would silently return garbage retrieval.
"""

from __future__ import annotations

from pathlib import Path

from edgariq.agents import answer_question
from edgariq.api.app import create_app
from edgariq.api.rate_limit import RateLimiter
from edgariq.config import settings
from edgariq.indexing import VectorStore, build_embedder
from edgariq.llm import GroqClient


def build_app(cfg=settings):
    index_path = Path(cfg.API_INDEX_PATH)
    if not index_path.exists():
        raise RuntimeError(
            f"Index not found at {index_path}. Build one and package it with "
            "scripts/build_index.py + scripts/package_index.py, then commit it."
        )

    store = VectorStore.load(index_path)
    embedder = build_embedder(cfg)

    if store.embedder_id != embedder.model_id:
        raise RuntimeError(
            f"Index was built with embedder {store.embedder_id!r} but the server is "
            f"configured for {embedder.model_id!r}. Rebuild the index with "
            "EMBEDDING_PROVIDER matching the server's."
        )

    llm = GroqClient(
        api_key=cfg.GROQ_API_KEY,
        model=cfg.GROQ_MODEL,
        min_request_interval=cfg.GROQ_MIN_REQUEST_INTERVAL_SECONDS,
    )

    def runner(question: str) -> dict:
        result = answer_question(question, store, embedder, llm)
        return result.model_dump()

    return create_app(
        runner=runner,
        limiter=RateLimiter(
            per_client_limit=cfg.API_RATE_LIMIT_PER_HOUR,
            daily_cap=cfg.API_DAILY_CAP,
        ),
        allowed_origins=[o.strip() for o in cfg.ALLOWED_ORIGINS.split(",") if o.strip()],
        ticker=cfg.API_TICKER,
        index_info={"chunks": len(store), "embedder": store.embedder_id, "model": cfg.GROQ_MODEL},
        max_pending=cfg.API_MAX_QUEUE,
    )

