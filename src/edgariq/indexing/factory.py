"""Pick the embedder from config (or explicitly), so scripts and the API share one switch."""

from __future__ import annotations

from edgariq.config import Settings
from edgariq.indexing.embeddings import GeminiEmbedder, OllamaEmbedder


def build_embedder(cfg: Settings | type[Settings], provider: str | None = None):
    provider = (provider or cfg.EMBEDDING_PROVIDER).lower()
    if provider == "ollama":
        return OllamaEmbedder(base_url=cfg.OLLAMA_BASE_URL, model=cfg.OLLAMA_EMBEDDING_MODEL)
    if provider == "gemini":
        return GeminiEmbedder(api_key=cfg.GEMINI_API_KEY)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {provider!r} -- use 'ollama' or 'gemini'.")


def embedder_for_store(cfg, store):
    """The embedder that matches an existing index, so query vectors live in the
    same space as the stored ones. Indexes built before embedder ids were
    recorded (None) came from Ollama."""
    recorded = store.embedder_id or "ollama:"
    return build_embedder(cfg, provider=recorded.split(":", 1)[0])
