"""Client for generating embeddings via a local Ollama server.

Ollama exposes a simple REST API on localhost (no API key needed since it's
your own machine). This client is intentionally thin — one responsibility,
one clear error message if Ollama isn't running.
"""

from __future__ import annotations

import requests


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Embed a single piece of text. Raises a clear error if Ollama
        isn't reachable, rather than letting a raw ConnectionError surface."""
        try:
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self._base_url}. Is it running? "
                "Start it and confirm with `ollama list`."
            ) from e

        data = resp.json()
        if "embedding" not in data:
            raise ValueError(f"Unexpected Ollama response, no 'embedding' key: {data}")
        return data["embedding"]

    def embed_batch(self, texts: list[str], on_progress=None) -> list[list[float]]:
        """Embed multiple texts. Ollama's embeddings endpoint doesn't batch
        server-side, so this loops — on_progress(i, total) lets a caller
        show a progress bar for large filings (hundreds of chunks)."""
        vectors = []
        for i, text in enumerate(texts):
            vectors.append(self.embed(text))
            if on_progress:
                on_progress(i + 1, len(texts))
        return vectors
