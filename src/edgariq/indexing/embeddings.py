"""Client for generating embeddings via a local Ollama server.

Ollama exposes a simple REST API on localhost (no API key needed since it's
your own machine). This client is intentionally thin — one responsibility,
one clear error message if Ollama isn't running.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque

import requests


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self._base_url = base_url.rstrip("/")
        self._model = model

    @property
    def model_id(self) -> str:
        """Identifies the embedding space. Stored in saved indexes so a
        server can refuse to query an index built with a different one."""
        return f"ollama:{self._model}"

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




GEMINI_BATCH_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"


class EmbeddingQuotaError(RuntimeError):
    """The embedding API refused the request because of a quota. `daily` is
    True when it is a per-day cap (retrying today is pointless)."""

    def __init__(self, message: str, daily: bool = False):
        super().__init__(message)
        self.daily = daily


def _parse_quota_error(resp) -> tuple[str, bool, float | None]:
    """(human message, is_daily_cap, server-suggested retry delay in seconds)."""
    message, daily, delay = resp.text[:300], False, None
    try:
        err = resp.json().get("error", {})
        message = err.get("message", message)
        for d in err.get("details", []):
            if str(d.get("retryDelay", "")).endswith("s"):
                delay = float(str(d["retryDelay"])[:-1])
            for v in d.get("violations", []):
                if "perday" in str(v.get("quotaId", "")).replace("_", "").lower():
                    daily = True
        if re.search(r"per day|daily", message, re.I):
            daily = True
    except (ValueError, AttributeError):
        pass
    retry_after = resp.headers.get("Retry-After")
    if delay is None and retry_after and retry_after.isdigit():
        delay = float(retry_after)
    return message, daily, delay


class GeminiEmbedder:
    """Hosted embeddings via the Google Gemini API (free key from AI Studio,
    works with a personal Google account).

    Exists because a deployed API (e.g. on Render) can't run Ollama.
    `embed()` is for queries (RETRIEVAL_QUERY) and `embed_batch()` for
    indexing documents (RETRIEVAL_DOCUMENT). Vectors are truncated to 768
    dims and re-normalised (the API only pre-normalises the full 3072).
    An index built with this class is NOT compatible with an Ollama-built
    one -- rebuild when switching.

    The free tier has per-minute request/token limits and a per-day cap, so
    indexing is paced by an estimated tokens-per-minute budget, honours the
    server's retry delay on 429, and stops with a clear message on a daily
    cap. Pair it with CachingEmbedder so an interrupted build resumes.
    """

    MODEL = "gemini-embedding-001"
    DIMENSIONS = 768

    def __init__(
        self,
        api_key: str,
        batch_size: int = 10,
        url_template: str = GEMINI_BATCH_URL,
        max_retries: int = 6,
        max_tokens_per_minute: int = 20000,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Create a free key at https://aistudio.google.com/apikey "
                "and add it to your environment."
            )
        self._api_key = api_key
        self._batch_size = batch_size
        self._url = url_template.format(model=self.MODEL)
        self._max_retries = max_retries
        self._tpm = max_tokens_per_minute  # 0 disables pacing
        self._sleep = sleep
        self._clock = clock
        self._spent: deque[tuple[float, int]] = deque()  # (time, est. tokens)
        # The free tier allows ~1,000 embedding requests/day, so never pay twice
        # for the same query text (the planner and repeat visitors repeat a lot).
        self._query_cache: dict[str, list[float]] = {}

    @property
    def model_id(self) -> str:
        return f"gemini:{self.MODEL}:{self.DIMENSIONS}"

    @staticmethod
    def _normalise(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _pace(self, texts: list[str]) -> None:
        """Wait until sending these texts keeps us under the tokens/minute budget."""
        if not self._tpm:
            return
        need = sum(max(1, len(t) // 3) for t in texts)  # rough over-estimate of tokens
        while True:
            now = self._clock()
            while self._spent and now - self._spent[0][0] >= 60:
                self._spent.popleft()
            used = sum(n for _, n in self._spent)
            if used + need <= self._tpm or not self._spent:
                break
            self._sleep(max(self._spent[0][0] + 60 - now, 0.5))
        self._spent.append((self._clock(), need))

    def _post(self, texts: list[str], task_type: str) -> list[list[float]]:
        self._pace(texts)
        body = {
            "requests": [
                {
                    "model": f"models/{self.MODEL}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task_type,
                    "outputDimensionality": self.DIMENSIONS,
                }
                for t in texts
            ]
        }
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            try:
                resp = requests.post(
                    self._url, headers={"x-goog-api-key": self._api_key}, json=body, timeout=60
                )
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                # Dropped connections happen on long indexing runs; retry, don't die.
                if last:
                    raise ConnectionError(
                        f"Could not reach the Gemini embedding API after {self._max_retries} retries."
                    ) from e
                self._sleep(min(2 ** attempt * 2, 30))
                continue
            if resp.status_code >= 500 and not last:
                self._sleep(min(2 ** attempt * 2, 30))
                continue
            if resp.status_code != 429:
                break
            message, daily, delay = _parse_quota_error(resp)
            if daily:
                raise EmbeddingQuotaError(
                    f"Gemini daily quota reached: {message} Re-run the same command later "
                    "(tomorrow, if it is a per-day cap) -- finished embeddings are cached.",
                    daily=True,
                )
            if last:
                raise EmbeddingQuotaError(
                    f"Gemini rate limit persisted after {self._max_retries} retries: {message}"
                )
            self._sleep(delay if delay is not None else min(2 ** attempt * 2, 60))
        if resp.status_code in (400, 401, 403):
            raise PermissionError(
                f"Gemini API rejected the request ({resp.status_code}): {resp.text[:200]}"
            )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise ValueError(f"Unexpected Gemini response shape: {str(data)[:200]}")
        return [self._normalise(e["values"]) for e in embeddings]

    def embed(self, text: str) -> list[float]:
        """Embed a search QUERY (memoised in memory)."""
        key = text.strip().lower()
        if key not in self._query_cache:
            if len(self._query_cache) >= 1000:
                self._query_cache.pop(next(iter(self._query_cache)))  # drop oldest
            self._query_cache[key] = self._post([text], "RETRIEVAL_QUERY")[0]
        return self._query_cache[key]

    def embed_batch(self, texts: list[str], on_progress=None) -> list[list[float]]:
        """Embed DOCUMENTS for indexing, in batches."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._post(texts[start : start + self._batch_size], "RETRIEVAL_DOCUMENT"))
            if on_progress:
                on_progress(len(vectors), len(texts))
        return vectors
