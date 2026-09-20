"""Client for Groq's chat completions API (OpenAI-compatible endpoint).

Kept deliberately thin — one method, one job — so agents depend on a
simple `complete(system, user) -> str` interface rather than the Groq SDK
directly. That makes every agent trivially testable with a fake LLM that
returns canned strings, no network or API key needed in tests.

Two layers of rate-limit defense, learned the hard way running the real
eval harness (7 golden-set cases x 3-4 calls each = ~25 calls in quick
succession):

1. Proactive throttling — space consecutive calls at least
   `min_request_interval` seconds apart, BEFORE hitting a 429 at all.
   Reactive retry alone (below) turned out not to be enough: exponential
   backoff can win back a single rate-limited call, but a pipeline that
   keeps firing calls immediately after each retry just gets rate-limited
   again on the next one — the eval run kept losing the race against
   Groq's free-tier requests-per-minute cap even with retries. Spacing
   calls out up front avoids the problem instead of just reacting to it.

2. Retry-with-backoff on 429 — a second line of defense for the rare case
   throttling alone doesn't avoid (e.g. another process sharing the same
   API key).
"""

from __future__ import annotations

import time

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqRateLimitError(Exception):
    """Raised internally on a 429 response — caught and retried automatically
    by the @retry decorator on _complete_with_retry, never meant to surface
    to a caller except after retries are exhausted."""


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        min_request_interval: float = 2.5,
    ):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._min_request_interval = min_request_interval
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)

    def complete(self, system: str, user: str) -> str:
        return self._complete_with_retry(system, user)

    @retry(
        retry=retry_if_exception_type(GroqRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _complete_with_retry(self, system: str, user: str) -> str:
        self._throttle()
        try:
            resp = requests.post(
                GROQ_CHAT_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "temperature": self._temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=30,
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("Could not reach Groq's API. Check your internet connection.") from e
        finally:
            # Recorded even on failure, so a 429 or error still counts
            # toward the spacing before the next attempt (including a retry
            # of this same call).
            self._last_request_time = time.monotonic()

        if resp.status_code == 429:
            # NOTE: a fuller implementation would read the Retry-After header
            # and wait exactly that long; exponential backoff is a
            # deliberate simplification that works well enough in practice
            # without needing to parse rate-limit headers.
            raise GroqRateLimitError(
                "Groq rate-limited this request (429) after retrying with backoff. "
                "Free/developer tier has a requests-per-minute cap — wait a bit and "
                "try again, or reduce how many chunks/questions you send per minute."
            )

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 401:
                raise ValueError("Groq rejected the API key — check GROQ_API_KEY in .env.") from e
            raise

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected Groq response shape: {data}") from e