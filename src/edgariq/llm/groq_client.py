"""Client for Groq's chat completions API (OpenAI-compatible endpoint).

Kept deliberately thin — one method, one job — so agents depend on a
simple `complete(system, user) -> str` interface rather than the Groq SDK
directly. That makes every agent trivially testable with a fake LLM that
returns canned strings, no network or API key needed in tests.

Includes automatic retry-with-backoff on 429 (rate limit) responses. This
matters in practice, not just in theory: a single question runs 3+ LLM
calls (planner, drafter, critic), and free/developer-tier API keys have a
requests-per-minute cap that a multi-agent pipeline can trip easily —
without this, the whole pipeline crashes on the 3rd call of an otherwise
successful run.
"""

from __future__ import annotations

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqRateLimitError(Exception):
    """Raised internally on a 429 response — caught and retried automatically
    by the @retry decorator on _complete_with_retry, never meant to surface
    to a caller except after retries are exhausted."""


class GroqClient:
    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and add it to your .env file."
            )
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        return self._complete_with_retry(system, user)

    @retry(
        retry=retry_if_exception_type(GroqRateLimitError),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _complete_with_retry(self, system: str, user: str) -> str:
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