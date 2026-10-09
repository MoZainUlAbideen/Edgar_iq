"""Tiny in-memory abuse protection for a public demo.

Per-client sliding window + a global daily cap. State lives in process
memory, which is fine for a single-instance demo (it resets on restart/
redeploy; documented in the README). The client key comes from
X-Forwarded-For, which a determined caller can spoof -- the global daily
cap is the real backstop that protects the Groq quota.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable


class RateLimitExceeded(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimiter:
    def __init__(
        self,
        per_client_limit: int,
        per_client_window: float = 3600,
        daily_cap: int = 150,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._limit = per_client_limit
        self._window = per_client_window
        self._daily_cap = daily_cap
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._global: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, client: str) -> None:
        """Record one request or raise RateLimitExceeded."""
        now = self._clock()
        with self._lock:
            day = self._global
            while day and now - day[0] > 86400:
                day.popleft()
            if len(day) >= self._daily_cap:
                raise RateLimitExceeded(
                    "The public demo has hit its daily question limit. Please come back tomorrow.",
                    retry_after=int(86400 - (now - day[0])),
                )

            q = self._events[client]
            while q and now - q[0] > self._window:
                q.popleft()
            if len(q) >= self._limit:
                raise RateLimitExceeded(
                    f"Limit of {self._limit} questions per hour reached. Please try again later.",
                    retry_after=int(self._window - (now - q[0])),
                )

            q.append(now)
            day.append(now)
