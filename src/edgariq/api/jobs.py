"""Run slow pipeline calls in the background and let clients poll.

A full question takes ~15-60s (several sequential LLM calls), and a free
Render instance may also be cold-starting. Holding one HTTP request open
that long is fragile (proxy/idle timeouts), so POST /ask just enqueues a
job and the browser polls GET /ask/{id}.

One worker thread on purpose: GroqClient throttles per instance, and the
free Groq tier's rate limit is shared by every visitor. Serialising jobs
keeps us under it; `max_pending` rejects overflow instead of letting an
unbounded queue build up.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable


class QueueFull(Exception):
    pass


@dataclass
class Job:
    id: str
    question: str
    status: str = "queued"  # queued | running | done | error
    result: dict[str, Any] | None = None
    error: str | None = None
    created: float = field(default_factory=time.monotonic)
    finished: float | None = None


class JobManager:
    def __init__(
        self,
        runner: Callable[[str], dict[str, Any]],
        max_pending: int = 4,
        ttl_seconds: float = 900,
    ):
        self._runner = runner
        self._max_pending = max_pending
        self._ttl = ttl_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="edgariq-job")

    def _pending_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def _evict_old(self) -> None:
        now = time.monotonic()
        stale = [
            jid for jid, j in self._jobs.items()
            if j.finished is not None and now - j.finished > self._ttl
        ]
        for jid in stale:
            del self._jobs[jid]

    def submit(self, question: str) -> Job:
        with self._lock:
            self._evict_old()
            if self._pending_count() >= self._max_pending:
                raise QueueFull()
            job = Job(id=uuid.uuid4().hex, question=question)
            self._jobs[job.id] = job
        self._pool.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int:
        """0 = running now; n = n jobs ahead of it."""
        with self._lock:
            ahead = [
                j for j in self._jobs.values()
                if j.status in ("queued", "running") and j.created < job.created
            ]
            return len(ahead) if job.status == "queued" else 0

    def _run(self, job: Job) -> None:
        job.status = "running"
        try:
            job.result = self._runner(job.question)
            job.status = "done"
        except Exception as e:  # surfaced to the client as a short message, not a stack trace
            job.error = _friendly_error(e)
            job.status = "error"
        finally:
            job.finished = time.monotonic()


def _friendly_error(e: Exception) -> str:
    name = type(e).__name__
    if name == "EmbeddingQuotaError":
        return "The demo's search quota is used up for now. Please try again later today, or use the recorded answers."
    if "RateLimit" in name:
        return "The language model is rate-limited right now. Please try again in a minute."
    if isinstance(e, ConnectionError):
        return "A backend dependency (LLM or embeddings) couldn't be reached. Please try again shortly."
    return "Something went wrong while answering. Please try again."
