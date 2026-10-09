"""FastAPI app wrapping the EdgarIQ pipeline.

Endpoints:
  GET  /health        liveness + which index/embedder is loaded
  POST /ask           {"question": "..."} -> 202 {"job_id": ...}
  GET  /ask/{job_id}  poll: queued | running | done | error

`create_app` takes its collaborators as arguments (a runner function, a
limiter, ...) so tests can drive the whole HTTP layer with fakes -- no
Groq, no index, no network. The real wiring lives in api/main.py.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from edgariq.api.jobs import JobManager, QueueFull
from edgariq.api.rate_limit import RateLimiter, RateLimitExceeded


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(
    runner: Callable[[str], dict[str, Any]],
    limiter: RateLimiter,
    allowed_origins: list[str],
    ticker: str = "NVDA",
    index_info: dict[str, Any] | None = None,
    max_pending: int = 4,
) -> FastAPI:
    app = FastAPI(title="EdgarIQ API", version="0.1.0")
    jobs = JobManager(runner=runner, max_pending=max_pending)
    started = time.time()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "ticker": ticker,
            "uptime_seconds": int(time.time() - started),
            **(index_info or {}),
        }

    @app.post("/ask", status_code=202)
    def ask(body: AskRequest, request: Request) -> dict[str, Any]:
        try:
            limiter.check(_client_key(request))
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=429, detail=str(e), headers={"Retry-After": str(e.retry_after)}
            )
        try:
            job = jobs.submit(body.question.strip())
        except QueueFull:
            raise HTTPException(
                status_code=503,
                detail="The assistant is busy answering other questions. Please retry in a moment.",
                headers={"Retry-After": "20"},
            )
        return {"job_id": job.id, "status": job.status}

    @app.get("/ask/{job_id}")
    def poll(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown or expired job.")
        payload: dict[str, Any] = {"job_id": job.id, "status": job.status}
        if job.status == "queued":
            payload["queue_position"] = jobs.queue_position(job)
        elif job.status == "done":
            payload.update(job.result or {})
        elif job.status == "error":
            payload["error"] = job.error
        return payload

    return app
