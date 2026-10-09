import time

import pytest
from fastapi.testclient import TestClient

from edgariq.api.app import create_app
from edgariq.api.rate_limit import RateLimiter

ORIGIN = "https://edgar-iq-web.vercel.app"


def _app(runner=None, limit=100, daily=1000, max_pending=4):
    runner = runner or (lambda q: {"answer": f"echo: {q}", "citations": [], "warnings": []})
    return create_app(
        runner=runner,
        limiter=RateLimiter(per_client_limit=limit, daily_cap=daily),
        allowed_origins=[ORIGIN],
        index_info={"chunks": 3},
        max_pending=max_pending,
    )


def _wait_done(client, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/ask/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health_reports_ok_and_index_info():
    body = TestClient(_app()).get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] == 3


def test_ask_then_poll_returns_answer_citations_and_warnings():
    runner = lambda q: {
        "answer": "Revenue was $75.2 billion.",
        "citations": [{"form_type": "10-Q", "filing_date": "2026-05-20", "source_url": "https://x"}],
        "warnings": ["something"],
    }
    client = TestClient(_app(runner))
    r = client.post("/ask", json={"question": "What was data center revenue?"})
    assert r.status_code == 202
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert body["answer"].startswith("Revenue")
    assert body["citations"][0]["filing_date"] == "2026-05-20"
    assert body["warnings"] == ["something"]


def test_runner_exception_becomes_friendly_error_without_leaking_details():
    def boom(q):
        raise RuntimeError("secret internal path /srv/x")

    client = TestClient(_app(boom))
    job = client.post("/ask", json={"question": "anything goes here"}).json()
    body = _wait_done(client, job["job_id"])
    assert body["status"] == "error"
    assert "secret" not in body["error"]


@pytest.mark.parametrize("question", ["", "hi", "x" * 501])
def test_rejects_too_short_or_too_long_questions(question):
    r = TestClient(_app()).post("/ask", json={"question": question})
    assert r.status_code == 422


def test_unknown_job_is_404():
    assert TestClient(_app()).get("/ask/nope").status_code == 404


def test_per_client_rate_limit_returns_429_with_retry_after():
    client = TestClient(_app(limit=2))
    q = {"question": "What was revenue?"}
    assert client.post("/ask", json=q).status_code == 202
    assert client.post("/ask", json=q).status_code == 202
    r = client.post("/ask", json=q)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) > 0


def test_rate_limit_is_per_client_via_forwarded_for():
    client = TestClient(_app(limit=1))
    q = {"question": "What was revenue?"}
    assert client.post("/ask", json=q, headers={"x-forwarded-for": "1.1.1.1"}).status_code == 202
    assert client.post("/ask", json=q, headers={"x-forwarded-for": "2.2.2.2"}).status_code == 202
    assert client.post("/ask", json=q, headers={"x-forwarded-for": "1.1.1.1"}).status_code == 429


def test_daily_cap_applies_across_clients():
    client = TestClient(_app(limit=10, daily=2))
    q = {"question": "What was revenue?"}
    for ip in ("1.1.1.1", "2.2.2.2"):
        assert client.post("/ask", json=q, headers={"x-forwarded-for": ip}).status_code == 202
    assert client.post("/ask", json=q, headers={"x-forwarded-for": "3.3.3.3"}).status_code == 429


def test_full_queue_returns_503():
    import threading

    gate = threading.Event()
    client = TestClient(_app(lambda q: (gate.wait(5), {"answer": "a", "citations": [], "warnings": []})[1], max_pending=1))
    q = {"question": "What was revenue?"}
    assert client.post("/ask", json=q).status_code == 202
    r = client.post("/ask", json=q)
    assert r.status_code == 503
    gate.set()


def test_cors_allows_configured_origin_only():
    client = TestClient(_app())
    ok = client.options(
        "/ask",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "content-type"},
    )
    assert ok.headers.get("access-control-allow-origin") == ORIGIN
    bad = client.options(
        "/ask",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


def test_rate_limiter_window_expires():
    now = [0.0]
    lim = RateLimiter(per_client_limit=1, per_client_window=60, clock=lambda: now[0])
    lim.check("a")
    with pytest.raises(Exception):
        lim.check("a")
    now[0] = 61
    lim.check("a")  # window passed -> allowed again


def test_embedding_quota_error_gets_a_specific_friendly_message():
    from edgariq.indexing.embeddings import EmbeddingQuotaError

    def runner(q):
        raise EmbeddingQuotaError("daily", daily=True)

    client = TestClient(_app(runner))
    job = client.post("/ask", json={"question": "anything goes here"}).json()
    body = _wait_done(client, job["job_id"])
    assert body["status"] == "error"
    assert "quota" in body["error"].lower()
