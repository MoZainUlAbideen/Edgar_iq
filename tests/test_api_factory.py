import time
import types

import pytest
import responses
from fastapi.testclient import TestClient

from edgariq.api.factory import build_app
from edgariq.indexing import Chunk, VectorStore
from edgariq.indexing.embeddings import GEMINI_BATCH_URL
from edgariq.llm.groq_client import GROQ_CHAT_URL


def _cfg(index_path, provider="gemini"):
    return types.SimpleNamespace(
        API_INDEX_PATH=str(index_path), EMBEDDING_PROVIDER=provider, GEMINI_API_KEY="nk",
        OLLAMA_BASE_URL="http://localhost:11434", OLLAMA_EMBEDDING_MODEL="nomic-embed-text",
        GROQ_API_KEY="gk", GROQ_MODEL="m", GROQ_MIN_REQUEST_INTERVAL_SECONDS=0,
        API_RATE_LIMIT_PER_HOUR=5, API_DAILY_CAP=50, ALLOWED_ORIGINS="https://a.example",
        API_TICKER="NVDA", API_MAX_QUEUE=2,
    )


def _index(tmp_path, embedder_id):
    store = VectorStore(embedder_id=embedder_id)
    store.add(
        [Chunk(text="Data Center revenue was $75.2 billion, up 92% from a year ago.",
               chunk_type="text",
               metadata={"form_type": "10-Q", "filing_date": "2026-05-20", "source_url": "https://s"})],
        [[1.0, 0.0]],
    )
    path = tmp_path / "NVDA.json.gz"
    store.save(path)
    return path


def test_missing_index_fails_with_actionable_message(tmp_path):
    with pytest.raises(RuntimeError, match="package_index"):
        build_app(_cfg(tmp_path / "nope.json.gz"))


def test_embedder_mismatch_refuses_to_start(tmp_path):
    path = _index(tmp_path, "ollama:nomic-embed-text")
    with pytest.raises(RuntimeError, match="Rebuild the index"):
        build_app(_cfg(path))


@responses.activate
def test_real_wiring_answers_a_question_end_to_end_with_mocked_services(tmp_path):
    path = _index(tmp_path, "gemini:gemini-embedding-001:768")
    responses.add(responses.POST, GEMINI_BATCH_URL.format(model="gemini-embedding-001"), json={"embeddings": [{"values": [1.0, 0.0]}]})
    llm_replies = iter([
        {"choices": [{"message": {"content": "data center revenue"}}]},
        {"choices": [{"message": {"content": "Data Center revenue was $75.2 billion, up 92%."}}]},
        {"choices": [{"message": {"content": "APPROVED"}}]},
    ])
    responses.add_callback(
        responses.POST, GROQ_CHAT_URL,
        callback=lambda req: (200, {}, __import__("json").dumps(next(llm_replies))),
        content_type="application/json",
    )

    client = TestClient(build_app(_cfg(path)))
    assert client.get("/health").json()["embedder"] == "gemini:gemini-embedding-001:768"

    job = client.post("/ask", json={"question": "What was data center revenue?"}).json()
    deadline = time.time() + 5
    while time.time() < deadline:
        body = client.get(f"/ask/{job['job_id']}").json()
        if body["status"] in ("done", "error"):
            break
        time.sleep(0.02)
    assert body["status"] == "done", body
    assert "$75.2 billion" in body["answer"]
    assert body["citations"][0]["filing_date"] == "2026-05-20"
    assert body["warnings"] == []
