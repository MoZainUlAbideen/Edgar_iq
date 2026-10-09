import json
import math

import pytest
import responses

from edgariq.indexing import GeminiEmbedder, VectorStore, build_embedder
from edgariq.indexing.embeddings import GEMINI_BATCH_URL, EmbeddingQuotaError, OllamaEmbedder
from edgariq.indexing.models import Chunk

URL = GEMINI_BATCH_URL.format(model="gemini-embedding-001")


def _body(call):
    return json.loads(call.request.body)


def _reply(*vectors):
    return {"embeddings": [{"values": v} for v in vectors]}


def test_requires_api_key():
    with pytest.raises(ValueError):
        GeminiEmbedder(api_key="")


@responses.activate
def test_embed_uses_query_task_dimensions_auth_header_and_normalises():
    responses.add(responses.POST, URL, json=_reply([3.0, 4.0]))
    vec = GeminiEmbedder(api_key="k").embed("hello")
    call = responses.calls[0]
    req = _body(call)["requests"][0]
    assert req["taskType"] == "RETRIEVAL_QUERY"
    assert req["outputDimensionality"] == 768
    assert req["content"]["parts"][0]["text"] == "hello"
    assert call.request.headers["x-goog-api-key"] == "k"
    assert vec == pytest.approx([0.6, 0.8])
    assert math.isclose(sum(v * v for v in vec), 1.0)


@responses.activate
def test_embed_batch_uses_document_task_and_splits_into_batches():
    responses.add(responses.POST, URL, json=_reply([1.0], [1.0]))
    responses.add(responses.POST, URL, json=_reply([1.0]))
    progress = []
    out = GeminiEmbedder(api_key="k", batch_size=2).embed_batch(
        ["a", "b", "c"], on_progress=lambda i, n: progress.append((i, n))
    )
    assert len(out) == 3
    assert [len(_body(c)["requests"]) for c in responses.calls] == [2, 1]
    assert all(
        r["taskType"] == "RETRIEVAL_DOCUMENT" for c in responses.calls for r in _body(c)["requests"]
    )
    assert progress == [(2, 3), (3, 3)]


@responses.activate
def test_retries_on_429_then_succeeds():
    responses.add(responses.POST, URL, status=429)
    responses.add(responses.POST, URL, json=_reply([1.0, 0.0]))
    waits = []
    vec = GeminiEmbedder(api_key="k", sleep=waits.append).embed("x")
    assert vec == [1.0, 0.0]
    assert len(waits) == 1 and len(responses.calls) == 2


@responses.activate
def test_gives_up_after_max_retries_on_persistent_429():
    responses.add(responses.POST, URL, status=429)
    with pytest.raises(EmbeddingQuotaError):
        GeminiEmbedder(api_key="k", max_retries=2, sleep=lambda s: None).embed("x")
    assert len(responses.calls) == 3


@responses.activate
def test_bad_key_raises_clear_error():
    responses.add(responses.POST, URL, status=403)
    with pytest.raises(PermissionError):
        GeminiEmbedder(api_key="bad").embed("x")


@responses.activate
def test_mismatched_embedding_count_is_rejected():
    responses.add(responses.POST, URL, json={"embeddings": []})
    with pytest.raises(ValueError):
        GeminiEmbedder(api_key="k").embed("x")


def test_model_ids_differ_between_providers():
    assert GeminiEmbedder(api_key="k").model_id != OllamaEmbedder("http://x", "m").model_id


def test_factory_selects_provider_and_rejects_unknown():
    class Cfg:
        EMBEDDING_PROVIDER = "gemini"
        GEMINI_API_KEY = "k"
        OLLAMA_BASE_URL = "http://x"
        OLLAMA_EMBEDDING_MODEL = "m"

    assert isinstance(build_embedder(Cfg), GeminiEmbedder)
    Cfg.EMBEDDING_PROVIDER = "ollama"
    assert isinstance(build_embedder(Cfg), OllamaEmbedder)
    Cfg.EMBEDDING_PROVIDER = "other"
    with pytest.raises(ValueError):
        build_embedder(Cfg)


@pytest.mark.parametrize("name", ["idx.json", "idx.json.gz"])
def test_store_round_trips_embedder_id_for_plain_and_gzip(tmp_path, name):
    store = VectorStore(embedder_id="gemini:gemini-embedding-001:768")
    store.add([Chunk(text="t", chunk_type="text", metadata={"a": "b"})], [[1.0, 0.0]])
    path = tmp_path / name
    store.save(path)
    loaded = VectorStore.load(path)
    assert loaded.embedder_id == "gemini:gemini-embedding-001:768"
    assert loaded.search([1.0, 0.0], top_k=1)[0][0].text == "t"


def test_old_index_without_embedder_field_still_loads(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"chunks": [], "vectors": []}), encoding="utf-8")
    assert VectorStore.load(path).embedder_id is None


# --- quota handling, pacing and caching -------------------------------------

from edgariq.indexing import CachingEmbedder


def _quota_body(message="Quota exceeded", quota_id="EmbedContentRequestsPerMinute", delay="7s"):
    return {"error": {"code": 429, "message": message, "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
         "violations": [{"quotaId": quota_id}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
    ]}}


@responses.activate
def test_429_honours_server_retry_delay():
    responses.add(responses.POST, URL, status=429, json=_quota_body(delay="7s"))
    responses.add(responses.POST, URL, json=_reply([1.0, 0.0]))
    waits = []
    GeminiEmbedder(api_key="k", sleep=waits.append).embed("x")
    assert waits == [7.0]


@responses.activate
def test_daily_quota_fails_fast_with_clear_message():
    responses.add(responses.POST, URL, status=429, json=_quota_body(
        message="Quota exceeded for metric", quota_id="EmbedContentRequestsPerDayPerProjectPerModel-FreeTier"))
    with pytest.raises(EmbeddingQuotaError) as exc:
        GeminiEmbedder(api_key="k", sleep=lambda s: None).embed("x")
    assert exc.value.daily is True
    assert "cached" in str(exc.value)
    assert len(responses.calls) == 1  # did not pointlessly retry


@responses.activate
def test_persistent_minute_limit_raises_quota_error_with_api_message():
    responses.add(responses.POST, URL, status=429, json=_quota_body(message="slow down"))
    with pytest.raises(EmbeddingQuotaError, match="slow down"):
        GeminiEmbedder(api_key="k", max_retries=1, sleep=lambda s: None).embed("x")


@responses.activate
def test_token_pacing_waits_when_budget_is_spent():
    responses.add(responses.POST, URL, json=_reply([1.0]), )
    responses.add(responses.POST, URL, json=_reply([1.0]), )
    now = [0.0]
    waits = []

    def sleep(s):
        waits.append(s)
        now[0] += s

    emb = GeminiEmbedder(api_key="k", batch_size=1, max_tokens_per_minute=100,
                         sleep=sleep, clock=lambda: now[0])
    emb.embed_batch(["a" * 240, "b" * 240])  # ~80 est. tokens each -> second must wait
    assert waits and waits[0] > 0


class _FlakyEmbedder:
    model_id = "fake:model"

    def __init__(self, fail_after):
        self.calls, self.fail_after = [], fail_after

    def embed(self, text):
        return [1.0]

    def embed_batch(self, texts, on_progress=None):
        if len(self.calls) >= self.fail_after:
            raise RuntimeError("quota")
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


def test_cache_resumes_after_failure_without_re_embedding(tmp_path):
    texts = ["aa", "bbb", "c", "dddd"]
    first = CachingEmbedder(_FlakyEmbedder(fail_after=1), tmp_path, chunk_size=2)
    with pytest.raises(RuntimeError):
        first.embed_batch(texts)
    assert first.cached_count == 2  # first chunk was saved before the crash

    inner = _FlakyEmbedder(fail_after=99)
    second = CachingEmbedder(inner, tmp_path, chunk_size=2)  # new process, reads the file
    progress = []
    out = second.embed_batch(texts, on_progress=lambda i, n: progress.append(i))
    assert out == [[2.0], [3.0], [1.0], [4.0]]
    assert inner.calls == [["c", "dddd"]]  # only the missing ones were embedded
    assert progress[-1] == 4


def test_cache_survives_corrupt_trailing_line_and_keeps_models_separate(tmp_path):
    inner = _FlakyEmbedder(fail_after=99)
    CachingEmbedder(inner, tmp_path).embed_batch(["x"])
    f = next(tmp_path.glob("*.jsonl"))
    f.write_text(f.read_text(encoding="utf-8") + '{"k": "broken', encoding="utf-8")
    again = CachingEmbedder(_FlakyEmbedder(99), tmp_path)
    assert again.cached_count == 1

    other = _FlakyEmbedder(99)
    other.model_id = "other:model"
    assert CachingEmbedder(other, tmp_path).cached_count == 0


def test_embedder_for_store_follows_the_index_not_the_env():
    from edgariq.indexing import embedder_for_store

    class Cfg:
        EMBEDDING_PROVIDER = "ollama"  # env says ollama...
        GEMINI_API_KEY = "k"
        OLLAMA_BASE_URL = "http://x"
        OLLAMA_EMBEDDING_MODEL = "m"

    gemini_store = VectorStore(embedder_id="gemini:gemini-embedding-001:768")
    assert isinstance(embedder_for_store(Cfg, gemini_store), GeminiEmbedder)  # ...index wins
    assert isinstance(embedder_for_store(Cfg, VectorStore(embedder_id="ollama:nomic-embed-text")), OllamaEmbedder)
    assert isinstance(embedder_for_store(Cfg, VectorStore()), OllamaEmbedder)  # legacy index


@responses.activate
def test_dropped_connection_is_retried_then_succeeds():
    from requests.exceptions import ConnectionError as RequestsConnectionError

    responses.add(responses.POST, URL, body=RequestsConnectionError("RemoteDisconnected"))
    responses.add(responses.POST, URL, json=_reply([1.0, 0.0]))
    waits = []
    assert GeminiEmbedder(api_key="k", sleep=waits.append).embed("x") == [1.0, 0.0]
    assert len(waits) == 1


@responses.activate
def test_persistent_connection_failure_raises_connection_error_after_retries():
    from requests.exceptions import ConnectionError as RequestsConnectionError

    responses.add(responses.POST, URL, body=RequestsConnectionError("down"))
    with pytest.raises(ConnectionError, match="retries"):
        GeminiEmbedder(api_key="k", max_retries=2, sleep=lambda s: None).embed("x")
    assert len(responses.calls) == 3


@responses.activate
def test_server_error_503_is_retried():
    responses.add(responses.POST, URL, status=503)
    responses.add(responses.POST, URL, json=_reply([1.0, 0.0]))
    assert GeminiEmbedder(api_key="k", sleep=lambda s: None).embed("x") == [1.0, 0.0]
    assert len(responses.calls) == 2


@responses.activate
def test_repeated_query_is_served_from_memory_not_the_api():
    responses.add(responses.POST, URL, json=_reply([1.0, 0.0]))
    emb = GeminiEmbedder(api_key="k")
    a = emb.embed("Data center revenue")
    b = emb.embed("  data center REVENUE ")
    assert a == b
    assert len(responses.calls) == 1  # second call cost no quota
