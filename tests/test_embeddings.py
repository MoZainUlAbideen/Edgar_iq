import pytest
import responses
from requests.exceptions import ConnectionError as RequestsConnectionError

from edgariq.indexing.embeddings import OllamaEmbedder

BASE_URL = "http://localhost:11434"


@responses.activate
def test_embed_returns_vector_from_ollama_response():
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/embeddings",
        json={"embedding": [0.1, 0.2, 0.3]},
        status=200,
    )
    embedder = OllamaEmbedder(base_url=BASE_URL, model="nomic-embed-text")

    vector = embedder.embed("NVIDIA data center revenue")

    assert vector == [0.1, 0.2, 0.3]


@responses.activate
def test_embed_sends_correct_model_and_prompt():
    responses.add(
        responses.POST, f"{BASE_URL}/api/embeddings", json={"embedding": [0.0]}, status=200
    )
    embedder = OllamaEmbedder(base_url=BASE_URL, model="nomic-embed-text")

    embedder.embed("test text")

    sent_body = responses.calls[0].request.body
    assert b'"model": "nomic-embed-text"' in sent_body
    assert b'"prompt": "test text"' in sent_body


@responses.activate
def test_embed_raises_clear_error_when_ollama_unreachable():
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/embeddings",
        body=RequestsConnectionError("connection refused"),
    )
    embedder = OllamaEmbedder(base_url=BASE_URL, model="nomic-embed-text")

    with pytest.raises(ConnectionError, match="Is it running"):
        embedder.embed("test")


@responses.activate
def test_embed_raises_on_missing_embedding_key():
    responses.add(
        responses.POST, f"{BASE_URL}/api/embeddings", json={"error": "model not found"}, status=200
    )
    embedder = OllamaEmbedder(base_url=BASE_URL, model="nomic-embed-text")

    with pytest.raises(ValueError, match="Unexpected Ollama response"):
        embedder.embed("test")


@responses.activate
def test_embed_batch_calls_progress_callback():
    responses.add(
        responses.POST, f"{BASE_URL}/api/embeddings", json={"embedding": [1.0]}, status=200
    )
    embedder = OllamaEmbedder(base_url=BASE_URL, model="nomic-embed-text")
    progress_calls = []

    embedder.embed_batch(["a", "b", "c"], on_progress=lambda i, total: progress_calls.append((i, total)))

    assert progress_calls == [(1, 3), (2, 3), (3, 3)]
