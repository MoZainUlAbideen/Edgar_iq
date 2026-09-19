import pytest
import responses

from edgariq.llm.groq_client import GROQ_CHAT_URL, GroqClient, GroqRateLimitError


def test_rejects_empty_api_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqClient(api_key="", model="llama-3.3-70b-versatile")


@responses.activate
def test_complete_returns_message_content():
    responses.add(
        responses.POST,
        GROQ_CHAT_URL,
        json={"choices": [{"message": {"content": "The answer is 42."}}]},
        status=200,
    )
    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")

    result = client.complete(system="You are helpful.", user="What is the answer?")

    assert result == "The answer is 42."


@responses.activate
def test_complete_sends_auth_header_and_model():
    responses.add(
        responses.POST, GROQ_CHAT_URL, json={"choices": [{"message": {"content": "ok"}}]}, status=200
    )
    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")

    client.complete(system="sys", user="usr")

    sent = responses.calls[0].request
    assert sent.headers["Authorization"] == "Bearer fake-key"
    assert b'"model": "llama-3.3-70b-versatile"' in sent.body


@responses.activate
def test_complete_raises_clear_error_on_bad_api_key():
    responses.add(responses.POST, GROQ_CHAT_URL, json={"error": "invalid api key"}, status=401)
    client = GroqClient(api_key="wrong-key", model="llama-3.3-70b-versatile")

    with pytest.raises(ValueError, match="rejected the API key"):
        client.complete(system="sys", user="usr")


@responses.activate
def test_complete_raises_on_unexpected_response_shape():
    responses.add(responses.POST, GROQ_CHAT_URL, json={"unexpected": "shape"}, status=200)
    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")

    with pytest.raises(ValueError, match="Unexpected Groq response"):
        client.complete(system="sys", user="usr")


@responses.activate
def test_complete_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)  # skip real backoff delay in tests
    responses.add(responses.POST, GROQ_CHAT_URL, status=429)
    responses.add(
        responses.POST,
        GROQ_CHAT_URL,
        json={"choices": [{"message": {"content": "recovered after retry"}}]},
        status=200,
    )
    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")

    result = client.complete(system="sys", user="usr")

    assert result == "recovered after retry"
    assert len(responses.calls) == 2


@responses.activate
def test_complete_gives_up_after_max_retries_on_persistent_429(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    for _ in range(5):
        responses.add(responses.POST, GROQ_CHAT_URL, status=429)
    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")

    with pytest.raises(GroqRateLimitError, match="rate-limited"):
        client.complete(system="sys", user="usr")

    assert len(responses.calls) == 5  # stop_after_attempt(5) — no more, no fewer