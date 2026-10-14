"""Tests for the LLM answer generator.

The Groq API is never called: requests.post is replaced by a fake, so the
tests are free, deterministic and run without an API key or network access.
"""

import pytest
import requests

import src.rag.generator as generator_module
from src.rag.generator import ReviewGenerator

RETRIEVED = [('Game: "Dread Manor" | Recommendation: Recommended | Review: Scary', 0.82)]


class FakeResponse:
    def __init__(self, status_code=200, answer="A grounded answer."):
        self.status_code = status_code
        self._answer = answer

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return {"choices": [{"message": {"content": self._answer}}]}


@pytest.fixture
def api_calls(monkeypatch):
    """Record every request the generator makes and reply with a canned answer."""
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(generator_module.requests, "post", fake_post)
    return calls


@pytest.fixture
def generator():
    return ReviewGenerator(api_key="test-key")


def test_no_retrieved_reviews_skips_the_api(generator, api_calls):
    answer = generator.generate("Best horror games?", retrieved_reviews=[])
    assert "couldn't find any relevant reviews" in answer
    assert api_calls == []


def test_successful_call_returns_model_answer(generator, api_calls):
    assert generator.generate("Best horror games?", RETRIEVED) == "A grounded answer."


def test_request_contains_system_prompt_query_and_context(generator, api_calls):
    generator.generate("Best horror games?", RETRIEVED)
    payload = api_calls[0]["json"]
    messages = payload["messages"]

    assert messages[0]["role"] == "system"
    assert "ONLY on the provided Steam reviews" in messages[0]["content"]
    assert "Best horror games?" in messages[-1]["content"]
    assert "Dread Manor" in messages[-1]["content"]
    assert payload["model"] == generator.model


def test_api_key_is_sent_as_bearer_token(generator, api_calls):
    generator.generate("Best horror games?", RETRIEVED)
    assert api_calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_request_has_a_timeout(generator, api_calls):
    generator.generate("Best horror games?", RETRIEVED)
    assert api_calls[0]["timeout"] is not None


def test_history_stores_the_question_not_the_retrieved_context(generator, api_calls):
    generator.generate("Best horror games?", RETRIEVED)
    assert generator.get_history() == [
        {"role": "user", "content": "Best horror games?"},
        {"role": "assistant", "content": "A grounded answer."},
    ]


def test_history_sent_to_the_api_is_capped_at_three_exchanges(generator, api_calls):
    for i in range(5):
        generator.generate(f"Question {i}", RETRIEVED)

    messages = api_calls[-1]["json"]["messages"]
    # system prompt + last 3 exchanges (6 messages) + the new question
    assert len(messages) == 8
    assert messages[1]["content"] == "Question 1"


def test_history_can_be_disabled(generator, api_calls):
    generator.generate("First question", RETRIEVED)
    generator.generate("Second question", RETRIEVED, use_history=False)
    assert len(api_calls[-1]["json"]["messages"]) == 2


def test_clear_history(generator, api_calls):
    generator.generate("Best horror games?", RETRIEVED)
    generator.clear_history()
    assert generator.get_history() == []


def test_timeout_returns_message_and_keeps_history_clean(generator, monkeypatch):
    def timeout(*args, **kwargs):
        raise requests.exceptions.Timeout

    monkeypatch.setattr(generator_module.requests, "post", timeout)
    answer = generator.generate("Best horror games?", RETRIEVED)

    assert "timed out" in answer
    assert generator.get_history() == []


def test_http_error_reports_status_code(generator, monkeypatch):
    monkeypatch.setattr(
        generator_module.requests, "post", lambda *a, **k: FakeResponse(status_code=401)
    )
    answer = generator.generate("Best horror games?", RETRIEVED)

    assert "401" in answer
    assert generator.get_history() == []
