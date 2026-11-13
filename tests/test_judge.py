"""Tests for the LLM-as-judge scorers."""

import httpx
import pytest

from evalharness.judge import (
    FakeJudge,
    OpenAIJudge,
    get_judge,
    parse_judge_response,
)


def _mock_client(payload, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(status, json=payload)

    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1",
    )


def test_fake_judge_scores_full_marks_for_correct_answer():
    verdict = FakeJudge().judge(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Score 1.0 for the correct capital.",
        expected="Paris",
    )
    assert verdict.score == 1.0
    assert "Fake judge" in verdict.rationale


def test_fake_judge_scores_zero_for_empty_output():
    verdict = FakeJudge().judge(
        "What is the capital of France?", "", "any rubric", expected="Paris"
    )
    assert verdict.score == 0.0


def test_fake_judge_partial_credit_without_expected_answer():
    # No expected answer: only topical overlap (0.30) + length (0.20) apply.
    verdict = FakeJudge().judge(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "any rubric",
    )
    assert verdict.score == 0.5


def test_parse_judge_response_reads_json_verdict():
    verdict = parse_judge_response('{"score": 0.8, "rationale": "mostly right"}')
    assert verdict.score == 0.8
    assert verdict.rationale == "mostly right"


def test_parse_judge_response_falls_back_to_bare_number():
    verdict = parse_judge_response("I would give this a 0.6 overall.")
    assert verdict.score == 0.6


def test_parse_judge_response_scores_zero_when_unparseable():
    verdict = parse_judge_response("no numbers here at all")
    assert verdict.score == 0.0
    assert "could not be parsed" in verdict.rationale


def test_openai_judge_parses_json_verdict_from_api():
    client = _mock_client(
        {"choices": [{"message": {"content": '{"score": 0.9, "rationale": "correct and clear"}'}}]}
    )
    judge = OpenAIJudge(api_key="test-key", client=client)
    verdict = judge.judge("prompt", "output", "rubric")
    assert verdict.score == 0.9
    assert verdict.rationale == "correct and clear"


def test_openai_judge_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIJudge()


def test_openai_judge_http_error_raises():
    client = _mock_client({"error": {"message": "bad key"}}, status=401)
    judge = OpenAIJudge(api_key="bad-key", client=client)
    with pytest.raises(httpx.HTTPStatusError):
        judge.judge("prompt", "output", "rubric")


def test_get_judge_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown judge"):
        get_judge("claude")
