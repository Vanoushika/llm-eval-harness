"""Unit tests for the deterministic scorers."""

import pytest

from evalharness.models import ScorerConfig
from evalharness.scorers import run_scorer


def check(scorer_type, output, **kwargs):
    expected = kwargs.pop("expected", None)
    cfg = ScorerConfig(type=scorer_type, **kwargs)
    return run_scorer(cfg, output, expected)


def test_exact_match_passes_on_identical_text():
    result = check("exact_match", "  Paris ", expected="Paris")
    assert result.score == 1.0 and result.passed


def test_exact_match_fails_on_different_text():
    result = check("exact_match", "London", expected="Paris")
    assert result.score == 0.0 and not result.passed


def test_exact_match_is_case_insensitive_by_default():
    result = check("exact_match", "paris", expected="Paris")
    assert result.score == 1.0


def test_contains_scores_partial_credit():
    result = check("contains", "The capital is Paris.", keywords=["Paris", "Berlin"])
    assert result.score == 0.5 and not result.passed
    assert "1/2" in result.detail


def test_contains_passes_when_all_keywords_present():
    result = check("contains", "Paris and Berlin", keywords=["Paris", "Berlin"])
    assert result.score == 1.0 and result.passed


def test_contains_any_passes_on_one_keyword():
    result = check("contains_any", "only Berlin here", keywords=["Paris", "Berlin"])
    assert result.score == 1.0 and result.passed


def test_regex_matches_and_misses():
    assert check("regex", "order 12345 shipped", pattern=r"\d{5}").score == 1.0
    assert check("regex", "no digits here", pattern=r"\d{5}").score == 0.0


def test_json_valid_accepts_object_and_rejects_prose():
    assert check("json_valid", '{"name": "Ada"}').score == 1.0
    assert check("json_valid", "just a sentence").score == 0.0


def test_length_between_bounds():
    assert check("length_between", "hello", min_length=3, max_length=10).score == 1.0
    assert check("length_between", "hi", min_length=3).score == 0.0
    assert check("length_between", "a very long answer indeed", max_length=5).score == 0.0


def test_starts_with():
    assert check("starts_with", "Hello world", expected="Hello").score == 1.0
    assert check("starts_with", "Goodbye world", expected="Hello").score == 0.0


def test_threshold_below_one_allows_partial_credit():
    cfg = ScorerConfig(type="contains", keywords=["Paris", "Berlin"], threshold=0.5)
    result = run_scorer(cfg, "Paris only", None)
    assert result.score == 0.5 and result.passed


def test_groundedness_passes_when_all_source_facts_covered():
    result = check(
        "groundedness",
        "The red Acme Widget costs $19 and ships in 2 days.",
        keywords=["red", "$19", "2 days"],
        forbidden=["blue", "$29"],
    )
    assert result.score == 1.0 and result.passed


def test_groundedness_scores_partial_credit_for_missing_facts():
    result = check(
        "groundedness",
        "The red Widget costs $19.",
        keywords=["red", "$19", "2 days"],
    )
    assert result.score == pytest.approx(2 / 3)
    assert not result.passed


def test_groundedness_scores_zero_when_a_forbidden_claim_appears():
    # One invented fact poisons the answer even when the rest is grounded.
    result = check(
        "groundedness",
        "The red Acme Widget costs $19 with free shipping.",
        keywords=["red", "$19"],
        forbidden=["free shipping"],
    )
    assert result.score == 0.0 and not result.passed
    assert "hallucinated" in result.detail


def test_groundedness_needs_facts_or_forbidden_terms():
    result = check("groundedness", "anything at all")
    assert result.score == 0.0


def test_llm_judge_uses_the_fake_judge_offline():
    cfg = ScorerConfig(type="llm_judge", threshold=0.5)
    result = run_scorer(
        cfg,
        "The capital of France is Paris.",
        "Paris",
        prompt="What is the capital of France?",
    )
    assert result.scorer_type == "llm_judge"
    assert result.score == 1.0 and result.passed
    assert "Fake judge" in result.detail


def test_llm_judge_fails_a_wrong_answer():
    cfg = ScorerConfig(type="llm_judge", threshold=0.5)
    result = run_scorer(
        cfg,
        "The answer is 5.",
        "4",
        prompt="What is 2 + 2? Answer with just the number.",
    )
    assert result.score == 0.2 and not result.passed
