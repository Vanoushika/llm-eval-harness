"""Tests for the runner and the offline provider."""

from evalharness.models import EvalCase, EvalSuite, ScorerConfig
from evalharness.providers import DeterministicOfflineProvider, get_provider
from evalharness.runner import render_prompt, run_suite


def make_suite(cases):
    return EvalSuite(name="test-suite", model="offline", cases=cases)


def test_offline_provider_returns_stub_output_verbatim():
    provider = DeterministicOfflineProvider()
    case = EvalCase(
        name="stubbed",
        prompt_template="anything",
        stub_output="pinned answer",
    )
    assert provider.generate("anything", case) == "pinned answer"


def test_offline_provider_answers_known_prompts_deterministically():
    provider = DeterministicOfflineProvider()
    first = provider.generate("What is the capital of France?")
    second = provider.generate("What is the capital of France?")
    assert first == second == "The capital of France is Paris."


def test_get_provider_rejects_unknown_names():
    try:
        get_provider("gpt-9")
    except ValueError as exc:
        assert "unknown provider" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown provider")


def test_render_prompt_substitutes_inputs():
    assert render_prompt("Hello {name}, you are {role}.", {"name": "Ada", "role": "engineer"}) == (
        "Hello Ada, you are engineer."
    )


def test_runner_marks_passing_and_failing_cases():
    suite = make_suite(
        [
            EvalCase(
                name="passes",
                prompt_template="What is the capital of France?",
                expected="Paris",
                scorers=[ScorerConfig(type="contains", keywords=["Paris"])],
            ),
            EvalCase(
                name="fails",
                prompt_template="What is the capital of France?",
                expected="Berlin",
                scorers=[ScorerConfig(type="exact_match")],
            ),
        ]
    )
    run = run_suite(suite, DeterministicOfflineProvider())
    assert run.total_count == 2
    assert run.passed_count == 1
    assert run.results[0].passed is True
    assert run.results[1].passed is False
    assert run.status == "completed"
    assert run.finished_at is not None


def test_runner_mean_score_is_mean_of_case_means():
    suite = make_suite(
        [
            EvalCase(
                name="perfect",
                prompt_template="x",
                stub_output="Paris",
                scorers=[ScorerConfig(type="exact_match")],
                expected="Paris",
            ),
            EvalCase(
                name="half",
                prompt_template="x",
                stub_output="Paris only",
                scorers=[ScorerConfig(type="contains", keywords=["Paris", "Berlin"])],
            ),
        ]
    )
    run = run_suite(suite, DeterministicOfflineProvider())
    assert run.results[0].mean_score == 1.0
    assert run.results[1].mean_score == 0.5
    assert run.mean_score == 0.75


def test_runner_runs_llm_judge_scorer_with_fake_judge():
    suite = make_suite(
        [
            EvalCase(
                name="judged",
                prompt_template="What is the capital of France?",
                expected="Paris",
                stub_output="The capital of France is Paris.",
                scorers=[ScorerConfig(type="llm_judge", threshold=0.5)],
            ),
            EvalCase(
                name="grounded",
                prompt_template="Summarise the note.",
                stub_output="The red Widget costs $19.",
                scorers=[
                    ScorerConfig(
                        type="groundedness",
                        keywords=["red", "$19"],
                        forbidden=["blue"],
                    )
                ],
            ),
        ]
    )
    run = run_suite(suite, DeterministicOfflineProvider())
    assert run.passed_count == 2
    assert run.results[0].scores[0].scorer_type == "llm_judge"
    assert run.results[0].scores[0].score == 1.0
    assert run.results[1].scores[0].scorer_type == "groundedness"
    assert run.results[1].scores[0].score == 1.0


def test_runner_raises_for_unknown_judge_name():
    suite = make_suite(
        [
            EvalCase(
                name="judged",
                prompt_template="x",
                stub_output="y",
                scorers=[ScorerConfig(type="llm_judge", judge="nope")],
            ),
        ]
    )
    try:
        run_suite(suite, DeterministicOfflineProvider())
    except ValueError as exc:
        assert "unknown judge" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown judge")
