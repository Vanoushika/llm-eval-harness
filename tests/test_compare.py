"""Tests for run comparison / regression reporting."""

from evalharness.compare import compare_runs, comparison_to_text
from evalharness.models import (
    CaseResult,
    EvalRun,
    ScorerResult,
)


def _run(run_id, cases):
    """cases: list of (name, score, passed)."""
    results = [
        CaseResult(
            case_id=name,
            case_name=name,
            prompt="p",
            output="o",
            scores=[ScorerResult(scorer_type="contains", score=score, passed=passed, detail="")],
            mean_score=score,
            passed=passed,
        )
        for name, score, passed in cases
    ]
    return EvalRun(
        id=run_id,
        suite_id="suite-1",
        suite_name="demo",
        model="offline",
        results=results,
        mean_score=sum(c[1] for c in cases) / len(cases) if cases else 0.0,
        passed_count=sum(1 for c in cases if c[2]),
        total_count=len(cases),
    )


def test_identical_runs_have_no_regressions():
    baseline = _run("a", [("one", 1.0, True), ("two", 0.5, False)])
    candidate = _run("b", [("one", 1.0, True), ("two", 0.5, False)])
    comparison = compare_runs(baseline, candidate)
    assert comparison.regression_count == 0
    assert comparison.mean_delta == 0.0
    assert len(comparison.unchanged) == 2


def test_score_drop_beyond_threshold_is_a_regression():
    baseline = _run("a", [("one", 1.0, True)])
    candidate = _run("b", [("one", 0.8, True)])
    comparison = compare_runs(baseline, candidate)
    assert comparison.regression_count == 1
    assert comparison.regressions[0].case_name == "one"
    assert comparison.regressions[0].delta == -0.2


def test_small_drop_within_threshold_is_unchanged():
    baseline = _run("a", [("one", 1.0, True)])
    candidate = _run("b", [("one", 0.95, True)])
    comparison = compare_runs(baseline, candidate)
    assert comparison.regression_count == 0
    assert len(comparison.unchanged) == 1


def test_pass_to_fail_flip_is_a_regression_even_with_small_delta():
    baseline = _run("a", [("one", 1.0, True)])
    candidate = _run("b", [("one", 0.95, False)])
    comparison = compare_runs(baseline, candidate)
    assert comparison.regression_count == 1


def test_improvement_is_reported():
    baseline = _run("a", [("one", 0.0, False)])
    candidate = _run("b", [("one", 1.0, True)])
    comparison = compare_runs(baseline, candidate)
    assert len(comparison.improvements) == 1
    assert comparison.mean_delta == 1.0


def test_added_and_removed_cases_are_listed():
    baseline = _run("a", [("one", 1.0, True), ("old", 1.0, True)])
    candidate = _run("b", [("one", 1.0, True), ("new", 0.5, False)])
    comparison = compare_runs(baseline, candidate)
    assert [c.case_name for c in comparison.new_cases] == ["new"]
    assert [c.case_name for c in comparison.removed_cases] == ["old"]


def test_compare_rejects_runs_from_different_suites():
    baseline = _run("a", [("one", 1.0, True)])
    candidate = _run("b", [("one", 1.0, True)])
    candidate.suite_id = "suite-2"
    try:
        compare_runs(baseline, candidate)
    except ValueError as exc:
        assert "different suites" in str(exc)
    else:
        raise AssertionError("expected ValueError for different suites")


def test_comparison_text_names_regressions():
    baseline = _run("a", [("one", 1.0, True)])
    candidate = _run("b", [("one", 0.5, False)])
    text = comparison_to_text(compare_runs(baseline, candidate))
    assert "[REGRESSED] one" in text
    assert "Mean score delta: -0.50" in text
