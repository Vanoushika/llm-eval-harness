"""Tests for suite loading — including the Day 3 fix that two loads of the
same suite file must share one ID, so runs from separate CLI invocations
can be compared."""

from pathlib import Path

import pytest

from evalharness.compare import compare_runs
from evalharness.datasets import load_suite, stable_suite_id
from evalharness.providers import DeterministicOfflineProvider
from evalharness.runner import run_suite

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_same_file_loads_with_the_same_suite_id():
    first = load_suite(EXAMPLES / "sample_suite.yaml")
    second = load_suite(EXAMPLES / "sample_suite.yaml")
    assert first.id == second.id
    assert first.id == stable_suite_id("sample-factual-qa")


def test_different_suite_names_get_different_ids():
    sample = load_suite(EXAMPLES / "sample_suite.yaml")
    gate = load_suite(EXAMPLES / "ci_gate_suite.yaml")
    assert sample.id != gate.id


def test_explicit_id_in_file_is_respected(tmp_path):
    suite_file = tmp_path / "custom.yaml"
    suite_file.write_text(
        "id: my-pinned-id\nname: custom\nmodel: offline\ncases: []\n",
        encoding="utf-8",
    )
    suite = load_suite(suite_file)
    assert suite.id == "my-pinned-id"


def test_runs_from_separate_loads_of_the_same_file_compare():
    provider = DeterministicOfflineProvider()
    baseline = run_suite(load_suite(EXAMPLES / "sample_suite.yaml"), provider)
    candidate = run_suite(load_suite(EXAMPLES / "sample_suite.yaml"), provider)
    comparison = compare_runs(baseline, candidate)
    assert comparison.regression_count == 0
    assert comparison.mean_delta == 0.0


def test_load_suite_rejects_missing_and_bad_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_suite(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.txt"
    bad.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported suite file type"):
        load_suite(bad)
