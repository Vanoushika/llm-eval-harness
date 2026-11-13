"""CLI tests — run, compare and report subcommands."""

import json

from evalharness.cli import main
from evalharness.datasets import load_suite
from evalharness.models import EvalRun
from evalharness.providers import DeterministicOfflineProvider
from evalharness.runner import run_suite


def _two_runs(tmp_path, baseline_stub, candidate_stub):
    """Run the same loaded suite twice with different stubs, saving both
    runs as JSON files. (One load, so both runs share the suite id.)"""
    suite = load_suite("examples/sample_suite.yaml")
    paths = []
    for stub in (baseline_stub, candidate_stub):
        for case in suite.cases:
            if case.name == "capital-of-france":
                case.stub_output = stub
        run = run_suite(suite, DeterministicOfflineProvider())
        path = tmp_path / f"run-{stub}.json"
        path.write_text(run.model_dump_json(), encoding="utf-8")
        paths.append(path)
    return paths


def _one_run(tmp_path, stub):
    return _two_runs(tmp_path, stub, stub)[0]


def test_cli_run_judge_suite_reports_two_passes_two_failures(capsys):
    exit_code = main(["run", "examples/judge_suite.yaml"])
    assert exit_code == 1  # two cases are deliberately wrong
    out = capsys.readouterr().out
    assert "2/4 cases passed" in out
    assert "llm_judge" in out
    assert "groundedness" in out


def test_cli_compare_exits_one_on_regression(tmp_path, capsys):
    baseline_path, candidate_path = _two_runs(tmp_path, "Paris", "Berlin")
    exit_code = main(
        ["compare", "--baseline", str(baseline_path), "--candidate", str(candidate_path)]
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "[REGRESSED] capital-of-france" in out


def test_cli_compare_exits_zero_for_identical_runs(tmp_path):
    baseline_path, candidate_path = _two_runs(tmp_path, "Paris", "Paris")
    exit_code = main(
        ["compare", "--baseline", str(baseline_path), "--candidate", str(candidate_path)]
    )
    assert exit_code == 0


def test_cli_report_writes_html_file(tmp_path):
    run_path = _one_run(tmp_path, "Paris")
    run = EvalRun(**json.loads(run_path.read_text(encoding="utf-8")))
    out_path = tmp_path / "report.html"
    exit_code = main(["report", str(run_path), "--out", str(out_path)])
    assert exit_code == 0
    html_text = out_path.read_text(encoding="utf-8")
    assert "sample-factual-qa" in html_text
    assert "capital-of-france" in html_text
    assert run.id in html_text


def test_cli_run_sample_suite_exit_code_matches_day1(capsys):
    exit_code = main(["run", "examples/sample_suite.yaml"])
    assert exit_code == 1  # the deliberately-wrong case still fails
    out = capsys.readouterr().out
    assert "3/4 cases passed, mean score 0.75" in out


def test_cli_compare_rejects_missing_run(tmp_path):
    try:
        main(["compare", "--baseline", "nope.json", "--candidate", "also-nope.json"])
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing run")
