"""CLI runner — the CI-gating entry point.

Usage:
    python -m evalharness.cli run examples/sample_suite.yaml
    python -m evalharness.cli run my_suite.yaml --model offline
    python -m evalharness.cli compare --baseline <run-id> --candidate <run-id>
    python -m evalharness.cli report <run-id> --out report.html

Exit codes: ``run`` exits 0 when every case passes and 1 otherwise;
``compare`` exits 0 when there are no regressions and 1 when any case
regressed, so a CI step can gate a deploy on either result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import DEFAULT_REGRESSION_THRESHOLD, compare_runs, comparison_to_text
from .datasets import load_suite
from .models import EvalRun
from .providers import get_provider
from .report import render_run_html
from .runner import run_suite
from .store import Store


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite_file)
    if args.model:
        suite.model = args.model
    provider = get_provider(suite.model)
    run = run_suite(suite, provider)

    print(f"Suite: {suite.name}  (model: {run.model})")
    print(f"Result: {run.passed_count}/{run.total_count} cases passed, "
          f"mean score {run.mean_score:.2f}")
    for result in run.results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"  [{mark}] {result.case_name} — score {result.mean_score:.2f}")
        for scorer in result.scores:
            print(f"         {scorer.scorer_type}: {scorer.score:.2f} — {scorer.detail}")
    return 0 if run.passed_count == run.total_count else 1


def _load_run(ref: str) -> EvalRun:
    """Load a run by store ID, or from a JSON file if the ref is a path."""
    path = Path(ref)
    if path.exists():
        return EvalRun(**json.loads(path.read_text(encoding="utf-8")))
    run = Store().get_run(ref)
    if run is None:
        raise ValueError(f"run {ref!r} not found in the store and not a file path")
    return run


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = _load_run(args.baseline)
    candidate = _load_run(args.candidate)
    comparison = compare_runs(
        baseline, candidate, regression_threshold=args.threshold
    )
    print(comparison_to_text(comparison))
    return 0 if not comparison.has_regressions else 1


def cmd_report(args: argparse.Namespace) -> int:
    run = _load_run(args.run_id)
    html_text = render_run_html(run)
    out = Path(args.out)
    out.write_text(html_text, encoding="utf-8")
    print(f"Wrote HTML report for run {run.id} ({run.suite_name}) to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalharness", description="LLM eval harness CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="run a suite file and report the result")
    run_parser.add_argument("suite_file", help="path to a .yaml/.yml/.json suite file")
    run_parser.add_argument("--model", default=None, help="provider name (default: from the suite file)")

    compare_parser = sub.add_parser(
        "compare", help="compare two runs and report regressions"
    )
    compare_parser.add_argument("--baseline", required=True, help="baseline run id or run JSON file")
    compare_parser.add_argument("--candidate", required=True, help="candidate run id or run JSON file")
    compare_parser.add_argument(
        "--threshold", type=float, default=DEFAULT_REGRESSION_THRESHOLD,
        help="score drop that counts as a regression (default 0.10)",
    )

    report_parser = sub.add_parser("report", help="write a run as an HTML report")
    report_parser.add_argument("run_id", help="run id or run JSON file")
    report_parser.add_argument("--out", default="report.html", help="output file (default report.html)")

    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "report":
        return cmd_report(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
