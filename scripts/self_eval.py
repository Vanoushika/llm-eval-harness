"""Self-eval: the harness evaluates itself, end to end.

Dogfoods the public paths a user would actually rely on:

1. The pytest suite (the harness's own tests of its scorers, judge,
   providers, runner, compare and API).
2. The CLI on both example suites — the sample suite (which deliberately
   contains a failing case, so the CLI must exit 1) and the judge suite
   (same), proving failure detection and the CI gate behaviour.
3. The CI gate suite — every case must pass and the CLI must exit 0.
4. Run comparison — the same suite run twice must show no regressions,
   and a run against a worse stub must flag the regression.

Everything here runs fully offline; no API key is used. Run it with:
    python scripts/self_eval.py
It is also a step in .github/workflows/ci.yml.

Exit code 0 means every self-eval check held; 1 means something regressed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evalharness.cli import main as cli_main  # noqa: E402
from evalharness.compare import compare_runs  # noqa: E402
from evalharness.datasets import load_suite  # noqa: E402
from evalharness.providers import DeterministicOfflineProvider  # noqa: E402
from evalharness.runner import run_suite  # noqa: E402


def _run_pytest() -> tuple[int, int]:
    """Run pytest and return (passed, failed) counts from its last line."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    passed = failed = 0
    for line in proc.stdout.splitlines():
        if "passed" in line or "failed" in line:
            parts = line.split(",")
            for part in parts:
                part = part.strip()
                if part.endswith(" passed"):
                    passed = int(part.split()[0])
                elif part.endswith(" failed"):
                    failed = int(part.split()[0])
    return passed, failed


def _cli_run(suite_path: str) -> int:
    """Run the CLI on a suite file and return its exit code."""
    return cli_main(["run", str(ROOT / suite_path)])


def _comparison_checks() -> tuple[bool, bool]:
    """Same-suite runs must not regress; a worse stub must be flagged."""
    suite = load_suite(ROOT / "examples" / "sample_suite.yaml")
    provider = DeterministicOfflineProvider()
    baseline = run_suite(suite, provider)
    identical = run_suite(suite, provider)
    no_regression = not compare_runs(baseline, identical).has_regressions

    worse = load_suite(ROOT / "examples" / "sample_suite.yaml")
    for case in worse.cases:
        if case.name == "capital-of-france":
            case.stub_output = "The capital of France is London."
    degraded = run_suite(worse, provider)
    regression_flagged = compare_runs(baseline, degraded).has_regressions
    return no_regression, regression_flagged


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    passed, failed = _run_pytest()
    checks.append(("pytest suite", failed == 0 and passed > 0,
                   f"{passed} passed, {failed} failed"))

    sample_exit = _cli_run("examples/sample_suite.yaml")
    checks.append(("sample suite fails loudly (exit 1)",
                   sample_exit == 1, f"exit code {sample_exit}"))

    judge_exit = _cli_run("examples/judge_suite.yaml")
    checks.append(("judge suite fails loudly (exit 1)",
                   judge_exit == 1, f"exit code {judge_exit}"))

    gate_exit = _cli_run("examples/ci_gate_suite.yaml")
    checks.append(("CI gate suite passes (exit 0)",
                   gate_exit == 0, f"exit code {gate_exit}"))

    no_regression, regression_flagged = _comparison_checks()
    checks.append(("identical runs show no regression",
                   no_regression, "compare clean" if no_regression else "false regression"))
    checks.append(("worse run is flagged as a regression",
                   regression_flagged, "regression caught" if regression_flagged else "regression missed"))

    # Prove the CLI compare path too, via run JSON files.
    suite = load_suite(ROOT / "examples" / "sample_suite.yaml")
    provider = DeterministicOfflineProvider()
    run_a = run_suite(suite, provider)
    run_b = run_suite(suite, provider)
    with tempfile.TemporaryDirectory() as tmp:
        a_path = Path(tmp) / "a.json"
        b_path = Path(tmp) / "b.json"
        a_path.write_text(run_a.model_dump_json(), encoding="utf-8")
        b_path.write_text(run_b.model_dump_json(), encoding="utf-8")
        cmp_exit = cli_main(["compare", "--baseline", str(a_path),
                             "--candidate", str(b_path)])
    checks.append(("CLI compare exits 0 on identical runs",
                   cmp_exit == 0, f"exit code {cmp_exit}"))

    print("\nSelf-eval — llm-eval-harness evaluating itself")
    print("=" * 52)
    all_ok = True
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        all_ok = all_ok and ok
        print(f"  [{mark}] {name} — {detail}")
    print(f"\nResult: {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks held")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
