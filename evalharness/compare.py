"""Run comparison — the regression report.

Compare two runs of the same suite (a baseline and a candidate, usually
before and after a prompt or model change) and flag what moved. A case is
a *regression* when its score drops by at least ``regression_threshold``
(default 0.10) or it flips from pass to fail; an *improvement* is the
mirror image. Cases added or removed between runs are listed separately,
never silently dropped.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from .models import EvalRun

DEFAULT_REGRESSION_THRESHOLD = 0.10


class CaseComparison(BaseModel):
    case_name: str
    baseline_score: float | None = None
    candidate_score: float | None = None
    delta: float | None = None
    baseline_passed: bool | None = None
    candidate_passed: bool | None = None
    status: str  # regressed | improved | unchanged | new | removed


class RunComparison(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    suite_name: str
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD
    mean_delta: float = 0.0
    regressions: list[CaseComparison] = Field(default_factory=list)
    improvements: list[CaseComparison] = Field(default_factory=list)
    unchanged: list[CaseComparison] = Field(default_factory=list)
    new_cases: list[CaseComparison] = Field(default_factory=list)
    removed_cases: list[CaseComparison] = Field(default_factory=list)

    @computed_field
    @property
    def regression_count(self) -> int:
        return len(self.regressions)

    @computed_field
    @property
    def has_regressions(self) -> bool:
        return bool(self.regressions)


def compare_runs(
    baseline: EvalRun,
    candidate: EvalRun,
    regression_threshold: float = DEFAULT_REGRESSION_THRESHOLD,
) -> RunComparison:
    if baseline.suite_id != candidate.suite_id:
        raise ValueError(
            "cannot compare runs from different suites: "
            f"{baseline.suite_id!r} vs {candidate.suite_id!r}"
        )
    comparison = RunComparison(
        baseline_run_id=baseline.id,
        candidate_run_id=candidate.id,
        suite_name=baseline.suite_name,
        regression_threshold=regression_threshold,
        mean_delta=round(candidate.mean_score - baseline.mean_score, 4),
    )
    baseline_by_name = {r.case_name: r for r in baseline.results}
    candidate_by_name = {r.case_name: r for r in candidate.results}

    for name, base in baseline_by_name.items():
        cand = candidate_by_name.get(name)
        if cand is None:
            comparison.removed_cases.append(
                CaseComparison(
                    case_name=name,
                    baseline_score=base.mean_score,
                    baseline_passed=base.passed,
                    status="removed",
                )
            )
            continue
        delta = round(cand.mean_score - base.mean_score, 4)
        flipped_to_fail = base.passed and not cand.passed
        flipped_to_pass = not base.passed and cand.passed
        if delta <= -regression_threshold or flipped_to_fail:
            status = "regressed"
        elif delta >= regression_threshold or flipped_to_pass:
            status = "improved"
        else:
            status = "unchanged"
        entry = CaseComparison(
            case_name=name,
            baseline_score=base.mean_score,
            candidate_score=cand.mean_score,
            delta=delta,
            baseline_passed=base.passed,
            candidate_passed=cand.passed,
            status=status,
        )
        if status == "regressed":
            comparison.regressions.append(entry)
        elif status == "improved":
            comparison.improvements.append(entry)
        else:
            comparison.unchanged.append(entry)

    for name, cand in candidate_by_name.items():
        if name not in baseline_by_name:
            comparison.new_cases.append(
                CaseComparison(
                    case_name=name,
                    candidate_score=cand.mean_score,
                    candidate_passed=cand.passed,
                    status="new",
                )
            )

    return comparison


def comparison_to_text(comparison: RunComparison) -> str:
    """One-line-per-finding plain-text report, for the CLI and CI logs."""
    lines = [
        f"Comparison: {comparison.suite_name} "
        f"(baseline {comparison.baseline_run_id} -> candidate {comparison.candidate_run_id})",
        f"Mean score delta: {comparison.mean_delta:+.2f} "
        f"(threshold for a regression: {comparison.regression_threshold})",
    ]
    if comparison.regressions:
        lines.append(f"Regressions: {len(comparison.regressions)}")
        for entry in comparison.regressions:
            lines.append(
                f"  [REGRESSED] {entry.case_name}: "
                f"{entry.baseline_score:.2f} -> {entry.candidate_score:.2f} "
                f"({entry.delta:+.2f})"
            )
    else:
        lines.append("Regressions: none")
    if comparison.improvements:
        lines.append(f"Improvements: {len(comparison.improvements)}")
        for entry in comparison.improvements:
            lines.append(
                f"  [IMPROVED] {entry.case_name}: "
                f"{entry.baseline_score:.2f} -> {entry.candidate_score:.2f} "
                f"({entry.delta:+.2f})"
            )
    for entry in comparison.new_cases:
        lines.append(f"  [NEW] {entry.case_name}: score {entry.candidate_score:.2f}")
    for entry in comparison.removed_cases:
        lines.append(f"  [REMOVED] {entry.case_name}: was {entry.baseline_score:.2f}")
    return "\n".join(lines)
