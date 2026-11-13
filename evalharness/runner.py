"""The runner — executes a suite against a provider and scores every case.

Prompt rendering is deliberately simple: ``prompt_template.format(**input)``
with the case's input dict. A case passes when every one of its scorers
passes; the run's mean score is the mean of the cases' mean scores.
``llm_judge`` scorers resolve their judge once per run and are cached by
judge name, so a suite with ten judged cases makes ten judge calls, not
ten judge constructions.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .judge import Judge, get_judge
from .models import CaseResult, EvalRun, EvalSuite, utcnow
from .providers import ModelProvider
from .scorers import run_scorer


def render_prompt(template: str, inputs: dict) -> str:
    try:
        return template.format(**inputs)
    except KeyError as exc:
        raise ValueError(f"prompt template needs input key {exc}") from exc


def run_suite(suite: EvalSuite, provider: ModelProvider) -> EvalRun:
    run = EvalRun(
        suite_id=suite.id,
        suite_name=suite.name,
        model=provider.name,
        started_at=utcnow(),
        total_count=len(suite.cases),
    )
    case_results: list[CaseResult] = []
    judges: dict[str, Judge] = {}

    for case in suite.cases:
        started = time.perf_counter()
        prompt = render_prompt(case.prompt_template, case.input)
        output = provider.generate(prompt, case)
        latency_ms = (time.perf_counter() - started) * 1000

        scores = []
        for cfg in case.scorers:
            judge = None
            if cfg.type == "llm_judge":
                if cfg.judge not in judges:
                    judges[cfg.judge] = get_judge(cfg.judge)
                judge = judges[cfg.judge]
            scores.append(run_scorer(cfg, output, case.expected, prompt=prompt, judge=judge))
        mean = sum(s.score for s in scores) / len(scores) if scores else 0.0
        passed = bool(scores) and all(s.passed for s in scores)

        case_results.append(
            CaseResult(
                case_id=case.id,
                case_name=case.name,
                prompt=prompt,
                output=output,
                scores=scores,
                mean_score=round(mean, 4),
                passed=passed,
                latency_ms=round(latency_ms, 2),
            )
        )

    run.results = case_results
    run.passed_count = sum(1 for r in case_results if r.passed)
    run.mean_score = round(
        sum(r.mean_score for r in case_results) / len(case_results), 4
    ) if case_results else 0.0
    run.finished_at = datetime.now(timezone.utc)
    return run
