"""Deterministic scorers — the first layer of the eval stack.

Every scorer is a pure function of (output, config, expected) returning a
score in [0, 1] plus a human-readable detail string. No model calls, no
network, no randomness: the same output always scores the same way, which
is what makes these safe to gate a deploy on.
"""

from __future__ import annotations

import json
import re

from .judge import DEFAULT_RUBRIC, Judge, get_judge, judge_to_scorer_result
from .models import ScorerConfig, ScorerResult


def _norm(text: str, case_sensitive: bool) -> str:
    return text if case_sensitive else text.lower()


def score_exact_match(output: str, expected: str, case_sensitive: bool = False) -> tuple[float, str]:
    if expected is None:
        return 0.0, "exact_match needs an expected value"
    a, b = _norm(output.strip(), case_sensitive), _norm(expected.strip(), case_sensitive)
    return (1.0, "output matches expected exactly") if a == b else (0.0, f"expected {expected!r}, got {output.strip()!r}")


def score_contains(output: str, keywords: list[str], case_sensitive: bool = False) -> tuple[float, str]:
    if not keywords:
        return 0.0, "contains needs at least one keyword"
    hay = _norm(output, case_sensitive)
    found = [k for k in keywords if _norm(k, case_sensitive) in hay]
    missing = [k for k in keywords if k not in found]
    score = len(found) / len(keywords)
    detail = f"found {len(found)}/{len(keywords)} keywords" + (f"; missing: {missing}" if missing else "")
    return score, detail


def score_contains_any(output: str, keywords: list[str], case_sensitive: bool = False) -> tuple[float, str]:
    if not keywords:
        return 0.0, "contains_any needs at least one keyword"
    hay = _norm(output, case_sensitive)
    found = [k for k in keywords if _norm(k, case_sensitive) in hay]
    return (1.0, f"found keyword {found[0]!r}") if found else (0.0, f"none of {keywords} present")


def score_regex(output: str, pattern: str) -> tuple[float, str]:
    if not pattern:
        return 0.0, "regex needs a pattern"
    try:
        match = re.search(pattern, output)
    except re.error as exc:
        return 0.0, f"invalid regex {pattern!r}: {exc}"
    return (1.0, f"pattern {pattern!r} matched") if match else (0.0, f"pattern {pattern!r} did not match")


def score_json_valid(output: str) -> tuple[float, str]:
    try:
        parsed = json.loads(output.strip())
    except json.JSONDecodeError as exc:
        return 0.0, f"not valid JSON: {exc}"
    kind = type(parsed).__name__
    return 1.0, f"valid JSON ({kind})"


def score_length_between(output: str, min_length: int | None, max_length: int | None) -> tuple[float, str]:
    n = len(output)
    if min_length is not None and n < min_length:
        return 0.0, f"output is {n} chars, below minimum {min_length}"
    if max_length is not None and n > max_length:
        return 0.0, f"output is {n} chars, above maximum {max_length}"
    return 1.0, f"output length {n} chars within bounds"


def score_starts_with(output: str, prefix: str, case_sensitive: bool = False) -> tuple[float, str]:
    if prefix is None:
        return 0.0, "starts_with needs an expected prefix"
    out = _norm(output.strip(), case_sensitive)
    pre = _norm(prefix, case_sensitive)
    return (1.0, f"output starts with {prefix!r}") if out.startswith(pre) else (0.0, f"output does not start with {prefix!r}")


def score_groundedness(
    output: str,
    keywords: list[str],
    forbidden: list[str],
    case_sensitive: bool = False,
) -> tuple[float, str]:
    """Groundedness: the output must cover the source facts and must not
    state any of the forbidden (hallucinated) claims.

    Score is the fraction of required facts present; any forbidden term
    present caps the score at 0, because one invented fact poisons the
    answer even when the rest is grounded.
    """
    if not keywords and not forbidden:
        return 0.0, "groundedness needs keywords (source facts) or forbidden terms"
    hay = _norm(output, case_sensitive)
    found_forbidden = [f for f in forbidden if _norm(f, case_sensitive) in hay]
    if found_forbidden:
        return 0.0, f"hallucinated claim present: {found_forbidden}"
    if not keywords:
        return 1.0, "no forbidden terms present (no source facts required)"
    found = [k for k in keywords if _norm(k, case_sensitive) in hay]
    missing = [k for k in keywords if k not in found]
    score = len(found) / len(keywords)
    detail = f"grounded {len(found)}/{len(keywords)} source facts" + (
        f"; missing: {missing}" if missing else ""
    )
    return score, detail


def run_scorer(
    config: ScorerConfig,
    output: str,
    expected: str | None = None,
    prompt: str = "",
    judge: Judge | None = None,
) -> ScorerResult:
    """Run one configured scorer and return its score, pass/fail and detail.

    ``prompt`` and ``judge`` are only used by the ``llm_judge`` scorer;
    deterministic scorers ignore them. When no judge is given, the
    scorer's configured judge name is resolved (``fake`` by default).
    """
    if config.type == "llm_judge":
        active_judge = judge if judge is not None else get_judge(config.judge)
        verdict = active_judge.judge(
            prompt, output, config.rubric or DEFAULT_RUBRIC, expected or config.expected
        )
        return judge_to_scorer_result(verdict, config.threshold)

    if config.type == "groundedness":
        score, detail = score_groundedness(
            output, config.keywords, config.forbidden, config.case_sensitive
        )
    elif config.type == "exact_match":
        score, detail = score_exact_match(output, expected or config.expected or "", config.case_sensitive)
    elif config.type == "contains":
        score, detail = score_contains(output, config.keywords, config.case_sensitive)
    elif config.type == "contains_any":
        score, detail = score_contains_any(output, config.keywords, config.case_sensitive)
    elif config.type == "regex":
        score, detail = score_regex(output, config.pattern or "")
    elif config.type == "json_valid":
        score, detail = score_json_valid(output)
    elif config.type == "length_between":
        score, detail = score_length_between(output, config.min_length, config.max_length)
    elif config.type == "starts_with":
        score, detail = score_starts_with(output, expected or config.expected or "", config.case_sensitive)
    else:  # pragma: no cover — Pydantic constrains the type, this is defensive
        score, detail = 0.0, f"unknown scorer type {config.type!r}"

    return ScorerResult(
        scorer_type=config.type,
        score=score,
        passed=score >= config.threshold,
        detail=detail,
    )
