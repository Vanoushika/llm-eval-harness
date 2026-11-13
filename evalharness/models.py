"""Core data models for the LLM evaluation harness.

Suites, cases, runs and results are plain Pydantic models so they can move
unchanged between the API, the CLI runner and storage. The store persists
them as JSON; nothing here depends on a particular database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


ScorerType = Literal[
    "exact_match",
    "contains",
    "contains_any",
    "regex",
    "json_valid",
    "length_between",
    "starts_with",
    "groundedness",
    "llm_judge",
]


class ScorerConfig(BaseModel):
    """One deterministic check applied to a model's output."""

    type: ScorerType
    # expected text (exact_match), prefix (starts_with)
    expected: str | None = None
    # keywords that must appear (contains / contains_any)
    keywords: list[str] = Field(default_factory=list)
    # regex pattern (regex)
    pattern: str | None = None
    # length bounds (length_between)
    min_length: int | None = None
    max_length: int | None = None
    case_sensitive: bool = False
    # source facts that must be covered (groundedness); terms that must
    # NOT appear — hallucinated claims (groundedness)
    forbidden: list[str] = Field(default_factory=list)
    # rubric in natural language (llm_judge) and which judge to use:
    # "fake" (offline, deterministic) or "openai" (API key)
    rubric: str | None = None
    judge: str = "fake"
    # a case passes this scorer when score >= threshold
    threshold: float = 1.0


class EvalCase(BaseModel):
    """One test case: an input, a prompt template, and its scorers."""

    id: str = Field(default_factory=new_id)
    name: str
    # input values substituted into prompt_template with {placeholders}
    input: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str
    # optional expected output, used by exact_match and as documentation
    expected: str | None = None
    scorers: list[ScorerConfig] = Field(default_factory=list)
    # when set, the offline provider returns this verbatim instead of
    # generating — the deterministic stand-in for "the model under test"
    stub_output: str | None = None


class EvalSuite(BaseModel):
    """A named collection of cases, run together against one provider."""

    id: str = Field(default_factory=new_id)
    name: str
    description: str = ""
    # provider name, e.g. "offline" (day 1); hosted providers land day 2
    model: str = "offline"
    cases: list[EvalCase] = Field(default_factory=list)


class ScorerResult(BaseModel):
    scorer_type: str
    score: float  # 0.0 – 1.0
    passed: bool
    detail: str


class CaseResult(BaseModel):
    case_id: str
    case_name: str
    prompt: str
    output: str
    scores: list[ScorerResult] = Field(default_factory=list)
    mean_score: float = 0.0
    passed: bool = False
    latency_ms: float = 0.0


class EvalRun(BaseModel):
    """The scored result of running a whole suite once."""

    id: str = Field(default_factory=new_id)
    suite_id: str
    suite_name: str
    model: str
    status: Literal["completed", "failed"] = "completed"
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    results: list[CaseResult] = Field(default_factory=list)
    mean_score: float = 0.0
    passed_count: int = 0
    total_count: int = 0
