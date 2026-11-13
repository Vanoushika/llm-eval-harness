"""LLM-as-judge scoring — the second layer of the eval stack.

A judge scores an output against a rubric in natural language. The
interface is one method, ``judge(prompt, output, rubric, expected)``,
returning a score in [0, 1] plus a short rationale.

Two judges ship on day 2:

* ``FakeJudge`` — deterministic and offline. It scores from explainable
  components (expected-answer containment, topical overlap with the
  prompt, length sanity) so the judge path is testable with no API key
  and no network. It is a stand-in, and its rationale says so.
* ``OpenAIJudge`` — calls any OpenAI-compatible chat-completions
  endpoint (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``,
  ``OPENAI_JUDGE_MODEL``). It asks the model for a JSON verdict and
  parses it defensively; a verdict that cannot be parsed scores 0 and
  says so in the rationale rather than inventing a number.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import httpx

from .models import ScorerResult

DEFAULT_RUBRIC = (
    "Score the output for correctness and relevance to the prompt: "
    "1.0 is fully correct and directly answers the prompt, 0.0 is "
    "wrong or unrelated."
)

_STOPWORDS = frozenset(
    "a an the is are was were be been being to of in on at for with and or "
    "what how why when where which who whom this that it its as by from do "
    "does did answer just only".split()
)


class JudgeVerdict:
    """What a judge returns: a score plus the reasoning behind it."""

    def __init__(self, score: float, rationale: str, raw: str | None = None):
        self.score = max(0.0, min(1.0, score))
        self.rationale = rationale
        self.raw = raw


class Judge(Protocol):
    name: str

    def judge(
        self,
        prompt: str,
        output: str,
        rubric: str,
        expected: str | None = None,
    ) -> JudgeVerdict: ...


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


class FakeJudge:
    """Deterministic offline judge for testing the judge path.

    Components (all explainable, all in the rationale):
    * 0.50 — the expected answer appears in the output (when one is given)
    * 0.30 — topical overlap: the output shares content words with the prompt
    * 0.20 — length sanity: non-empty and under 5,000 characters
    """

    name = "fake"

    def judge(
        self,
        prompt: str,
        output: str,
        rubric: str,
        expected: str | None = None,
    ) -> JudgeVerdict:
        parts: list[str] = []
        score = 0.0

        if expected:
            hit = expected.strip().lower() in output.lower()
            score += 0.50 if hit else 0.0
            parts.append(f"expected answer {'present' if hit else 'absent'} (+0.50)" if hit else "expected answer absent (+0.00)")
        else:
            parts.append("no expected answer given (expected component skipped)")

        prompt_words = _content_words(prompt)
        output_words = _content_words(output)
        overlap = prompt_words & output_words
        overlap_score = 0.30 * (len(overlap) / len(prompt_words)) if prompt_words else 0.0
        score += overlap_score
        parts.append(f"topical overlap {len(overlap)}/{len(prompt_words)} words (+{overlap_score:.2f})")

        sane = bool(output.strip()) and len(output) <= 5000
        score += 0.20 if sane else 0.0
        parts.append(f"length {'sane' if sane else 'empty or over 5,000 chars'} (+{'0.20' if sane else '0.00'})")

        rationale = "Fake judge (offline stand-in): " + "; ".join(parts) + "."
        return JudgeVerdict(score=round(score, 4), rationale=rationale)


class OpenAIJudge:
    """Judge backed by an OpenAI-compatible chat-completions endpoint."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI judge needs OPENAI_API_KEY set (or pass api_key=); "
                "use judge 'fake' for offline runs"
            )
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        self._client = client
        self._timeout = timeout

    def _client_or_default(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self._timeout,
        )

    def judge(
        self,
        prompt: str,
        output: str,
        rubric: str,
        expected: str | None = None,
    ) -> JudgeVerdict:
        expected_line = f"\nExpected answer: {expected}" if expected else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict evaluator. Score the model's output "
                    "against the rubric below. Reply with JSON only, exactly "
                    'in this shape: {"score": <number 0-1>, "rationale": "<one or two sentences>"}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Rubric: {rubric}\n\nPrompt: {prompt}\n\n"
                    f"Model output: {output}{expected_line}"
                ),
            },
        ]
        client = self._client_or_default()
        response = client.post(
            "/chat/completions",
            json={"model": self.model, "messages": messages, "temperature": 0},
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return parse_judge_response(content)


def parse_judge_response(content: str) -> JudgeVerdict:
    """Parse a judge's JSON verdict; fall back to a bare number, else 0."""
    text = content.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("score"), (int, float)):
            return JudgeVerdict(
                score=float(data["score"]),
                rationale=str(data.get("rationale", "judge gave no rationale")),
                raw=text,
            )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
    if match:
        return JudgeVerdict(
            score=float(match.group(1)),
            rationale=f"judge reply was not JSON; score extracted from: {text[:200]!r}",
            raw=text,
        )
    return JudgeVerdict(
        score=0.0,
        rationale=f"judge reply could not be parsed as a score: {text[:200]!r}",
        raw=text,
    )


def get_judge(name: str, client: httpx.Client | None = None) -> Judge:
    """Resolve a judge by name: 'fake' (offline) or 'openai' (API key)."""
    if name == "fake":
        return FakeJudge()
    if name == "openai":
        return OpenAIJudge(client=client)
    raise ValueError(f"unknown judge {name!r} — use 'fake' or 'openai'")


def judge_to_scorer_result(
    verdict: JudgeVerdict, threshold: float
) -> ScorerResult:
    return ScorerResult(
        scorer_type="llm_judge",
        score=verdict.score,
        passed=verdict.score >= threshold,
        detail=verdict.rationale,
    )
