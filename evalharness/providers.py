"""Model providers — the thing being evaluated.

A provider is anything with a ``generate(prompt) -> str`` method. The
harness only depends on that small interface, so swapping the model under
test is one line. Day 1 shipped the deterministic offline provider; day 2
adds a hosted provider behind an API key (``OPENAI_API_KEY``) for real
model runs.
"""

from __future__ import annotations

import os
from typing import Protocol

import httpx

from .models import EvalCase


class ModelProvider(Protocol):
    name: str

    def generate(self, prompt: str, case: EvalCase | None = None) -> str: ...


class DeterministicOfflineProvider:
    """Fully offline, deterministic stand-in for a real model.

    Behaviour, in order:
    1. If the case carries a ``stub_output``, return it verbatim. This is
       how the harness itself is tested and how a suite author pins the
       exact output a case should be scored against.
    2. A small set of keyword rules for the sample suite, so the demo runs
       without stubs.
    3. A generic deterministic reply that echoes the prompt's key words.
    """

    name = "offline"

    _RULES: tuple[tuple[str, str], ...] = (
        ("capital of france", "The capital of France is Paris."),
        ("2 + 2", "4"),
        ("2+2", "4"),
    )

    def generate(self, prompt: str, case: EvalCase | None = None) -> str:
        if case is not None and case.stub_output is not None:
            return case.stub_output
        lowered = prompt.lower()
        for keyword, answer in self._RULES:
            if keyword in lowered:
                return answer
        if "json" in lowered and "name" in lowered:
            return '{"name": "Ada", "role": "engineer"}'
        return f"Offline reply to: {prompt.strip()[:120]}"


def get_provider(name: str, client: httpx.Client | None = None) -> ModelProvider:
    """Resolve a provider by name.

    * ``offline`` — deterministic, no key, no network.
    * ``openai`` — hosted model via ``OPENAI_API_KEY`` (model from
      ``OPENAI_MODEL``, default ``gpt-4o-mini``); ``openai:<model>``
      overrides the model for this run.
    """
    if name == "offline":
        return DeterministicOfflineProvider()
    if name == "openai" or name.startswith("openai:"):
        model = name.split(":", 1)[1] if ":" in name else None
        return OpenAIProvider(model=model, client=client)
    raise ValueError(f"unknown provider {name!r} — use 'offline' or 'openai'")


class OpenAIProvider:
    """Hosted model via an OpenAI-compatible chat-completions endpoint.

    Reads ``OPENAI_API_KEY`` (required), ``OPENAI_BASE_URL`` (default
    ``https://api.openai.com/v1``) and ``OPENAI_MODEL`` (default
    ``gpt-4o-mini``) from the environment. Temperature is 0 so runs are
    as reproducible as a hosted model gets.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OpenAI provider needs OPENAI_API_KEY set; "
                "use provider 'offline' for keyless runs"
            )
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = client
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def _client_or_default(self) -> httpx.Client:
        if self._client is not None:
            return self._client
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self._timeout,
        )

    def generate(self, prompt: str, case: EvalCase | None = None) -> str:
        client = self._client_or_default()
        response = client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
