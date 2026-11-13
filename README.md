# 🎯 LLM Eval Harness

> A self-hosted evaluation harness for LLM-powered features: define a suite
> of test cases once, score every model output against deterministic checks,
> and gate deploys on the result — no hosted eval platform required.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)](https://fastapi.tiangolo.com)
[![pytest](https://img.shields.io/badge/pytest-tested-0A7EA4)](https://pytest.org)

---

## Why this exists

Evaluation tooling is the breakout category of late 2026: the November
comparison guides list Braintrust, Langfuse, Arize Phoenix, LangSmith,
DeepEval, Ragas and Promptfoo as the tools teams are standardising on — and
OpenAI is shutting down its hosted Evals platform (read-only from
31 October 2026, fully shut down 30 November 2026), pushing teams toward
self-hosted harnesses. The consistent advice across those guides is the
same: deterministic checks first, LLM-as-judge layered on top, and a fixed
regression suite run on every prompt or model change, gated in CI.

This project is a small, honest version of that loop: suites as YAML files,
seven deterministic scorers plus LLM-as-judge and groundedness checks, run
history in SQLite (PostgreSQL-ready), run comparison for regression
detection, and a CLI whose exit code gates a deploy.

---

## ✨ What it does (Day 3 — 100%)

- 📋 **Suites as code** — a suite is a YAML/JSON file: cases with a prompt
  template, inputs, expected outputs and scorers
- 🧪 **Deterministic scorers** — `exact_match`, `contains`, `contains_any`,
  `regex`, `json_valid`, `length_between`, `starts_with`, each with a
  per-scorer pass threshold and partial credit where it makes sense
- 🧑‍⚖️ **LLM-as-judge scorer** — `llm_judge` scores an output against a
  rubric in plain language. The `fake` judge is deterministic and offline
  (expected-answer containment, topical overlap, length sanity — its
  rationale lists the components); the `openai` judge calls any
  OpenAI-compatible chat-completions endpoint and returns a parsed JSON
  verdict, scoring 0 rather than inventing a number when the reply
  cannot be parsed
- 🛡️ **Groundedness scorer** — the output must cover the source facts
  (`keywords`) and must not state any forbidden claim (`forbidden`); one
  invented fact caps the score at 0
- 🔌 **Pluggable providers** — the model under test is one small interface
  (`generate(prompt) -> str`); the deterministic offline provider keeps
  every run reproducible with no API key, and the `openai` provider
  (`OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL`) runs suites
  against a real hosted model at temperature 0
- 📊 **Run comparison** — compare any two runs of a suite: per-case score
  deltas, regressions (a drop of 0.10 or more, or a pass-to-fail flip),
  improvements, and added/removed cases; the CLI exits 1 on any
  regression, so CI can gate on it
- 📄 **HTML run report** — `GET /runs/{id}/html` renders a run as a single
  self-contained page (no JavaScript, no frontend build); the CLI
  `report` command writes the same page to a file
- 📡 **REST API** — create suites, run them, fetch scored runs and history
- 🖥️ **CLI runner** — `python -m evalharness.cli run suite.yaml` prints a
  per-case report and exits 1 when any case fails, so CI can gate on it
- 💾 **Run history** — suites and runs persist via SQLAlchemy (SQLite by
  default, `DATABASE_URL` switches to PostgreSQL)
- 🚦 **CI gating** — `.github/workflows/ci.yml` runs pytest, the all-passing
  gate suite (`examples/ci_gate_suite.yaml`, CLI exits 0 only when every
  case passes) and the self-eval on every push and pull request
- 🪞 **Self-eval** — `scripts/self_eval.py` runs the harness against
  itself: the test suite, both demo suites (which must fail loudly),
  the gate suite (which must pass), and run comparison in both
  directions; 7 checks, all must hold
- 🆔 **Stable suite identity** — a suite file always loads with the same
  ID (derived from its name), so runs produced by separate CLI
  invocations can be compared; an explicit `id` in the file still wins

## 🗓️ Build plan (30 / 30 / 40)

| Day | Share | Scope |
|-----|-------|-------|
| **1 — Mon 9 Nov 2026** | **30%** | Scaffold, models, deterministic scorers, offline provider, runner, API, CLI, first tests, README |
| **2 — Tue 10 Nov 2026** | **30% → 60%** | LLM-as-judge scorer, hosted provider via API key, run comparison / regression report, groundedness scorer, HTML results view, bulk of tests |
| **3 — Wed 11 Nov 2026** | **40% → 100%** | CI gate workflow + gate suite, self-eval script, stable suite IDs (bug fix), full test suite + self-eval, polish, publish to GitHub |

## 🏗️ Architecture

```
YAML/JSON suite ──▶ datasets.load_suite ──▶ EvalSuite
                                              │
POST /suites/{id}/runs ──▶ runner.run_suite ──▶ provider.generate(prompt)
                                              │        │
                                              │        ▼
                                              │   scorers (deterministic)
                                              ▼
                                    EvalRun (per-case scores, mean,
                                    pass/fail) ──▶ store (SQLite / PostgreSQL)
```

A case passes when **every** scorer passes. The run's mean score is the
mean of the cases' mean scores. Nothing is random: the same suite against
the same provider always produces the same scores.

## 🚀 Quick Start

```bash
git clone https://github.com/Vanoushika/llm-eval-harness.git
cd llm-eval-harness

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the sample suite (no API key needed):

```bash
python -m evalharness.cli run examples/sample_suite.yaml
```

Or start the API:

```bash
uvicorn evalharness.main:app --reload
# → http://localhost:8000  (interactive docs at /docs)
```

```bash
# Create a suite, then run it
curl -X POST http://localhost:8000/suites \
  -H 'Content-Type: application/json' \
  -d '{"name":"smoke","model":"offline","cases":[
        {"name":"capital","prompt_template":"What is the capital of France?",
         "expected":"Paris","scorers":[{"type":"contains","keywords":["Paris"]}]}]}'

curl -X POST http://localhost:8000/suites/<id>/runs
```

Or run with Docker:

```bash
docker build -t llm-eval-harness .
docker run -p 8000:8000 llm-eval-harness
```

## 📡 API Endpoints

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| GET | `/health` | Service health check → `{"status": "ok"}` |
| POST | `/suites` | Create a suite (`201`) |
| GET | `/suites` | List suites |
| GET | `/suites/{id}` | Fetch one suite |
| POST | `/suites/{id}/runs` | Run a suite; returns the scored run (`201`) |
| GET | `/runs` | List runs (optional `?suite_id=`) |
| GET | `/runs/compare?baseline=<id>&candidate=<id>` | Regression report between two runs of one suite |
| GET | `/runs/{id}` | Fetch one run with per-case scores |
| GET | `/runs/{id}/html` | One run as a readable HTML report |

## 🧑‍⚖️ LLM-as-judge and groundedness

```yaml
# in a suite file — see examples/judge_suite.yaml
scorers:
  - type: llm_judge
    rubric: "Score 1.0 if the output names the correct capital city of France, 0.0 otherwise."
    judge: fake        # or: openai (needs OPENAI_API_KEY)
    threshold: 0.5
  - type: groundedness
    keywords: ["red", "$19", "2 days"]          # source facts that must be covered
    forbidden: ["blue", "free shipping", "$29"]  # invented claims — any one scores 0
```

Run a suite against a hosted model instead of the offline stand-in:

```bash
OPENAI_API_KEY=sk-... python -m evalharness.cli run my_suite.yaml --model openai
# or pin a model: --model openai:gpt-4o
```

Compare two runs (regression report) and write an HTML report:

```bash
python -m evalharness.cli compare --baseline <run-id-or-file> --candidate <run-id-or-file>
# exits 1 when any case regressed by 0.10 or more, or flipped pass -> fail
python -m evalharness.cli report <run-id-or-file> --out report.html
```

## 🧪 Results — Day 3 (actually observed, 11 Nov 2026)

Test suite: **69 tests, 69 passed** (`pytest`, fully offline — 5 new tests
for suite loading and the comparison fix below).

Self-eval (`python scripts/self_eval.py`, fully offline): **7/7 checks
held** — pytest 69 passed / 0 failed; sample suite fails loudly
(3/4, exit 1); judge suite fails loudly (2/4, exit 1); CI gate suite
passes (3/3, exit 0); identical runs show no regression; a deliberately
worse run is flagged as a regression; CLI `compare` exits 0 on identical
runs.

One real bug found and fixed while writing the self-eval: `compare_runs`
requires matching suite IDs, but every `load_suite` call generated a fresh
random ID — so runs from two separate CLI invocations of the same suite
file could never be compared, which is exactly how a CI gate uses it.
Fix: a file-loaded suite now gets a deterministic ID derived from its
name (`datasets.stable_suite_id`); an explicit `id` in the file still
wins. Covered by 5 new tests in `tests/test_datasets.py`.

API smoke test the same day (live HTTP): suite created and run twice over
HTTP, 2/2 cases passed, mean score 1.0 on both runs; `GET /runs/compare`
returned mean delta 0.0 with 0 regressions; `GET /runs/{id}/html` served
the report page.

Docker: the Dockerfile is included and unchanged, but its build was **not
verified in this session** — no container runtime (Docker, Podman or
similar) is installed in this environment. The API and CLI paths above
were verified; the image build is the one unverified piece.

## 🧪 Results — Day 2 (actually observed, 10 Nov 2026)

Test suite: **64 tests, 64 passed** (`pytest`, fully offline — the hosted
provider and judge are tested through mocked HTTP transports, no API key
used).

Judge suite run (`examples/judge_suite.yaml`, offline provider + fake judge):

```
Suite: judge-and-groundedness-demo  (model: offline)
Result: 2/4 cases passed, mean score 0.51
  [PASS] judge-correct-answer — score 0.85
  [PASS] grounded-summary — score 1.00
  [FAIL] grounded-hallucinated-fact — score 0.00
  [FAIL] judge-wrong-answer — score 0.20
```

The two failures are deliberate: one stub invents facts the source note
never stated (the groundedness scorer catches it), one gives the wrong
answer (the fake judge scores it 0.20). Sample suite unchanged from day 1:
3/4 cases passed, mean score 0.75, exit code 1.

Comparison demo (same suite, baseline stub "Paris" vs candidate "London"):

```
Mean score delta: -0.25 (threshold for a regression: 0.1)
Regressions: 1
  [REGRESSED] capital-of-france: 1.00 -> 0.00 (-1.00)
```

API smoke test the same day (live HTTP): suite with an `llm_judge` and a
`groundedness` case created and run over HTTP, 2/2 passed, mean score 1.0;
`GET /runs/compare` returned the regression report and
`GET /runs/{id}/html` served the HTML report. The hosted OpenAI provider
and judge were not run against the live API in this session (no key
configured); they are exercised in tests via mocked HTTP.

## 🧪 Results — Day 1 (actually observed, 9 Nov 2026)

Test suite: **23 tests, 23 passed** (`pytest`, fully offline).

Sample suite run (`examples/sample_suite.yaml`, offline provider):

```
Suite: sample-factual-qa  (model: offline)
Result: 3/4 cases passed, mean score 0.75
  [PASS] capital-of-france — score 1.00
  [PASS] json-shape — score 1.00
  [PASS] arithmetic — score 1.00
  [FAIL] deliberately-wrong-expectation — score 0.00
```

The fourth case is deliberately wrong (it expects Berlin for the capital
of France) to prove the harness catches failures — and the CLI exits 1,
which is the behaviour a CI gate depends on. API smoke test the same day:
suite created and run over HTTP, 1/1 cases passed, mean score 1.0.

These are smoke-check results on a four-case sample set, included to show
the harness working end to end — not production quality figures.

## 📁 Project Structure

```
llm-eval-harness/
├── .github/workflows/ci.yml   # pytest + gate suite + self-eval on push/PR
├── evalharness/
│   ├── main.py          # FastAPI app and endpoints
│   ├── models.py        # Pydantic models: suites, cases, runs, results
│   ├── scorers.py       # Deterministic scorers + groundedness + llm_judge dispatch
│   ├── judge.py         # LLM-as-judge: fake (offline) and OpenAI-compatible judges
│   ├── providers.py     # Provider interface + offline and OpenAI providers
│   ├── runner.py        # Suite execution and scoring
│   ├── compare.py       # Run comparison / regression report
│   ├── report.py        # HTML run report (no frontend build)
│   ├── store.py         # SQLAlchemy persistence (SQLite / PostgreSQL)
│   ├── datasets.py      # Suite loading from YAML/JSON (stable suite IDs)
│   └── cli.py           # CLI runner with CI-friendly exit codes
├── examples/
│   ├── sample_suite.yaml
│   ├── judge_suite.yaml
│   └── ci_gate_suite.yaml   # all-passing suite CI gates on
├── scripts/
│   └── self_eval.py     # the harness evaluating itself (7 checks)
├── tests/               # pytest suite — runs fully offline
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 💡 Why I built this

My day job is building production Generative AI systems — RAG pipelines
and LangGraph agentic workflows — where a prompt tweak that quietly breaks
answers is worse than no change at all. Evaluation is the discipline that
catches that: deterministic checks first, a fixed regression suite on every
change, and a deploy gate that actually blocks. This project distils that
loop into a small, fully testable harness I can run anywhere, with no
hosted platform and no API key required to start.

## 📄 License

MIT
