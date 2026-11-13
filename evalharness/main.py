"""FastAPI app — suites in, scored runs out.

Endpoints:
  GET  /health            service health
  POST /suites            create a suite
  GET  /suites            list suites
  GET  /suites/{id}       one suite
  POST /suites/{id}/runs  run a suite against its configured provider
  GET  /runs              list runs (optional ?suite_id=)
  GET  /runs/compare      compare two runs (?baseline=&candidate=)
  GET  /runs/{id}         one run with per-case scores
  GET  /runs/{id}/html    one run as a readable HTML report
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .compare import RunComparison, compare_runs
from .datasets import load_suite  # noqa: F401  (re-exported for the CLI/tests)
from .models import EvalRun, EvalSuite
from .providers import get_provider
from .report import render_run_html
from .runner import run_suite
from .store import Store

app = FastAPI(
    title="LLM Eval Harness",
    description="Self-hosted evaluation harness: deterministic scorers, "
    "regression suites and run history for LLM-powered features.",
    version="0.1.0",
)

store = Store()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "llm-eval-harness", "version": "0.1.0"}


@app.post("/suites", response_model=EvalSuite, status_code=201)
def create_suite(suite: EvalSuite) -> EvalSuite:
    return store.save_suite(suite)


@app.get("/suites", response_model=list[EvalSuite])
def list_suites() -> list[EvalSuite]:
    return store.list_suites()


@app.get("/suites/{suite_id}", response_model=EvalSuite)
def get_suite(suite_id: str) -> EvalSuite:
    suite = store.get_suite(suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {suite_id!r} not found")
    return suite


@app.post("/suites/{suite_id}/runs", response_model=EvalRun, status_code=201)
def run_suite_endpoint(suite_id: str) -> EvalRun:
    suite = store.get_suite(suite_id)
    if suite is None:
        raise HTTPException(status_code=404, detail=f"suite {suite_id!r} not found")
    try:
        provider = get_provider(suite.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    run = run_suite(suite, provider)
    return store.save_run(run)


@app.get("/runs", response_model=list[EvalRun])
def list_runs(suite_id: str | None = None) -> list[EvalRun]:
    return store.list_runs(suite_id)


@app.get("/runs/compare", response_model=RunComparison)
def compare_runs_endpoint(
    baseline: str, candidate: str, threshold: float = 0.10
) -> RunComparison:
    baseline_run = store.get_run(baseline)
    if baseline_run is None:
        raise HTTPException(status_code=404, detail=f"baseline run {baseline!r} not found")
    candidate_run = store.get_run(candidate)
    if candidate_run is None:
        raise HTTPException(status_code=404, detail=f"candidate run {candidate!r} not found")
    try:
        return compare_runs(baseline_run, candidate_run, regression_threshold=threshold)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/runs/{run_id}", response_model=EvalRun)
def get_run(run_id: str) -> EvalRun:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return run


@app.get("/runs/{run_id}/html", response_class=HTMLResponse)
def get_run_html(run_id: str) -> str:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return render_run_html(run)
