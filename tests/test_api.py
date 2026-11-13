"""API tests — suites in, scored runs out, over HTTP."""

SUITE_PAYLOAD = {
    "name": "api-demo",
    "description": "created through the API",
    "model": "offline",
    "cases": [
        {
            "name": "capital",
            "prompt_template": "What is the capital of France?",
            "expected": "Paris",
            "scorers": [{"type": "contains", "keywords": ["Paris"]}],
        },
        {
            "name": "wrong-expectation",
            "prompt_template": "What is the capital of France?",
            "expected": "Berlin",
            "scorers": [{"type": "exact_match"}],
        },
    ],
}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_and_fetch_suite(client):
    created = client.post("/suites", json=SUITE_PAYLOAD)
    assert created.status_code == 201
    suite_id = created.json()["id"]

    fetched = client.get(f"/suites/{suite_id}")
    assert fetched.status_code == 200
    assert fetched.json()["name"] == "api-demo"
    assert len(fetched.json()["cases"]) == 2


def test_list_suites(client):
    client.post("/suites", json=SUITE_PAYLOAD)
    response = client.get("/suites")
    assert response.status_code == 200
    assert any(s["name"] == "api-demo" for s in response.json())


def test_run_suite_returns_scored_run(client):
    suite_id = client.post("/suites", json=SUITE_PAYLOAD).json()["id"]
    run = client.post(f"/suites/{suite_id}/runs")
    assert run.status_code == 201
    body = run.json()
    assert body["suite_id"] == suite_id
    assert body["passed_count"] == 1
    assert body["total_count"] == 2
    assert body["results"][0]["passed"] is True
    assert body["results"][1]["passed"] is False


def test_get_run_by_id(client):
    suite_id = client.post("/suites", json=SUITE_PAYLOAD).json()["id"]
    run_id = client.post(f"/suites/{suite_id}/runs").json()["id"]
    fetched = client.get(f"/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == run_id


def test_missing_suite_returns_404(client):
    assert client.get("/suites/does-not-exist").status_code == 404
    assert client.post("/suites/does-not-exist/runs").status_code == 404
    assert client.get("/runs/does-not-exist").status_code == 404


def test_compare_endpoint_flags_regression(client):
    suite_id = client.post("/suites", json=SUITE_PAYLOAD).json()["id"]
    baseline_id = client.post(f"/suites/{suite_id}/runs").json()["id"]

    # Change the stub so the first case now fails, then run again.
    updated = dict(SUITE_PAYLOAD)
    updated["id"] = suite_id
    updated["cases"] = [
        {
            "name": "capital",
            "prompt_template": "What is the capital of France?",
            "expected": "Paris",
            "stub_output": "London",
            "scorers": [{"type": "contains", "keywords": ["Paris"]}],
        },
        SUITE_PAYLOAD["cases"][1],
    ]
    client.post("/suites", json=updated)
    candidate_id = client.post(f"/suites/{suite_id}/runs").json()["id"]

    response = client.get(
        "/runs/compare", params={"baseline": baseline_id, "candidate": candidate_id}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["regression_count"] == 1
    assert body["regressions"][0]["case_name"] == "capital"
    assert body["has_regressions"] is True


def test_compare_endpoint_rejects_different_suites(client):
    first = client.post("/suites", json=SUITE_PAYLOAD).json()["id"]
    second_payload = dict(SUITE_PAYLOAD)
    second_payload["name"] = "other-suite"
    second = client.post("/suites", json=second_payload).json()["id"]
    run_a = client.post(f"/suites/{first}/runs").json()["id"]
    run_b = client.post(f"/suites/{second}/runs").json()["id"]
    response = client.get(
        "/runs/compare", params={"baseline": run_a, "candidate": run_b}
    )
    assert response.status_code == 400


def test_compare_endpoint_missing_run_returns_404(client):
    response = client.get(
        "/runs/compare", params={"baseline": "nope", "candidate": "also-nope"}
    )
    assert response.status_code == 404


def test_html_report_endpoint(client):
    suite_id = client.post("/suites", json=SUITE_PAYLOAD).json()["id"]
    run_id = client.post(f"/suites/{suite_id}/runs").json()["id"]
    response = client.get(f"/runs/{run_id}/html")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "api-demo" in response.text
    assert "capital" in response.text
    assert client.get("/runs/does-not-exist/html").status_code == 404
