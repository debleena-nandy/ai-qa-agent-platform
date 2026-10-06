from __future__ import annotations

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_returns_structured_domain_scenarios(client: TestClient) -> None:
    response = client.post("/analyze", json={"requirement": "As a customer, I want to place an order to buy a toy"})

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "As a customer, I want to place an order to buy a toy"
    assert body["actors"] == ["customer"]
    assert body["domain"] == "e-commerce"
    assert body["business_object"] == "toy"
    assert body["scenario_count"] == len(body["generated_scenarios"]) >= 15

    first = body["generated_scenarios"][0]
    assert {"id", "title", "type", "priority", "preconditions", "steps", "expected_result"} <= first.keys()
    assert body["execution_summary"]["total_scenarios"] == body["scenario_count"]
    assert body["failure_classification"] == "not_executed"


def test_whitespace_payload_returns_422(client: TestClient) -> None:
    assert client.post("/analyze", json={"requirement": " " * 12}).status_code == 422


def test_missing_payload_field_returns_422(client: TestClient) -> None:
    assert client.post("/analyze", json={}).status_code == 422
