# File: tests/e2e/test_api_e2e.py
# Description: Tests complete API endpoint behavior through the public application interface.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from tests.conftest import MIN_SCENARIOS

TOY = "As a customer, I want to place an order to buy a toy"


class HealthyLLM:
    def generate_structured(self, prompt: str, schema: type) -> Any:
        raise AssertionError("not used")

    def health(self, timeout: float = 3) -> Dict[str, Any]:
        return {"reachable": True, "model_available": True, "model": "qwen2.5:3b"}

    def close(self) -> None:
        pass


def test_health_endpoint_reports_ok_when_model_is_ready(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("api.main.build_llm_client", lambda: HealthyLLM())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_workflow_endpoint_describes_both_graphs(client: TestClient) -> None:
    body = client.get("/workflow").json()
    assert body["engine"] in {"langgraph", "local"}
    assert body["design_graph"] and body["execution_graph"]


def test_analyze_returns_structured_domain_scenarios(client: TestClient) -> None:
    response = client.post("/analyze", json={"requirement": TOY})
    assert response.status_code == 200
    body = response.json()
    assert body["run_status"] == "created"
    assert body["actors"] == ["customer"]
    assert body["domain"] == "e-commerce"
    assert body["business_object"] == "toy"
    assert body["scenario_count"] == len(body["generated_scenarios"]) >= MIN_SCENARIOS
    assert body["execution_status"] == "not_started"
    assert body["test_plan"]["approval_status"] == "pending"
    assert body["test_plan"]["scenarios"][0]["eligibility"] == "blocked"  # no target configured
    assert body["qa_report"]["scenario_count"] == body["scenario_count"]
    assert body["root_cause"]["confidence"] is None
    first = body["generated_scenarios"][0]
    assert {"id", "title", "type", "priority", "preconditions", "steps", "expected_result"} <= first.keys()
    assert body["criteria"] == []
    assert body["execution_summary"]["total_scenarios"] == body["scenario_count"]
    assert body["failure_classification"] == "not_executed"


def test_analyze_exposes_acceptance_criteria_traceability(client: TestClient) -> None:
    requirement = TOY + "\nAcceptance criteria:\n- Customer can order at most 5 units per toy"
    body = client.post("/analyze", json={"requirement": requirement}).json()
    assert body["criteria"] == [{"id": "AC-001", "description": "Customer can order at most 5 units per toy"}]
    assert any("AC-001" in s["criterion_ids"] for s in body["generated_scenarios"])


def test_whitespace_payload_returns_422(client: TestClient) -> None:
    assert client.post("/analyze", json={"requirement": " " * 12}).status_code == 422


def test_missing_payload_field_returns_422(client: TestClient) -> None:
    assert client.post("/analyze", json={}).status_code == 422