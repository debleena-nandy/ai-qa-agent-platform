# File: tests/integration/test_api_run_lifecycle.py
# Description: Tests approval, execution, and retrieval across the QA run API lifecycle.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from demo_target.scenarios import DEMO_REQUIREMENT, demo_scenarios

TOKEN = "test-approval-token"
HEADERS = {"X-QA-Approval-Token": TOKEN}


def test_structured_run_approval_execution_and_retrieval(
    client: TestClient, demo_target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "qa_api_base_url", demo_target)
    monkeypatch.setattr(settings, "qa_approval_token", TOKEN)
    request = {
        "requirement": DEMO_REQUIREMENT,
        "scenarios": [s.model_dump(mode="json") for s in demo_scenarios()],
    }

    assert client.post("/runs", json=request).status_code == 403

    created_response = client.post("/runs", json=request, headers=HEADERS)
    assert created_response.status_code == 200, created_response.text
    created = created_response.json()
    assert created["run_status"] == "created"
    assert created["generation_mode"] == "manual"
    assert all(item["eligibility"] == "eligible" for item in created["test_plan"]["scenarios"])
    plan_id, run_id = created["test_plan"]["id"], created["run_id"]

    assert client.get(f"/plans/{plan_id}", headers=HEADERS).status_code == 200
    approved = client.post(f"/plans/{plan_id}/approval", json={"decision": "approved"}, headers=HEADERS)
    assert approved.status_code == 200, approved.text
    assert approved.json()["approval_status"] == "approved"

    executed = client.post(f"/runs/{run_id}/execute", headers=HEADERS)
    assert executed.status_code == 200, executed.text
    body = executed.json()
    assert body["run_status"] == "completed"
    assert body["execution_summary"]["passed"] == len(demo_scenarios())
    assert body["execution_summary"]["failed"] == 0
    assert body["execution_status"] == "completed"

    retrieved = client.get(f"/runs/{run_id}", headers=HEADERS)
    assert retrieved.status_code == 200
    assert retrieved.json()["qa_report"]["execution_status"] == "completed"
    assert client.get(f"/runs/{run_id}/report.md", headers=HEADERS).status_code == 200

    # A run can be executed only once.
    assert client.post(f"/runs/{run_id}/execute", headers=HEADERS).status_code == 409


def test_approval_rejects_plan_without_target(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "qa_approval_token", TOKEN)
    created = client.post(
        "/analyze", json={"requirement": "As a customer, I want to place an order to buy a toy"}
    ).json()
    response = client.post(
        f"/plans/{created['test_plan']['id']}/approval", json={"decision": "approved"}, headers=HEADERS
    )
    assert response.status_code == 409
    assert "target environment" in response.json()["detail"]


def test_plan_endpoints_require_configured_token(client: TestClient) -> None:
    created = client.post(
        "/analyze", json={"requirement": "As a customer, I want to place an order to buy a toy"}
    ).json()
    response = client.get(f"/plans/{created['test_plan']['id']}", headers=HEADERS)
    assert response.status_code == 503  # QA_APPROVAL_TOKEN is empty in tests