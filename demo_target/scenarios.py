# File: demo_target/scenarios.py
# Description: Builds example QA scenarios for the demo order API.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

from typing import List

from core.models.actions import ApiAssertion, ApiRequestAction
from core.models.scenario import QAScenario

Scenarios = List[QAScenario]

DEMO_REQUIREMENT = (
    "As a customer, I want to place an order to buy a toy\n"
    "Acceptance criteria:\n"
    "- Customer can order at most 5 units per toy"
)
AUTH = {"Authorization": "Bearer demo-customer-token"}


def demo_scenarios() -> Scenarios:
    """Hand-written scenarios with actions, for POST /runs (manual path)."""
    return [
        QAScenario(
            id="TC-001", title="Order 2 units of an in-stock toy", type="api", priority="high",
            steps=["POST /orders with product toy-001 and quantity 2"],
            expected_result="201 with status confirmed", criterion_ids=["AC-001"],
            api_actions=[ApiRequestAction(
                method="POST", path="/orders", headers=AUTH, json_body={"product_id": "toy-001", "quantity": 2},
                assertions=[ApiAssertion(type="status_code", expected=201),
                            ApiAssertion(type="json_equals", json_path="status", expected="confirmed")])],
        ),
        QAScenario(
            id="TC-002", title="Quantity 6 is rejected", type="boundary", priority="high",
            steps=["POST /orders with quantity 6"], expected_result="422", criterion_ids=["AC-001"],
            api_actions=[ApiRequestAction(
                method="POST", path="/orders", headers=AUTH, json_body={"product_id": "toy-001", "quantity": 6},
                assertions=[ApiAssertion(type="status_code", expected=422)])],
        ),
    ]