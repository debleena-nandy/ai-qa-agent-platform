from __future__ import annotations

import pytest

from app.graph.workflow import QAWorkflow


def test_workflow_produces_domain_specific_report() -> None:
    state = QAWorkflow().run("As a customer, I want to place an order to buy a toy")

    assert state.status == "completed"
    assert state.title.startswith("As a customer")
    assert state.actors == ["customer"]
    assert state.domain == "e-commerce"
    assert state.business_object == "toy"
    assert state.generation_mode == "rule_based"
    assert len(state.generated_scenarios) >= 15


def test_execution_summary_adds_up() -> None:
    state = QAWorkflow().run("As a customer, I want to log in securely.")
    summary = state.execution_summary

    assert summary["total_scenarios"] == len(state.generated_scenarios)
    assert summary["passed"] + summary["failed"] + summary["skipped"] + summary["not_executed"] == summary["total_scenarios"]
    assert summary["failed"] == 0
    assert state.failure_classification == "not_executed"


def test_workflow_uses_injected_llm(fake_llm_factory) -> None:
    state = QAWorkflow(llm_client=fake_llm_factory(scenario_count=10)).run(
        "As a customer, I want to place an order to buy a toy"
    )

    assert state.generation_mode == "llm"
    assert len(state.generated_scenarios) == 10


def test_workflow_rejects_empty_requirement() -> None:
    with pytest.raises(ValueError, match="Requirement text cannot be empty"):
        QAWorkflow().run("   ")
