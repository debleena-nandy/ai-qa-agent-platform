# File: tests/unit/test_workflow.py
# Description: Tests orchestration and state transitions of the QA workflow.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import pytest

from graph.engine import END, GraphSpec
from graph.workflow import QAWorkflow
from tests.conftest import MIN_SCENARIOS
from tests.fakes import DeadLLM, DemoFakeLLM

TOY = "As a customer, I want to place an order to buy a toy"


def test_workflow_produces_domain_specific_report() -> None:
    state = QAWorkflow(DemoFakeLLM()).run(TOY)
    assert state.status == "completed"
    assert state.title.startswith("As a customer")
    assert state.actors == ["customer"]
    assert (state.domain, state.business_object) == ("e-commerce", "toy")
    assert state.generation_mode == "llm"
    assert len(state.generated_scenarios) >= MIN_SCENARIOS
    assert state.execution_status == "not_started"
    assert state.test_plan.approval_status == "pending"
    assert all(item.eligibility == "blocked" for item in state.test_plan.scenarios)
    assert state.qa_report is not None
    assert state.qa_report.scenario_count == len(state.generated_scenarios)
    assert state.root_cause is not None and state.root_cause.confidence is None


def test_execution_summary_adds_up() -> None:
    state = QAWorkflow(DemoFakeLLM()).run("As a customer, I want to log in securely.")
    summary = state.execution_summary
    assert summary["total_scenarios"] == len(state.generated_scenarios)
    assert sum(summary[k] for k in ("passed", "failed", "skipped", "not_executed")) == summary["total_scenarios"]
    assert summary["failed"] == 0
    assert state.failure_classification == "not_executed"


def test_workflow_uses_injected_llm(fake_llm_factory) -> None:
    state = QAWorkflow(fake_llm_factory(scenario_count=15)).run(TOY)
    assert state.generation_mode == "llm"
    assert len(state.generated_scenarios) == 15
    assert state.execution_status == "not_started"


def test_workflow_rule_based_fallback() -> None:
    state = QAWorkflow(DeadLLM(), use_fallback=True).run(TOY)
    assert state.generation_mode == "rule_based"
    assert state.domain == "e-commerce"


def test_workflow_rejects_empty_requirement() -> None:
    with pytest.raises(ValueError, match="Requirement text cannot be empty"):
        QAWorkflow(DemoFakeLLM()).run("   ")


def test_workflow_describe_lists_both_graphs() -> None:
    graphs = QAWorkflow(DemoFakeLLM()).describe()
    assert "plan_tests -[await_approval]-> record_not_executed" in graphs["design_graph"]
    assert graphs["execution_graph"][-1] == "write_artifacts -> END"


def test_local_engine_runs_nodes_and_routes() -> None:
    spec = (GraphSpec(dict)
            .add_node("a", lambda s: {"x": 1})
            .add_node("b", lambda s: {"y": s["x"] + 1})
            .add_node("c", lambda s: {"z": "skipped"})
            .set_entry("a")
            .add_conditional_edges("a", lambda s: "go" if s["x"] == 1 else "skip", {"go": "b", "skip": "c"})
            .add_edge("b", END).add_edge("c", END))
    assert spec.compile(prefer_langgraph=False)({}) == {"x": 1, "y": 2}