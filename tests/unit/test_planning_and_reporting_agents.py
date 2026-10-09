# File: tests/unit/test_planning_and_reporting_agents.py
# Description: Tests QA test planning and report generation behavior.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import pytest

from agents.execution_agent.agent import ExecutionAgent
from agents.failure_analysis_agent.agent import FailureAnalysisAgent
from agents.reporting_agent.agent import QAReportAgent
from agents.root_cause_agent.agent import RootCauseAgent
from agents.test_planning_agent.agent import TestPlanningAgent
from core.models.actions import ApiAssertion, ApiRequestAction
from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario, Priority
from core.models.test_plan import UNCONFIGURED
from tests.fakes import DeadLLM, DemoFakeLLM


def _api_scenario(sid: str = "TC-001", priority: Priority = "medium", actions: bool = False) -> QAScenario:
    api_actions = [ApiRequestAction(method="POST", path="/orders",
                                    assertions=[ApiAssertion(type="status_code", expected=201)])] if actions else []
    return QAScenario(id=sid, title=f"Create an order {sid}", type="api", priority=priority,
                      steps=["POST /orders with a valid order"], expected_result="The order is created",
                      criterion_ids=["AC-001"], api_actions=api_actions)


def test_planning_blocks_scenarios_without_target_or_actions() -> None:
    plan = TestPlanningAgent(None).create_plan([_api_scenario()])
    item = plan.scenarios[0]
    assert plan.approval_status == "pending"
    assert plan.target_environment == UNCONFIGURED
    assert (item.runner, item.criterion_ids, item.eligibility) == ("api", ["AC-001"], "blocked")
    assert "Target environment is not configured." in item.blockers
    assert plan.approval_blockers()


def test_planning_marks_scenario_with_actions_and_target_eligible() -> None:
    plan = TestPlanningAgent(None, "http://127.0.0.1:8001").create_plan([_api_scenario(actions=True)])
    assert plan.scenarios[0].eligibility == "eligible"
    assert plan.approval_blockers() == []


def test_llm_prioritises_high_risk_first() -> None:
    scenarios = [_api_scenario("TC-001", "low"), _api_scenario("TC-002", "high")]
    plan = TestPlanningAgent(DemoFakeLLM()).create_plan(scenarios)
    assert plan.prioritisation == "llm"
    assert plan.scenarios[0].scenario_id == "TC-002"


def test_api_action_contract_rejects_external_or_traversal_paths() -> None:
    for path in ("https://example.com/orders", "//example.com/orders", "/../admin"):
        with pytest.raises(ValueError):
            ApiRequestAction(method="GET", path=path, assertions=[ApiAssertion(type="status_code", expected=200)])


def test_api_assertions_validate_required_json_path() -> None:
    with pytest.raises(ValueError, match="requires json_path"):
        ApiAssertion(type="json_equals", expected="created")


def test_execution_agent_does_not_execute_unapproved_plan() -> None:
    scenario = _api_scenario(actions=True)
    plan = TestPlanningAgent(None, "http://127.0.0.1:8001").create_plan([scenario])
    result = ExecutionAgent().execute(plan, [scenario])
    assert result.not_executed == 1
    assert result.passed == result.failed == 0


def test_report_exposes_coverage_and_never_claims_root_cause_without_evidence() -> None:
    llm = DeadLLM()
    scenario = QAScenario(id="TC-001", title="Verify order limit", type="boundary", steps=["Set quantity to 6"],
                          expected_result="The limit is enforced", criterion_ids=["AC-001"])
    criterion = AcceptanceCriterion(id="AC-001", description="Orders are limited to five items")
    plan = TestPlanningAgent(None).create_plan([scenario])
    execution = ExecutionAgent().execute(plan, [scenario])
    failure = FailureAnalysisAgent(llm, use_fallback=True).analyze(execution)
    root_cause = RootCauseAgent(llm, use_fallback=True).analyze(execution, failure)
    report = QAReportAgent(llm, use_fallback=True).generate(
        plan=plan, criteria=[criterion], scenarios=[scenario], execution=execution, failure=failure,
        root_cause=root_cause, workflow_status="completed", execution_status="not_started")
    assert report.covered_criterion_ids == ["AC-001"]
    assert report.uncovered_criterion_ids == []
    assert report.execution_summary == {"total_scenarios": 1, "passed": 0, "failed": 0, "skipped": 0,
                                        "not_executed": 1}
    assert report.pass_rate is None
    assert report.root_cause.confidence is None
    assert "No scenario execution evidence" in report.root_cause.hypothesis
    assert "Review and approve the test plan before enabling execution." in report.recommendations