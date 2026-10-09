# File: tests/unit/test_failure_analysis_agent.py
# Description: Tests classification of QA execution results and failures.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import pytest

from agents.failure_analysis_agent.agent import FailureAnalysisAgent
from core.llm.agent_support import AgentLLMError
from core.models.execution_result import ExecutionResult, ExecutionStepResult, derive_execution_status
from tests.fakes import DeadLLM


def _result(*statuses: str) -> ExecutionResult:
    result = ExecutionResult()
    for i, status in enumerate(statuses):
        result.add_step(ExecutionStepResult(scenario=f"TC-{i}", scenario_id=f"TC-{i}", status=status,
                                            execution_type="api", details="boom"))
    return result


def _agent() -> FailureAnalysisAgent:
    return FailureAnalysisAgent(DeadLLM(), use_fallback=True)


def test_all_passed_is_a_pass() -> None:
    report = _agent().analyze(_result("passed", "passed"))
    assert report.classification == "baseline_validation_pass"
    assert report.risk_level == "low"


def test_passed_and_skipped_is_still_a_pass_with_note() -> None:
    report = _agent().analyze(_result("passed", "skipped"))
    assert report.classification == "baseline_validation_pass"
    assert "excluded" in report.evidence[0]


def test_any_failure_is_detected_and_classified() -> None:
    report = _agent().analyze(_result("passed", "failed"))
    assert report.classification == "failures_detected"
    assert report.evidence == ["TC-1: boom"]
    assert len(report.step_classifications) == 1
    assert report.step_classifications[0].classification_source == "rule_based"
    assert report.risk_level in {"medium", "high"}


def test_failure_classification_raises_without_fallback() -> None:
    with pytest.raises(AgentLLMError):
        FailureAnalysisAgent(DeadLLM(), use_fallback=False).analyze(_result("failed"))


def test_nothing_executed_is_not_a_pass_or_failure() -> None:
    assert _agent().analyze(_result("not_executed", "not_executed")).classification == "not_executed"


def test_mixed_attempted_and_not_executed_is_partial() -> None:
    assert _agent().analyze(_result("passed", "not_executed")).classification == "partially_executed"


def test_empty_run_is_not_executed() -> None:
    assert _agent().analyze(ExecutionResult()).classification == "not_executed"


@pytest.mark.parametrize(
    "approved, statuses, expected",
    [(False, ("passed",), "not_started"), (True, (), "not_started"), (True, ("not_executed",), "not_started"),
     (True, ("passed", "failed"), "completed"), (True, ("passed", "not_executed"), "partial")],
)
def test_derive_execution_status(approved: bool, statuses: tuple, expected: str) -> None:
    assert derive_execution_status(approved, _result(*statuses)) == expected
