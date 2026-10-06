from __future__ import annotations

from app.agents.failure_analysis_agent import FailureAnalysisAgent
from app.models.execution_result import ExecutionResult, ExecutionStepResult


def _result(*statuses: str) -> ExecutionResult:
    result = ExecutionResult()
    for i, status in enumerate(statuses):
        result.add_step(ExecutionStepResult(scenario=f"TC-{i}", status=status, details="boom"))
    return result


def test_all_passed_is_a_pass() -> None:
    report = FailureAnalysisAgent().analyze(_result("passed", "passed"))
    assert report.classification == "baseline_validation_pass"
    assert report.risk_level == "low"


def test_any_failure_is_high_risk() -> None:
    report = FailureAnalysisAgent().analyze(_result("passed", "failed"))
    assert report.classification == "runtime_integration_failure"
    assert report.risk_level == "high"
    assert report.evidence == ["TC-1: boom"]


def test_nothing_executed_is_not_a_pass_or_failure() -> None:
    report = FailureAnalysisAgent().analyze(_result("not_executed", "not_executed"))
    assert report.classification == "not_executed"


def test_mixed_statuses_are_partial() -> None:
    report = FailureAnalysisAgent().analyze(_result("passed", "skipped"))
    assert report.classification == "partially_executed"


def test_empty_run_is_not_executed() -> None:
    assert FailureAnalysisAgent().analyze(ExecutionResult()).classification == "not_executed"
