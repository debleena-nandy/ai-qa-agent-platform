from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from app.models.execution_result import ExecutionResult


# What: Stores how a run is classified after execution.
# Why: QA results need a consistent structure to separate pass, failure, and not-run states.
@dataclass
class FailureReport:
    classification: str
    evidence: List[str] = field(default_factory=list)
    risk_level: str = "medium"


# What: Classifies a run from real step statuses, never from free-text keywords.
# Why: Keyword matching misclassified "No execution failures" as a failure.
# How: Failed steps win; a pass requires every scenario to pass; anything else is reported as not executed / partial.
class FailureAnalysisAgent:
    """Classifies and reports automation or application failures."""

    def analyze(self, execution_result: ExecutionResult) -> FailureReport:
        if execution_result.total_scenarios == 0:
            return FailureReport("not_executed", ["No scenarios were generated or executed."], "medium")

        failed_steps = [step for step in execution_result.steps if step.status == "failed"]
        if failed_steps:
            return FailureReport(
                classification="runtime_integration_failure",
                evidence=[f"{step.scenario}: {step.details or 'no details'}" for step in failed_steps],
                risk_level="high",
            )

        if execution_result.passed == execution_result.total_scenarios:
            return FailureReport("baseline_validation_pass", [], "low")

        if execution_result.not_executed == execution_result.total_scenarios:
            return FailureReport(
                "not_executed",
                [f"{execution_result.total_scenarios} scenarios generated; execution engine not connected yet."],
                "medium",
            )

        return FailureReport(
            "partially_executed",
            [
                f"passed={execution_result.passed}, skipped={execution_result.skipped}, "
                f"not_executed={execution_result.not_executed}"
            ],
            "medium",
        )
