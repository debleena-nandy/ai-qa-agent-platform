# File: agents/execution_agent/agent.py
# Description: Coordinates execution of a test plan across supported runners.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30


from __future__ import annotations

from typing import List

from core.models.execution_result import ExecutionResult, ExecutionStepResult
from core.models.scenario import QAScenario
from core.models.test_plan import TestPlan
from tools.api_tools.runner import ApiScenarioRunner
from tools.browser_tools.runner import BrowserScenarioRunner


class ExecutionAgent:
    """Execution Agent: enforces approval and eligibility, then dispatches in the (risk-ordered) plan order.
    Execution is deliberately deterministic: the LLM never decides whether a request is sent."""

    def __init__(self, api_runner: ApiScenarioRunner | None = None,
                 browser_runner: BrowserScenarioRunner | None = None):
        self.api_runner = api_runner
        self.browser_runner = browser_runner

    def execute(self, plan: TestPlan, scenarios: List[QAScenario]) -> ExecutionResult:
        by_id = {s.id: s for s in scenarios}
        planned = [item.scenario_id for item in plan.scenarios if item.scenario_id in by_id]
        order = planned + [s.id for s in scenarios if s.id not in planned]
        items = {item.scenario_id: item for item in plan.scenarios}
        result = ExecutionResult()
        for scenario_id in order:
            scenario, item = by_id[scenario_id], items.get(scenario_id)
            status, details = "not_executed", None
            if item is None:
                details = "Scenario is absent from the approved test plan."
            elif item.eligibility == "excluded":
                status, details = "skipped", f"Excluded by reviewer: {item.exclusion_reason or 'no reason given'}."
            elif plan.approval_status != "approved":
                details = f"Test plan approval is {plan.approval_status}; execution is blocked."
            elif not plan.is_target_configured:
                details = "Target environment is not configured; execution is blocked."
            elif item.eligibility != "eligible":
                details = "; ".join(item.blockers) or "Scenario is not eligible for execution."
            else:
                runner = self.api_runner if item.runner == "api" else self.browser_runner
                if runner is not None:
                    result.extend(runner.execute(scenario))
                    continue
                details = f"No {item.runner} runner is configured."
            result.add_step(ExecutionStepResult(
                scenario=f"{scenario.id} {scenario.title}", scenario_id=scenario.id, scenario_type=scenario.type,
                status=status, execution_type=item.runner if item else "api", details=details))
        return result