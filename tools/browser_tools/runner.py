"""Run structured browser scenarios and return test evidence."""
# File: tools/browser_tools/runner.py
# Description: Executes browser actions and records Playwright artifacts per scenario.
# Author Name: Debleena Nandy
# Date: 07-10-2026

from __future__ import annotations

from core.models.execution_result import AssertionOutcome, ExecutionResult, ExecutionStepResult
from core.models.scenario import QAScenario
from tools.api_tools.target_policy import ResolvedTarget
from tools.browser_tools.playwright_tool import PlaywrightBrowserTool
from tools.filesystem_tools.store import WorkspaceFileStore


class BrowserScenarioRunner:
    """Runs validated browser actions with Playwright and records evidence as artifacts."""

    def __init__(self, target: ResolvedTarget, tool: PlaywrightBrowserTool, artifacts: WorkspaceFileStore, run_prefix: str):
        self.target = target
        self.tool = tool
        self.artifacts = artifacts
        self.run_prefix = run_prefix

    def execute(self, scenario: QAScenario) -> ExecutionResult:
        step = ExecutionStepResult(
            scenario=f"{scenario.id} {scenario.title}", scenario_id=scenario.id,
            scenario_type=scenario.type, execution_type="browser",
        )
        result = ExecutionResult()
        if not scenario.browser_actions:
            step.status, step.details = "not_executed", "No structured browser actions are available."
            result.add_step(step)
            return result
        outcome = self.tool.run(self.target, scenario.browser_actions,
                                self.artifacts.resolve(f"{self.run_prefix}/{scenario.id}"))
        step.status = "passed" if outcome.passed else "failed"
        step.details = "All browser steps passed." if outcome.passed else outcome.failure
        step.error_type = outcome.error_type
        step.evidence = (outcome.evidence + [f"Console error: {e}" for e in outcome.console_errors[:5]]
                         + outcome.network_failures[:5])
        step.console_errors = outcome.console_errors
        step.http_statuses = outcome.http_statuses
        step.duration_ms = outcome.duration_ms
        step.artifacts = [self.artifacts.relative(path) for path in outcome.artifacts if path.exists()]
        if not outcome.passed and outcome.error_type == "AssertionError":
            step.assertions.append(AssertionOutcome(kind="browser", passed=False, description=outcome.failure or ""))
        result.add_step(step)
        return result