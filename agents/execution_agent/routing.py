# File: agents/execution_agent/routing.py
# Description: Runs QA scenarios and aggregates their execution results.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import re

from core.models.scenario import QAScenario
from core.models.test_plan import RunnerType

# HTTP verbs are matched case-sensitively so "Put the item in the cart" stays a browser step,
# while "PUT the profile" and "GET /orders/1" are API calls.
HTTP_STEP = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+(?:/|the\b|a\b)")
API_HINT = re.compile(r"\b(?:api|token|payload|status code|request)\b", re.IGNORECASE)
BROWSER_HINT = re.compile(r"\b(?:page|browser|click|button|screen|ui)\b", re.IGNORECASE)


def execution_type_for(scenario: QAScenario) -> RunnerType:
    # What: Picks the runner for a scenario.
    # Why: Security scenarios like "GET /orders/<id> with another user's token" need the API runner;
    #      "Open the API docs page in the browser" must not be sent to it.
    # How: Explicit HTTP verbs win; otherwise API hints count only when no step mentions the UI.    
    if scenario.browser_actions or scenario.type == "ui":
        return "browser"
    if scenario.api_actions or scenario.type == "api":
        return "api"
    steps = " ".join(scenario.steps)
    if HTTP_STEP.search(steps):
        return "api"
    if API_HINT.search(steps) and not BROWSER_HINT.search(steps):
        return "api"
    return "browser"

'''
class QAExecutor:
    """Executes generated scenario steps and records structured evidence."""

    def run_scenarios(self, scenarios: List[QAScenario]) -> ExecutionResult:
        result = ExecutionResult()
        for scenario in scenarios:
            result.add_step(
                ExecutionStepResult(
                    scenario=f"{scenario.id} {scenario.title}",
                    status="not_executed",
                    execution_type=execution_type_for(scenario),
                    evidence=[],
                    details="Execution engine not connected yet (roadmap: Playwright and API runners).",
                )
            )
        return result
        '''
