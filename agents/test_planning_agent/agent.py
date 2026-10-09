# File: agents/test_planning_agent/agent.py
# Description: Creates an executable test plan from QA scenarios.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from agents.execution_agent.routing import execution_type_for
from core.llm.agent_support import fallback_enabled, generate, run_llm_step
from core.llm.base import StructuredLLM
from core.llm.prompt_safety import wrap_untrusted
from core.models.scenario import QAScenario
from core.models.test_plan import UNCONFIGURED, ScenarioPlanItem, TestPlan

Items = List[ScenarioPlanItem]
Ordered = Tuple[List[ScenarioPlanItem], Dict[str, str]]
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}

PRIORITY_PROMPT = """You are a QA test manager. Order these test scenarios for execution by risk:
money, security, data integrity and acceptance criteria first; cosmetic checks last.
Return 'order' containing every scenario id exactly once, and 'risk_notes' (max 20 words each)
for the five riskiest scenarios.

{scenarios}
"""


class PrioritisedPlan(BaseModel):
    order: List[str]
    risk_notes: Dict[str, str] = Field(default_factory=dict)


class TestPlanningAgent:
    """Test Planning Agent. Eligibility is a deterministic safety gate (never decided by the LLM);
    the LLM performs risk-based prioritisation of the plan."""

    __test__ = False

    def __init__(self, llm: StructuredLLM | None, api_base_url: str = "", browser_enabled: bool = False,
                 use_fallback: bool | None = None):
        self.llm = llm  # None only for re-evaluating a single item during plan editing
        self.api_base_url = api_base_url
        self.browser_enabled = browser_enabled
        self.use_fallback = fallback_enabled(use_fallback)

    def plan_item(self, scenario: QAScenario, previous: Optional[ScenarioPlanItem] = None) -> ScenarioPlanItem:
        runner = execution_type_for(scenario)
        blockers = []
        if not self.api_base_url:
            blockers.append("Target environment is not configured.")
        if runner == "api" and not scenario.api_actions:
            blockers.append("Scenario has no structured, validated API actions.")
        if runner == "browser":
            if not self.browser_enabled:
                blockers.append("Browser runner is disabled (set BROWSER_ENABLED=true).")
            if not scenario.browser_actions:
                blockers.append("Scenario has no structured, validated browser actions.")
        item = ScenarioPlanItem(scenario_id=scenario.id, runner=runner, criterion_ids=scenario.criterion_ids,
                                eligibility="eligible" if not blockers else "blocked", blockers=blockers)
        if previous is not None and previous.eligibility == "excluded":
            item.eligibility, item.exclusion_reason = "excluded", previous.exclusion_reason
        return item

    def create_plan(self, scenarios: List[QAScenario]) -> TestPlan:
        items = [self.plan_item(scenario) for scenario in scenarios]
        plan = TestPlan(target_environment=self.api_base_url or UNCONFIGURED, scenarios=items)
        if self.llm is None or len(items) < 2:
            return plan
        fallback = (lambda: self._by_priority(items, scenarios)) if self.use_fallback else None
        (ordered, notes), mode = run_llm_step("test_planning_agent",
                                              lambda: self._prioritise(items, scenarios), fallback)
        for item in ordered:
            item.risk_note = notes.get(item.scenario_id)
        plan.scenarios, plan.prioritisation = ordered, mode
        return plan

    def _prioritise(self, items: Items, scenarios: List[QAScenario]) -> Ordered:
        assert self.llm is not None
        summary = [{"id": s.id, "title": s.title, "type": s.type, "priority": s.priority,
                    "criteria": s.criterion_ids} for s in scenarios]
        answer = generate(self.llm, PRIORITY_PROMPT.format(scenarios=wrap_untrusted(json.dumps(summary), "plan")),
                          PrioritisedPlan)
        by_id = {item.scenario_id: item for item in items}
        ordered = [by_id[sid] for sid in dict.fromkeys(answer.order) if sid in by_id]
        ordered += [item for item in items if item not in ordered]  # never drop a scenario
        notes = {sid: note[:160] for sid, note in answer.risk_notes.items() if sid in by_id}
        return ordered, notes

    @staticmethod
    def _by_priority(items: Items, scenarios: List[QAScenario]) -> Ordered:
        rank = {s.id: PRIORITY_RANK.get(s.priority, 1) for s in scenarios}
        return sorted(items, key=lambda item: rank.get(item.scenario_id, 1)), {}