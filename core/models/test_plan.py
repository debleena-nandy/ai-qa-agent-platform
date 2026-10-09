# File: core/models/test_plan.py
# Description: Defines test plan and per-scenario planning models.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations
from typing import List, Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

PlanApprovalStatus = Literal["pending", "approved", "rejected"]
PlanEligibility = Literal["eligible", "blocked", "excluded"]
RunnerType = Literal["api", "browser"]
Problems = List[str]
UNCONFIGURED = "unconfigured"

class ScenarioPlanItem(BaseModel):
    scenario_id: str
    runner: RunnerType
    criterion_ids: List[str] = Field(default_factory=list)
    eligibility: PlanEligibility
    blockers: List[str] = Field(default_factory=list)
    exclusion_reason: Optional[str] = None
    risk_note: Optional[str] = None

class TestPlan(BaseModel):
    __test__ = False
    id: str = Field(default_factory=lambda: f"PLAN-{uuid4()}")
    approval_status: PlanApprovalStatus = "pending"
    target_environment: str = UNCONFIGURED
    scenarios: List[ScenarioPlanItem] = Field(default_factory=list)
    prioritisation: Literal["llm", "rule_based", "none"] = "none"

    @property
    def is_target_configured(self) -> bool:
        return bool(self.target_environment) and self.target_environment != UNCONFIGURED
    
    def approval_blockers(self) -> Problems:
        problems = []
        if not self.is_target_configured:
            problems.append("Cannot approve a plan without a configured target environment.")
        if not any(item.eligibility == "eligible" for item in self.scenarios):
            problems.append("Cannot approve a plan with no eligible scenarios.")
        blocked = [item.scenario_id for item in self.scenarios if item.eligibility == "blocked"]
        if blocked:
            problems.append(f"Resolve or exclude all blocked scenarios before approving the plan ({len(blocked)} blocked).")
        return problems

