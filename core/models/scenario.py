# File: core/models/scenario.py
# Description: Defines QA scenarios, scenario sets, and domain profiles.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field
from core.models.actions import ApiRequestAction, BrowserAction

# Scenario types supported by the Test Design Agent.
ScenarioType = Literal[
    "functional", "positive", "negative", "boundary", "equivalence", "integration",
    "api", "ui", "security", "error_handling", "performance", "usability",
    "accessibility", "regression", "exploratory"]
Priority = Literal["high", "medium", "low"]
ActionSource = Literal["none", "rule_based", "llm", "manual"]

# What: One concrete, reviewable QA scenario.
# Why: Steps and an expected result make a scenario testable; structured actions are added separately for execution.
# How: Shared by the LLM schema, the rule-based fallback, the executor, and the API response.
class ScenarioDraft(BaseModel):
    """What the LLM may produce: reviewable scenario text, never executable actions."""
    id: str = Field(description="Sequential id such as TC-001")
    title: str = Field(description="Specific, domain-aware scenario title")
    type: ScenarioType
    priority: Priority = "medium"
    preconditions: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    expected_result: str
    criterion_ids: List[str] = Field(
            default_factory=list,
            description="Acceptance-criterion ids this scenario is designed to verify",
    )
    
class ScenarioDraftSet(BaseModel): 
    scenarios: List[ScenarioDraft]

class QAScenario(ScenarioDraft):
    """A scenario plus optional validated actions added by the action-design step or a reviewer."""
    api_actions: List[ApiRequestAction] = Field(
        default_factory=list,
        description="Validated API requests and assertions; an empty list means no executable API contract exists.",
    )
    browser_actions: List[BrowserAction] = Field(
        default_factory=list,
        description="Structured, reviewed browser actions; natural-language steps are never executed as code.",
    )
    automation_candidate: bool = True
    action_source: ActionSource = "none"


class ScenarioSet(BaseModel):
    scenarios: List[QAScenario]

# What: Domain information extracted from a user story.
# Why: Scenarios are only specific when the generator knows the business domain and what it must cover.
class DomainProfile(BaseModel):
    domain: str = Field(description="Business domain, e.g. e-commerce, authentication, account-management")
    business_object: str = Field(description="Main thing the actor acts on, e.g. toy, account")
    key_concerns: List[str] = Field(
        default_factory=list,
        description="Areas a QA analyst must cover",
    )
