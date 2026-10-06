from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

ScenarioType = Literal["functional", "negative", "boundary", "security", "api"]
Priority = Literal["high", "medium", "low"]


# What: One concrete, executable QA scenario.
# Why: A scenario title alone is not testable; steps and an expected result make it executable and reviewable.
# How: Shared by the LLM schema, the rule-based fallback, the executor, and the API response.
class QAScenario(BaseModel):
    id: str = Field(description="Sequential id such as TC-001")
    title: str = Field(description="Specific, domain-aware scenario title")
    type: ScenarioType
    priority: Priority = "medium"
    preconditions: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    expected_result: str
    automation_candidate: bool = True


class ScenarioSet(BaseModel):
    scenarios: List[QAScenario]


# What: Domain information extracted from a user story.
# Why: Scenarios are only specific when the generator knows the business domain and what it must cover.
class DomainProfile(BaseModel):
    domain: str = Field(description="Business domain, e.g. e-commerce, authentication, account-management")
    business_object: str = Field(description="Main thing the actor acts on, e.g. toy, account")
    key_concerns: List[str] = Field(
        default_factory=list,
        description="Areas a QA analyst must cover, e.g. inventory, payment, shipping",
    )
