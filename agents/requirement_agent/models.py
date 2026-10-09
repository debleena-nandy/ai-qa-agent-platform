# File: agents/requirement_agent/models.py
# Description: Detects and normalizes the model of a requirement agents.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from pydantic import BaseModel, Field

from core.models.requirement import AcceptanceCriterion

Criteria = List[AcceptanceCriterion]


@dataclass
class RequirementAnalysis:
    title: str
    actors: List[str] = field(default_factory=list)
    goal: str = ""
    business_object: str = ""
    benefit: str = ""
    business_rules: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    testable_conditions: List[str] = field(default_factory=list)
    security_flags: List[str] = field(default_factory=list)

    @property
    def criteria(self) -> Criteria:
        return [AcceptanceCriterion(id=f"AC-{i:03d}", description=d)
                for i, d in enumerate(self.acceptance_criteria, start=1)]


class RequirementExtraction(BaseModel):
    """Structured output schema for the Requirement Analysis Agent."""

    title: str = Field(description="Short title of the story, at most 120 characters")
    actors: List[str] = Field(default_factory=list, description="Personas, e.g. customer")
    goal: str = Field(description="What the actor wants to do")
    business_object: str = Field(description="Main thing the actor acts on, 1-3 words")
    benefit: str = Field(default="", description="The 'so that' value, empty if not stated")
    business_rules: List[str] = Field(default_factory=list, description="Explicit rules only")
    acceptance_criteria: List[str] = Field(default_factory=list,
                                           description="Only criteria written in the story, copied verbatim")
    missing_requirements: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    testable_conditions: List[str] = Field(default_factory=list)
    domain: str = Field(description="Business domain, e.g. e-commerce, authentication, payments")
    key_concerns: List[str] = Field(default_factory=list, description="5-10 areas a QA analyst must test")