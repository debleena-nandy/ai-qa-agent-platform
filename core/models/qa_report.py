# File: core/models/qa_report.py
# Description: Defines structured QA reports, plans, and root-cause assessments.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

FailureCategory = Literal[
    "application_defect",
    "automation_defect",
    "test_data_issue",
    "environment_issue",
    "dependency_issue",
    "network_issue",
    "configuration_issue",
    "unknown",
]


class StepClassification(BaseModel):
    scenario_id: str
    scenario: str
    category: FailureCategory
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence: List[str] = Field(default_factory=list)
    classification_source: Literal["llm", "rule_based"] = "llm"

class RootCauseAssessment(BaseModel):
    category: FailureCategory = "unknown"
    hypothesis: str
    evidence: List[str] = Field(default_factory=list)
    suggested_investigation: List[str] = Field(default_factory=list)
    suggested_remediation: List[str] = Field(default_factory=list)
    similar_known_issues: List[str] = Field(default_factory=list)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    confidence_basis: str
    hypothesis_source: Literal["rule_based", "llm"] = "rule_based"

class DefectCandidate(BaseModel):
    scenario_id: str
    title: str
    category: FailureCategory
    confidence: float
    evidence: List[str] = Field(default_factory=list)
    note: str = "Candidate only: requires human QA confirmation before a defect is raised."

class QAReport(BaseModel):
    plan_id: str
    workflow_status: str
    execution_status: str
    scenario_count: int
    criterion_count: int
    covered_criterion_ids: List[str] = Field(default_factory=list)
    uncovered_criterion_ids: List[str] = Field(default_factory=list)
    unknown_criterion_ids: List[str] = Field(default_factory=list)
    execution_summary: dict[str, int]
    pass_rate: Optional[float] = None
    failure_classification: str
    failed_tests: List[str] = Field(default_factory=list)
    step_classifications: List[StepClassification] = Field(default_factory=list)
    root_cause: RootCauseAssessment
    defect_candidates: List[DefectCandidate] = Field(default_factory=list)
    risk_areas: List[str] = Field(default_factory=list)
    recommended_regression_scope: List[str] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    executive_summary: str = ""
    narrative_source: Literal["llm", "rule_based", "none"] = "none"

