# File: graph/state.py
# Description: Defines typed and runtime state used throughout the QA workflow.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from agents.failure_analysis_agent.agent import FailureReport
from agents.requirement_agent.models import RequirementAnalysis
from agents.test_design_agent.action_design import ActionDesignReport
from core.models.execution_result import ExecutionResult
from core.models.qa_report import QAReport, RootCauseAssessment
from core.models.requirement import AcceptanceCriterion
from core.models.scenario import DomainProfile, QAScenario
from core.models.test_plan import TestPlan


class GraphState(TypedDict, total=False):
    requirement: str
    analysis: RequirementAnalysis
    domain: Optional[DomainProfile]
    analysis_mode: str
    knowledge: List[str]
    knowledge_sources: List[str]
    scenarios: List[QAScenario]
    generation_mode: str
    action_report: ActionDesignReport
    plan: TestPlan
    criteria: List[AcceptanceCriterion]
    execution: ExecutionResult
    failure: FailureReport
    root_cause: RootCauseAssessment
    report: QAReport
    run_prefix: str
    artifact_paths: List[str]


@dataclass
class WorkflowState:
    requirement: str = ""
    title: str = ""
    actors: List[str] = field(default_factory=list)
    goal: str = ""
    domain: str = ""
    domain_mode: str = "llm"
    business_object: str = ""
    key_concerns: List[str] = field(default_factory=list)
    knowledge_sources: List[str] = field(default_factory=list)
    generation_mode: str = "llm"
    generated_scenarios: List[QAScenario] = field(default_factory=list)
    action_design: Dict[str, Any] = field(default_factory=dict)
    failure_classification: str = ""
    risk_level: str = "medium"
    failure_evidence: List[str] = field(default_factory=list)
    status: str = "pending"
    execution_status: str = "not_started"
    execution_summary: Dict[str, int] = field(default_factory=dict)
    business_rules: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    criteria: List[AcceptanceCriterion] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    testable_conditions: List[str] = field(default_factory=list)
    security_flags: List[str] = field(default_factory=list)
    test_plan: TestPlan = field(default_factory=TestPlan)
    root_cause: Optional[RootCauseAssessment] = None
    qa_report: Optional[QAReport] = None