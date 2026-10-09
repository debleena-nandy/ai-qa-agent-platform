# File: graph/nodes.py
# Description: Maps QA workflow stages to callable graph nodes.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List

from agents.execution_agent.agent import ExecutionAgent
from agents.failure_analysis_agent.agent import FailureAnalysisAgent
from agents.reporting_agent.agent import QAReportAgent
from agents.requirement_agent.agent import RequirementAgent
from agents.root_cause_agent.agent import RootCauseAgent
from agents.test_design_agent.action_design import ActionDesignAgent, ActionDesignReport
from agents.test_design_agent.agent import TestDesignAgent
from agents.test_planning_agent.agent import TestPlanningAgent
from core.models.execution_result import ExecutionResult, ExecutionStepResult, derive_execution_status
from graph.state import GraphState
from tools.api_tools.openapi import OpenApiSpec
from tools.rag_tools.knowledge_base import KnowledgeBase, checklist_concerns
from tools.reporting_tools.writer import QAReportWriter

logger = logging.getLogger(__name__)
SpecLoader = Callable[[], OpenApiSpec]
Values = List[Any]
Summary = Dict[str, Any]


def _require(state: GraphState, *keys: str) -> Values:
    missing = [key for key in keys if state.get(key) is None]
    if missing:
        raise RuntimeError(f"Workflow state is missing: {', '.join(missing)}.")
    return [state.get(key) for key in keys]


class WorkflowNodes:
    """Node handlers shared by the design graph and the execution graph."""

    def __init__(
        self,
        *,
        requirement_agent: RequirementAgent,
        test_design_agent: TestDesignAgent,
        action_agent: ActionDesignAgent,
        test_planning_agent: TestPlanningAgent,
        execution_agent: ExecutionAgent,
        failure_agent: FailureAnalysisAgent,
        root_cause_agent: RootCauseAgent,
        report_agent: QAReportAgent,
        knowledge: KnowledgeBase | None = None,
        spec_loader: SpecLoader | None = None,
        report_writer: QAReportWriter | None = None,
        rag_top_k: int = 3,
    ):
        self.requirement_agent = requirement_agent
        self.test_design_agent = test_design_agent
        self.action_agent = action_agent
        self.test_planning_agent = test_planning_agent
        self.execution_agent = execution_agent
        self.failure_agent = failure_agent
        self.root_cause_agent = root_cause_agent
        self.report_agent = report_agent
        self.knowledge = knowledge
        self.spec_loader = spec_loader
        self.report_writer = report_writer
        self.rag_top_k = rag_top_k

    # ---- design phase (LLM) ------------------------------------------------------------
    def analyze_requirement(self, state: GraphState) -> GraphState:
        (requirement,) = _require(state, "requirement")
        analysis, domain, mode = self.requirement_agent.analyze(requirement)
        return {"analysis": analysis, "domain": domain, "analysis_mode": mode, "criteria": analysis.criteria}

    def retrieve_knowledge(self, state: GraphState) -> GraphState:
        """RAG: retrieve QA checklists and merge their concerns into the domain profile."""
        domain, requirement = _require(state, "domain", "requirement")
        if self.knowledge is None or not self.knowledge.chunks:
            return {"knowledge": [], "knowledge_sources": []}
        hits = self.knowledge.search(f"{domain.domain} {domain.business_object} {requirement}",
                                     top_k=self.rag_top_k, kind="checklist")
        concerns = list(domain.key_concerns)
        for hit in hits:
            for concern in checklist_concerns(hit.chunk):
                if concern not in concerns:
                    concerns.append(concern)
        return {
            "domain": domain.model_copy(update={"key_concerns": concerns[:20]}),
            "knowledge": [f"{hit.chunk.title}: {hit.chunk.text}" for hit in hits],
            "knowledge_sources": [f"{hit.chunk.id} ({hit.score})" for hit in hits],
        }

    def design_tests(self, state: GraphState) -> GraphState:
        analysis, domain, requirement = _require(state, "analysis", "domain", "requirement")
        scenarios, mode = self.test_design_agent.generate_scenarios(analysis, domain, requirement,
                                                                    state.get("knowledge") or [])
        return {"scenarios": scenarios, "generation_mode": mode}

    def design_actions(self, state: GraphState) -> GraphState:
        """Tool calling: ground scenarios in the target's OpenAPI spec and attach validated actions."""
        (scenarios,) = _require(state, "scenarios")
        unresolved = [s.id for s in scenarios]
        if self.spec_loader is None:
            return {"action_report": ActionDesignReport(unresolved=unresolved,
                                                         spec_error="No target configured; actions not designed.")}
        try:
            spec = self.spec_loader()
        except Exception as exc:  # unreachable target, missing OpenAPI document, policy violation
            logger.warning("Could not load the target OpenAPI spec: %s", exc)
            return {"action_report": ActionDesignReport(unresolved=unresolved, spec_error=str(exc)[:300])}
        designed, report = self.action_agent.design(scenarios, spec)
        return {"scenarios": designed, "action_report": report}

    def plan_tests(self, state: GraphState) -> GraphState:
        (scenarios,) = _require(state, "scenarios")
        return {"plan": self.test_planning_agent.create_plan(scenarios)}

    @staticmethod
    def route_after_plan(state: GraphState) -> str:
        plan = state.get("plan")
        return "execute" if plan is not None and plan.approval_status == "approved" else "await_approval"

    def record_not_executed(self, state: GraphState) -> GraphState:
        plan, scenarios = _require(state, "plan", "scenarios")
        items = {item.scenario_id: item for item in plan.scenarios}
        execution = ExecutionResult()
        for scenario in scenarios:
            item = items.get(scenario.id)
            details = "; ".join(item.blockers) if item and item.blockers else "Awaiting plan approval."
            execution.add_step(ExecutionStepResult(
                scenario=f"{scenario.id} {scenario.title}", scenario_id=scenario.id, scenario_type=scenario.type,
                status="not_executed", execution_type=item.runner if item else "api", details=details))
        return {"execution": execution}

    # ---- execution and analysis phase ------------------------------------------------------
    def execute(self, state: GraphState) -> GraphState:
        plan, scenarios = _require(state, "plan", "scenarios")
        return {"execution": self.execution_agent.execute(plan, scenarios)}

    def classify_failures(self, state: GraphState) -> GraphState:
        (execution,) = _require(state, "execution")
        return {"failure": self.failure_agent.analyze(execution)}

    def analyze_root_cause(self, state: GraphState) -> GraphState:
        execution, failure = _require(state, "execution", "failure")
        return {"root_cause": self.root_cause_agent.analyze(execution, failure)}

    def generate_report(self, state: GraphState) -> GraphState:
        plan, criteria, scenarios, execution, failure, root_cause = _require(
            state, "plan", "criteria", "scenarios", "execution", "failure", "root_cause")
        report = self.report_agent.generate(
            plan=plan, criteria=criteria, scenarios=scenarios, execution=execution, failure=failure,
            root_cause=root_cause, workflow_status="completed",
            execution_status=derive_execution_status(plan.approval_status == "approved", execution))
        return {"report": report}

    def write_artifacts(self, state: GraphState) -> GraphState:
        report, prefix = state.get("report"), state.get("run_prefix")
        if self.report_writer is None or report is None or not prefix:
            return {"artifact_paths": []}
        paths = self.report_writer.write_all(report, prefix, {"run_id": prefix.rsplit("/", 1)[-1]})
        relative = [self.report_writer.file_store.relative(path) for path in paths]
        merged = sorted(set(report.artifacts) | set(relative))
        return {"report": report.model_copy(update={"artifacts": merged}), "artifact_paths": relative}

    @staticmethod
    def summarize_action_report(report: ActionDesignReport | None) -> Summary:
        return asdict(report) if report is not None else {}