# File: graph/workflow.py
# Description: Orchestrates the end-to-end QA analysis and execution workflow.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, cast

from agents.execution_agent.agent import ExecutionAgent
from agents.failure_analysis_agent.agent import FailureAnalysisAgent
from agents.reporting_agent.agent import QAReportAgent
from agents.requirement_agent.agent import RequirementAgent
from agents.root_cause_agent.agent import RootCauseAgent
from agents.test_design_agent.action_design import ActionDesignAgent
from agents.test_design_agent.agent import TestDesignAgent
from agents.test_planning_agent.agent import TestPlanningAgent
from config.settings import Settings
from config.settings import settings as default_settings
from core.llm.base import StructuredLLM
from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario
from core.models.test_plan import TestPlan
from graph.engine import END, LANGGRAPH_AVAILABLE, GraphSpec
from graph.nodes import SpecLoader, WorkflowNodes
from graph.state import GraphState, WorkflowState
from tools.api_tools.client import ApiTool
from tools.api_tools.openapi import OpenApiSpec, fetch_openapi
from tools.api_tools.runner import ApiScenarioRunner
from tools.browser_tools.playwright_tool import PlaywrightBrowserTool, playwright_available
from tools.browser_tools.runner import BrowserScenarioRunner
from tools.filesystem_tools.store import WorkspaceFileStore
from tools.rag_tools.knowledge_base import KnowledgeBase
from tools.reporting_tools.writer import QAReportWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN_FLOW = ("analyze_requirement", "retrieve_knowledge", "design_tests", "design_actions", "plan_tests")
ANALYSIS_FLOW = ("classify_failures", "analyze_root_cause", "generate_report")

Payload = Dict[str, Any]
Steps = List[Payload]


def load_knowledge(config: Settings, llm: StructuredLLM) -> KnowledgeBase | None:
    if not config.rag_enabled:
        return None
    directory = Path(config.knowledge_dir)
    if not directory.is_absolute() and not directory.is_dir():
        directory = PROJECT_ROOT / directory
    embedder: Any = llm if config.rag_use_embeddings and hasattr(llm, "embed") else None
    return KnowledgeBase.from_directory(directory, embedder=embedder, embedding_model=config.ollama_embedding_model)


class QAWorkflow:
    """Two graphs around a human approval gate (the LLM drives every analysis step).

    design graph:    requirement (LLM) -> RAG -> test design (LLM) -> action design (LLM tool calling)
                     -> plan (LLM prioritisation) -> [approved? execute : record_not_executed] -> analysis -> report
    execution graph: execute -> failure analysis (LLM) -> root cause (LLM) -> report (LLM) -> write artifacts

    The approval is persisted in SQLite (RunRepository), which is the checkpoint between the graphs.
    """

    def __init__(self, llm: StructuredLLM, config: Settings | None = None, spec_loader: SpecLoader | None = None,
                 knowledge: KnowledgeBase | None = None, use_fallback: bool | None = None):
        self.llm = llm
        self.config = config or default_settings
        self.use_fallback = use_fallback
        self.knowledge = knowledge if knowledge is not None else load_knowledge(self.config, llm)
        self._spec_loader = spec_loader or (self._load_target_spec if self.config.qa_api_base_url else None)
        self._nodes = self._build_nodes(ExecutionAgent(), None)
        self._design_graph = self._design_spec(self._nodes).compile()

    # ---- construction ----------------------------------------------------------------------
    def _build_nodes(self, execution_agent: ExecutionAgent, writer: QAReportWriter | None) -> WorkflowNodes:
        fallback = self.use_fallback
        return WorkflowNodes(
            requirement_agent=RequirementAgent(self.llm, use_fallback=fallback),
            test_design_agent=TestDesignAgent(self.llm, use_fallback=fallback),
            action_agent=ActionDesignAgent(self.llm, max_tool_steps=self.config.llm_max_tool_steps,
                                           use_fallback=fallback),
            test_planning_agent=TestPlanningAgent(self.llm, self.config.qa_api_base_url, self._browser_ready(),
                                                  use_fallback=fallback),
            execution_agent=execution_agent,
            failure_agent=FailureAnalysisAgent(self.llm, self.knowledge, use_fallback=fallback),
            root_cause_agent=RootCauseAgent(self.llm, self.knowledge, use_fallback=fallback),
            report_agent=QAReportAgent(self.llm, use_fallback=fallback),
            knowledge=self.knowledge,
            spec_loader=self._spec_loader,
            report_writer=writer,
            rag_top_k=self.config.rag_top_k,
        )

    @staticmethod
    def _design_spec(nodes: WorkflowNodes) -> GraphSpec:
        spec = GraphSpec(GraphState)
        for name in (*DESIGN_FLOW, "record_not_executed", "execute", *ANALYSIS_FLOW):
            spec.add_node(name, getattr(nodes, name))
        spec.set_entry(DESIGN_FLOW[0])
        for current, following in zip(DESIGN_FLOW, DESIGN_FLOW[1:]):
            spec.add_edge(current, following)
        spec.add_conditional_edges("plan_tests", nodes.route_after_plan,
                                   {"execute": "execute", "await_approval": "record_not_executed"})
        spec.add_edge("record_not_executed", ANALYSIS_FLOW[0])
        spec.add_edge("execute", ANALYSIS_FLOW[0])
        for current, following in zip(ANALYSIS_FLOW, ANALYSIS_FLOW[1:]):
            spec.add_edge(current, following)
        spec.add_edge(ANALYSIS_FLOW[-1], END)
        return spec

    @staticmethod
    def _execution_spec(nodes: WorkflowNodes) -> GraphSpec:
        flow = ("execute", *ANALYSIS_FLOW, "write_artifacts")
        spec = GraphSpec(GraphState)
        for name in flow:
            spec.add_node(name, getattr(nodes, name))
        spec.set_entry(flow[0])
        for current, following in zip(flow, flow[1:]):
            spec.add_edge(current, following)
        spec.add_edge(flow[-1], END)
        return spec

    def describe(self) -> Payload:
        return {
            "engine": "langgraph" if LANGGRAPH_AVAILABLE else "local",
            "design_graph": self._design_spec(self._nodes).describe(),
            "execution_graph": self._execution_spec(self._nodes).describe(),
        }

    def _browser_ready(self) -> bool:
        return self.config.browser_enabled and playwright_available()

    def _load_target_spec(self) -> OpenApiSpec:
        tool = ApiTool(self.config.allowed_hosts_for(self.config.qa_api_base_url), self.config.qa_allow_private_targets)
        try:
            target = tool.validate_target_url(self.config.qa_api_base_url)
            return fetch_openapi(tool, target, self.config.qa_api_timeout_seconds)
        finally:
            tool.close()

    # ---- phase 1: analysis and design --------------------------------------------------------
    def run(self, requirement: str) -> WorkflowState:
        normalized = (requirement or "").strip()
        if not normalized:
            raise ValueError("Requirement text cannot be empty.")
        final = cast(GraphState, self._design_graph({"requirement": normalized}))
        # FIX: GraphState is total=False, so final["analysis"] etc. were flagged as unsafe.
        # Read every key with .get() and narrow it with an explicit None check (replaces the assert).
        analysis, domain, plan = final.get("analysis"), final.get("domain"), final.get("plan")
        scenarios, execution = final.get("scenarios"), final.get("execution")
        failure, root_cause, report = final.get("failure"), final.get("root_cause"), final.get("report")
        if (analysis is None or domain is None or plan is None or scenarios is None or execution is None
                or failure is None or root_cause is None or report is None):
            raise RuntimeError("Workflow ended without producing its complete result state.")
        return WorkflowState(
            requirement=normalized, title=analysis.title, actors=analysis.actors, goal=analysis.goal,
            domain=domain.domain, domain_mode=final.get("analysis_mode", "llm"),
            business_object=domain.business_object, key_concerns=domain.key_concerns,
            knowledge_sources=final.get("knowledge_sources", []),
            generation_mode=final.get("generation_mode", "llm"),
            generated_scenarios=scenarios,
            action_design=WorkflowNodes.summarize_action_report(final.get("action_report")),
            failure_classification=failure.classification, risk_level=failure.risk_level,
            failure_evidence=failure.evidence, status="completed",
            execution_status=report.execution_status, execution_summary=execution.summary(),
            business_rules=analysis.business_rules, acceptance_criteria=analysis.acceptance_criteria,
            criteria=analysis.criteria, ambiguities=analysis.ambiguities,
            missing_requirements=analysis.missing_requirements,
            testable_conditions=analysis.testable_conditions, security_flags=analysis.security_flags,
            test_plan=plan, root_cause=root_cause, qa_report=report,
        )

    # ---- phase 2: execute an approved, persisted plan -----------------------------------------
    def execute_run(self, run: Payload) -> Payload:
        plan = TestPlan.model_validate(run["test_plan"])
        if not plan.is_target_configured:
            raise ValueError("Approved plan does not include a configured target environment.")
        scenarios = [QAScenario.model_validate(item) for item in run["generated_scenarios"]]
        criteria = [AcceptanceCriterion.model_validate(item) for item in run.get("criteria", [])]
        run_prefix = f"runs/{run['run_id']}"
        store = WorkspaceFileStore(self.config.qa_artifacts_dir)  # now defined in Settings
        hosts = self.config.allowed_hosts_for(plan.target_environment)
        eligible = [item for item in plan.scenarios if item.eligibility == "eligible"]
        api_runner = None
        if any(item.runner == "api" for item in eligible):
            api_runner = ApiScenarioRunner(plan.target_environment, hosts, self.config.qa_allow_private_targets,
                                           timeout_seconds=self.config.qa_api_timeout_seconds)
        try:
            browser_runner = None
            if any(item.runner == "browser" for item in eligible) and self._browser_ready():
                tool = ApiTool(hosts, self.config.qa_allow_private_targets)
                try:
                    target = tool.validate_target_url(plan.target_environment)
                finally:
                    tool.close()
                browser_runner = BrowserScenarioRunner(
                    target,
                    PlaywrightBrowserTool(self.config.browser_headless, self.config.browser_timeout_ms,
                                          self.config.playwright_chromium_executable),
                    store, f"{run_prefix}/browser")
            nodes = self._build_nodes(ExecutionAgent(api_runner, browser_runner), QAReportWriter(store))
            final = self._execution_spec(nodes).compile()({
                "plan": plan, "scenarios": scenarios, "criteria": criteria,
                "run_prefix": run_prefix, "domain": None,
            })
        finally:
            if api_runner is not None:
                api_runner.close()
        execution, failure, report = final["execution"], final["failure"], final["report"]
        return {
            "execution_summary": execution.summary(),
            "execution_status": report.execution_status,
            "execution_steps": _steps(execution.steps),
            "failure_classification": failure.classification,
            "failure_evidence": failure.evidence,
            "risk_level": failure.risk_level,
            "root_cause": final["root_cause"],
            "qa_report": report,
        }


def _steps(steps: List[Any]) -> Steps:
    return [{
        "scenario": s.scenario, "scenario_id": s.scenario_id, "status": s.status,
        "execution_type": s.execution_type, "details": s.details, "evidence": s.evidence,
        "error_type": s.error_type, "http_statuses": s.http_statuses, "artifacts": s.artifacts,
        "duration_ms": s.duration_ms,
    } for s in steps]