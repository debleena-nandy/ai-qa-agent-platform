from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TypedDict, cast

from app.agents.domain_agent import DomainAgent
from app.agents.failure_analysis_agent import FailureAnalysisAgent, FailureReport
from app.agents.requirement_agent import RequirementAgent, RequirementAnalysis
from app.agents.test_design_agent import TestDesignAgent
from app.execution.executor import QAExecutor
from app.llm.base import StructuredLLM, close_llm
from app.llm.ollama_client import build_llm_client
from app.models.execution_result import ExecutionResult
from app.models.scenario import DomainProfile, QAScenario

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    LANGGRAPH_AVAILABLE = False


class GraphState(TypedDict, total=False):
    requirement: str
    analysis: RequirementAnalysis
    domain: DomainProfile
    domain_mode: str
    scenarios: List[QAScenario]
    generation_mode: str
    execution: ExecutionResult
    failure: FailureReport


class CompletedGraphState(TypedDict):
    requirement: str
    analysis: RequirementAnalysis
    domain: DomainProfile
    domain_mode: str
    scenarios: List[QAScenario]
    generation_mode: str
    execution: ExecutionResult
    failure: FailureReport


@dataclass
class WorkflowState:
    requirement: str = ""
    title: str = ""
    actors: List[str] = field(default_factory=list)
    goal: str = ""
    domain: str = ""
    domain_mode: str = "rule_based"
    business_object: str = ""
    key_concerns: List[str] = field(default_factory=list)
    generation_mode: str = "rule_based"
    generated_scenarios: List[QAScenario] = field(default_factory=list)
    failure_classification: str = ""
    risk_level: str = "medium"
    failure_evidence: List[str] = field(default_factory=list)
    status: str = "pending"
    execution_summary: Dict[str, int] = field(default_factory=dict)
    business_rules: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)


class QAWorkflow:
    """Orchestrates the end-to-end QA workflow for a requirement."""

    NODE_ORDER = ["analyze_requirement", "extract_domain", "design_tests", "execute", "classify_failures"]

    def __init__(self, llm_client: Optional[StructuredLLM] = None, use_settings_llm: bool = True):
        self.llm: Optional[StructuredLLM] = (
            llm_client if llm_client is not None else (build_llm_client() if use_settings_llm else None)
        )
        self.requirement_agent = RequirementAgent()
        self.domain_agent = DomainAgent(self.llm)
        self.test_design_agent = TestDesignAgent(self.llm)
        self.executor = QAExecutor()
        self.failure_agent = FailureAnalysisAgent()
        self._nodes: Dict[str, Callable[[GraphState], GraphState]] = {
            "analyze_requirement": self._analyze_requirement,
            "extract_domain": self._extract_domain,
            "design_tests": self._design_tests,
            "execute": self._execute,
            "classify_failures": self._classify_failures,
        }
        self._graph = self._build_graph() if LANGGRAPH_AVAILABLE else None

    def _analyze_requirement(self, state: GraphState) -> GraphState:
        requirement = state.get("requirement")
        if requirement is None:
            raise RuntimeError("Workflow state is missing the requirement.")
        return {"analysis": self.requirement_agent.analyze(requirement)}

    def _extract_domain(self, state: GraphState) -> GraphState:
        analysis = state.get("analysis")
        requirement = state.get("requirement")
        if analysis is None or requirement is None:
            raise RuntimeError("Workflow state is missing requirement analysis inputs.")
        domain, mode = self.domain_agent.extract(analysis, requirement)
        return {"domain": domain, "domain_mode": mode}

    def _design_tests(self, state: GraphState) -> GraphState:
        analysis = state.get("analysis")
        domain = state.get("domain")
        requirement = state.get("requirement")
        if analysis is None or domain is None or requirement is None:
            raise RuntimeError("Workflow state is missing test-design inputs.")
        scenarios, mode = self.test_design_agent.generate_scenarios(
            analysis,
            domain,
            requirement,
            # Skip only when the server was unreachable/timed out; a single bad JSON answer is not a reason.
            use_llm=not self.domain_agent.llm_unavailable,
        )
        return {"scenarios": scenarios, "generation_mode": mode}

    def _execute(self, state: GraphState) -> GraphState:
        scenarios = state.get("scenarios")
        if scenarios is None:
            raise RuntimeError("Workflow state is missing scenarios to execute.")
        return {"execution": self.executor.run_scenarios(scenarios)}

    def _classify_failures(self, state: GraphState) -> GraphState:
        execution = state.get("execution")
        if execution is None:
            raise RuntimeError("Workflow state is missing execution results.")
        return {"failure": self.failure_agent.analyze(execution)}

    def close(self) -> None:
        # Releases the LLM client's HTTP session; called by the API dependency after each request.
        close_llm(self.llm)

    def __enter__(self) -> QAWorkflow:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _build_graph(self):
        graph = StateGraph(GraphState)
        for name in self.NODE_ORDER:
            graph.add_node(name, self._nodes[name])
        graph.add_edge(START, self.NODE_ORDER[0])
        for current, nxt in zip(self.NODE_ORDER, self.NODE_ORDER[1:]):
            graph.add_edge(current, nxt)
        graph.add_edge(self.NODE_ORDER[-1], END)
        return graph.compile()

    def _invoke(self, initial: GraphState) -> GraphState:
        if self._graph is not None:
            return cast(GraphState, self._graph.invoke(initial))
        state = cast(GraphState, dict(initial))
        for name in self.NODE_ORDER:
            state.update(self._nodes[name](state))
        return state

    def run(self, requirement: str) -> WorkflowState:
        normalized_requirement = (requirement or "").strip()
        if not normalized_requirement:
            raise ValueError("Requirement text cannot be empty.")

        final = self._invoke({"requirement": normalized_requirement})
        required_keys = (
            "requirement", "analysis", "domain", "execution", "failure",
            "domain_mode", "generation_mode", "scenarios",
        )
        if any(key not in final for key in required_keys):
            raise RuntimeError("Workflow ended without producing its complete result state.")
        completed = cast(CompletedGraphState, final)
        analysis = completed["analysis"]
        domain = completed["domain"]
        execution = completed["execution"]
        failure = completed["failure"]

        return WorkflowState(
            requirement=normalized_requirement,
            title=analysis.title,
            actors=analysis.actors,
            goal=analysis.goal,
            domain=domain.domain,
            domain_mode=completed["domain_mode"],
            business_object=domain.business_object,
            key_concerns=domain.key_concerns,
            generation_mode=completed["generation_mode"],
            generated_scenarios=completed["scenarios"],
            failure_classification=failure.classification,
            risk_level=failure.risk_level,
            failure_evidence=failure.evidence,
            status="completed",
            execution_summary=execution.summary(),
            business_rules=analysis.business_rules,
            acceptance_criteria=analysis.acceptance_criteria,
            ambiguities=analysis.ambiguities,
        )
