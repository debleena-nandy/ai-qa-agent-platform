# File: api/main.py
# Description: Defines the FastAPI application and QA workflow endpoints.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Literal, Optional, TypeVar

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.failure_analysis_agent.agent import FailureAnalysisAgent
from agents.reporting_agent.agent import QAReportAgent
from agents.requirement_agent.agent import RequirementAgent
from agents.root_cause_agent.agent import RootCauseAgent
from agents.test_planning_agent.agent import TestPlanningAgent
from config.settings import settings
from core.llm.agent_support import AgentLLMError, ensure_llm_ready
from core.llm.base import LLMResponseError, LLMUnavailableError, StructuredLLM, check_llm_health, close_llm
from core.llm.ollama_client import build_llm_client
from core.models.actions import ApiRequestAction, BrowserAction
from core.models.execution_result import ExecutionResult, ExecutionStepResult, derive_execution_status
from core.models.qa_report import QAReport, RootCauseAssessment
from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario
from core.models.test_plan import UNCONFIGURED, TestPlan
from core.persistence.run_repository import RunNotFoundError, RunRepository, RunStateError
from graph.workflow import QAWorkflow
from tools.browser_tools.playwright_tool import playwright_available
from tools.reporting_tools.writer import render_markdown

logger = logging.getLogger(__name__)

Mode = Literal["llm", "rule_based", "manual"]
Payload = Dict[str, Any]
Summaries = List[Payload]
Lifespan = AsyncIterator[None]
LLMIterator = Iterator[StructuredLLM]
R = TypeVar("R")
LLM_FAILURES = (AgentLLMError, LLMUnavailableError, LLMResponseError)


@asynccontextmanager
async def lifespan(application: FastAPI) -> Lifespan:
    """The LLM is mandatory: refuse to start when the model is not reachable (LLM_CHECK_ON_STARTUP)."""
    if settings.llm_check_on_startup:
        client = build_llm_client()
        try:
            application.state.llm_health = ensure_llm_ready(client)
        finally:
            close_llm(client)
    yield


app = FastAPI(title="AI QA Agent Platform", version="3.0.0", lifespan=lifespan)


# ---- request / response models ------------------------------------------------------------------
class RequirementRequest(BaseModel):
    requirement: str = Field(..., description="User story or requirement text")

    @field_validator("requirement")
    @classmethod
    def clean_requirement(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 10:
            raise ValueError("Requirement text must be at least 10 characters after trimming.")
        if len(normalized) > settings.max_requirement_chars:
            raise ValueError(f"Requirement text must be at most {settings.max_requirement_chars} characters.")
        return normalized


class PlanDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]


class ScenarioEditRequest(BaseModel):
    action: Literal["exclude", "include", "set_actions"]
    reason: Optional[str] = Field(default=None, max_length=500)
    api_actions: Optional[List[ApiRequestAction]] = None
    browser_actions: Optional[List[BrowserAction]] = None


class BulkExcludeRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class StructuredRunRequest(RequirementRequest):
    scenarios: List[QAScenario] = Field(min_length=1, max_length=100)


class ExecutionSummary(BaseModel):
    total_scenarios: int
    passed: int
    failed: int
    skipped: int
    not_executed: int


class RunResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    run_status: str
    title: str
    actors: List[str]
    goal: str
    domain: str
    domain_mode: Mode
    business_object: str
    key_concerns: List[str]
    knowledge_sources: List[str] = Field(default_factory=list)
    status: str
    generation_mode: Mode
    scenario_count: int
    generated_scenarios: List[QAScenario]
    action_design: Dict[str, Any] = Field(default_factory=dict)
    failure_classification: str
    failure_evidence: List[str]
    risk_level: str
    execution_summary: ExecutionSummary
    execution_steps: List[Dict[str, Any]] = Field(default_factory=list)
    execution_error: Optional[str] = None
    business_rules: List[str]
    acceptance_criteria: List[str]
    ambiguities: List[str]
    missing_requirements: List[str] = Field(default_factory=list)
    testable_conditions: List[str] = Field(default_factory=list)
    security_flags: List[str] = Field(default_factory=list)
    criteria: List[AcceptanceCriterion]
    execution_status: Literal["not_started", "partial", "completed"]
    test_plan: TestPlan
    root_cause: RootCauseAssessment
    qa_report: QAReport


class RunSummary(BaseModel):
    run_id: str
    plan_id: str
    run_status: str
    title: str
    execution_status: str
    created_at: str


# ---- dependencies ------------------------------------------------------------------------------
def get_llm() -> LLMIterator:
    client = build_llm_client()
    try:
        yield client
    finally:
        close_llm(client)


def get_run_repository() -> RunRepository:
    return RunRepository(settings.qa_database_path, max_runs=settings.max_stored_runs)


def require_approval_token(token: str | None = Header(default=None, alias="X-QA-Approval-Token")) -> None:
    configured = settings.qa_approval_token
    if not configured:
        raise HTTPException(status_code=503, detail="Plan and run endpoints are disabled until QA_APPROVAL_TOKEN is set.")
    if token is None or not hmac.compare_digest(token, configured):
        raise HTTPException(status_code=403, detail="A valid X-QA-Approval-Token is required.")


def analyze_guard(token: str | None = Header(default=None, alias="X-QA-Approval-Token")) -> None:
    if settings.analyze_requires_token:
        require_approval_token(token)


def _planner(plan: TestPlan) -> TestPlanningAgent:
    target = plan.target_environment if plan.is_target_configured else ""
    return TestPlanningAgent(None, target, settings.browser_enabled and playwright_available())


def _plan_errors(call: Callable[[], R]) -> R:
    try:
        return call()
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunStateError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


# ---- service ------------------------------------------------------------------------------------
@app.get("/")
def root() -> Payload:
    return {"service": settings.app_name, "environment": settings.app_env, "status": "running"}


@app.get("/health")
def health_check() -> Payload:
    client = build_llm_client()
    try:
        llm = check_llm_health(client, timeout=settings.llm_healthcheck_timeout_seconds) or {}
    finally:
        close_llm(client)
    ready = bool(llm.get("reachable") and llm.get("model_available"))
    return {
        "status": "ok" if ready else "degraded", "service": settings.app_name, "environment": settings.app_env,
        "llm": llm, "rule_based_fallback": settings.rule_based_fallback,
        "browser_enabled": settings.browser_enabled, "target_configured": bool(settings.qa_api_base_url),
    }


@app.get("/workflow")
def describe_workflow(llm: StructuredLLM = Depends(get_llm)) -> Payload:
    return QAWorkflow(llm).describe()


# ---- phase 1: analysis and design ------------------------------------------------------------------
@app.post("/analyze", response_model=RunResponse)
async def analyze_requirement(
    payload: RequirementRequest,
    _guard: None = Depends(analyze_guard),
    llm: StructuredLLM = Depends(get_llm),
    repository: RunRepository = Depends(get_run_repository),
) -> Payload:
    workflow = QAWorkflow(llm)
    try:
        state = await run_in_threadpool(workflow.run, payload.requirement)
    except LLM_FAILURES as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    body = asdict(state)
    body.pop("requirement")
    body["generated_scenarios"] = state.generated_scenarios
    body["scenario_count"] = len(state.generated_scenarios)
    return repository.create_run(jsonable_encoder(body))


@app.post("/runs", response_model=RunResponse)
async def create_structured_run(
    payload: StructuredRunRequest,
    llm: StructuredLLM = Depends(get_llm),
    repository: RunRepository = Depends(get_run_repository),
    _authorized: None = Depends(require_approval_token),
) -> Payload:
    try:
        analysis, domain, mode = await run_in_threadpool(RequirementAgent(llm).analyze, payload.requirement)
    except LLM_FAILURES as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    criteria = analysis.criteria
    ids = [s.id for s in payload.scenarios]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="Scenario ids must be unique.")
    unknown = sorted({cid for s in payload.scenarios for cid in s.criterion_ids} - {c.id for c in criteria})
    if unknown:
        raise HTTPException(status_code=422, detail=f"Scenarios reference unknown acceptance criteria: {', '.join(unknown)}.")
    scenarios = [s.model_copy(update={"action_source": "manual"}) if (s.api_actions or s.browser_actions) else s
                 for s in payload.scenarios]
    planner = TestPlanningAgent(None, settings.qa_api_base_url, settings.browser_enabled and playwright_available())
    plan = planner.create_plan(scenarios)
    items = {item.scenario_id: item for item in plan.scenarios}
    execution = ExecutionResult()
    for scenario in scenarios:
        item = items[scenario.id]
        execution.add_step(ExecutionStepResult(
            scenario=f"{scenario.id} {scenario.title}", scenario_id=scenario.id, scenario_type=scenario.type,
            status="not_executed", execution_type=item.runner, details="; ".join(item.blockers) or "Awaiting approval."))
    failure = FailureAnalysisAgent(llm).analyze(execution)
    root_cause = RootCauseAgent(llm).analyze(execution, failure)
    report = QAReportAgent(None).generate(plan, criteria, scenarios, execution, failure, root_cause, "completed",
                                          derive_execution_status(False, execution))
    body = {
        "title": analysis.title, "actors": analysis.actors, "goal": analysis.goal, "domain": domain.domain,
        "domain_mode": mode, "business_object": domain.business_object, "key_concerns": domain.key_concerns,
        "status": "completed", "generation_mode": "manual", "scenario_count": len(scenarios),
        "generated_scenarios": scenarios, "failure_classification": failure.classification,
        "failure_evidence": failure.evidence, "risk_level": failure.risk_level,
        "execution_summary": execution.summary(), "business_rules": analysis.business_rules,
        "acceptance_criteria": analysis.acceptance_criteria, "ambiguities": analysis.ambiguities,
        "missing_requirements": analysis.missing_requirements, "testable_conditions": analysis.testable_conditions,
        "security_flags": analysis.security_flags, "criteria": criteria,
        "execution_status": report.execution_status, "test_plan": plan, "root_cause": root_cause, "qa_report": report,
    }
    return repository.create_run(jsonable_encoder(body))


@app.get("/runs", response_model=List[RunSummary])
def list_runs(limit: int = 50, repository: RunRepository = Depends(get_run_repository),
              _authorized: None = Depends(require_approval_token)) -> Summaries:
    return repository.list_runs(min(max(limit, 1), 200))


@app.get("/runs/{run_id}", response_model=RunResponse)
def get_run(run_id: str, repository: RunRepository = Depends(get_run_repository),
            _authorized: None = Depends(require_approval_token)) -> Payload:
    return _plan_errors(lambda: repository.get_run(run_id))


@app.get("/runs/{run_id}/report.md", response_class=PlainTextResponse)
def get_run_report(run_id: str, repository: RunRepository = Depends(get_run_repository),
                   _authorized: None = Depends(require_approval_token)) -> str:
    run = _plan_errors(lambda: repository.get_run(run_id))
    return render_markdown(QAReport.model_validate(run["qa_report"]), run)


# ---- human review of the plan -----------------------------------------------------------------------
@app.get("/plans/{plan_id}", response_model=TestPlan)
def get_plan(plan_id: str, repository: RunRepository = Depends(get_run_repository),
             _authorized: None = Depends(require_approval_token)) -> TestPlan:
    return _plan_errors(lambda: repository.get_plan(plan_id))


@app.patch("/plans/{plan_id}/scenarios/{scenario_id}", response_model=TestPlan)
def edit_plan_scenario(plan_id: str, scenario_id: str, payload: ScenarioEditRequest,
                       repository: RunRepository = Depends(get_run_repository),
                       _authorized: None = Depends(require_approval_token)) -> TestPlan:
    def change(plan: TestPlan, run: Payload) -> TestPlan:
        index = next((i for i, item in enumerate(plan.scenarios) if item.scenario_id == scenario_id), None)
        if index is None:
            raise RunNotFoundError(f"Scenario '{scenario_id}' is not part of plan '{plan_id}'.")
        item = plan.scenarios[index]
        if payload.action == "exclude":
            item = item.model_copy(update={"eligibility": "excluded",
                                           "exclusion_reason": payload.reason or "Excluded by reviewer."})
        else:
            stored = run["generated_scenarios"]
            position = next(i for i, s in enumerate(stored) if s["id"] == scenario_id)
            scenario = QAScenario.model_validate(stored[position])
            if payload.action == "set_actions":
                if not payload.api_actions and not payload.browser_actions:
                    raise RunStateError("set_actions requires api_actions or browser_actions.")
                scenario = scenario.model_copy(update={
                    "api_actions": payload.api_actions or [], "browser_actions": payload.browser_actions or [],
                    "action_source": "manual"})
                stored[position] = jsonable_encoder(scenario)
            item = _planner(plan).plan_item(scenario).model_copy(update={"risk_note": item.risk_note})
        plan.scenarios[index] = item
        return plan

    return _plan_errors(lambda: repository.update_plan(plan_id, change))


@app.post("/plans/{plan_id}/exclude-blocked", response_model=TestPlan)
def exclude_blocked(plan_id: str, payload: BulkExcludeRequest,
                    repository: RunRepository = Depends(get_run_repository),
                    _authorized: None = Depends(require_approval_token)) -> TestPlan:
    def change(plan: TestPlan, _run: Payload) -> TestPlan:
        plan.scenarios = [
            item.model_copy(update={"eligibility": "excluded",
                                    "exclusion_reason": f"{payload.reason} ({'; '.join(item.blockers)})"})
            if item.eligibility == "blocked" else item
            for item in plan.scenarios
        ]
        return plan

    return _plan_errors(lambda: repository.update_plan(plan_id, change))


@app.post("/plans/{plan_id}/approval", response_model=TestPlan)
def decide_plan(plan_id: str, payload: PlanDecisionRequest,
                repository: RunRepository = Depends(get_run_repository),
                _authorized: None = Depends(require_approval_token)) -> TestPlan:
    return _plan_errors(lambda: repository.decide_plan(plan_id, payload.decision))


# ---- phase 2: execution of an approved plan --------------------------------------------------------------
@app.post("/runs/{run_id}/execute", response_model=RunResponse)
async def execute_run(run_id: str, llm: StructuredLLM = Depends(get_llm),
                      repository: RunRepository = Depends(get_run_repository),
                      _authorized: None = Depends(require_approval_token)) -> Payload:
    run = _plan_errors(lambda: repository.claim_execution(run_id))
    if TestPlan.model_validate(run["test_plan"]).target_environment == UNCONFIGURED:
        repository.fail_run(run_id, "Approved plan does not include a configured target environment.")
        raise HTTPException(status_code=409, detail="Approved plan does not include a configured target environment.")
    try:
        updates = await run_in_threadpool(QAWorkflow(llm).execute_run, run)
    except LLM_FAILURES as exc:
        repository.fail_run(run_id, f"LLM step failed during analysis: {str(exc)[:300]}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        repository.fail_run(run_id, str(exc))
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        repository.fail_run(run_id, "Execution failed; inspect server logs.")
        logger.exception("Execution failed for run %s", run_id)
        raise
    return repository.complete_run(run_id, jsonable_encoder(updates))