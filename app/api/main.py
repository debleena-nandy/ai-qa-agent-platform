from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterator, List, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config.settings import settings
from app.graph.workflow import QAWorkflow
from app.llm.base import check_llm_health, close_llm
from app.llm.ollama_client import build_llm_client
from app.models.scenario import QAScenario

app = FastAPI(title="AI QA Agent Platform", version="1.3.0")

Mode = Literal["llm", "rule_based"]


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


class ExecutionSummary(BaseModel):
    total_scenarios: int
    passed: int
    failed: int
    skipped: int
    not_executed: int


class AnalysisResponse(BaseModel):
    title: str
    actors: List[str]
    goal: str
    domain: str
    domain_mode: Mode
    business_object: str
    key_concerns: List[str]
    status: str
    generation_mode: Mode
    scenario_count: int
    generated_scenarios: List[QAScenario]
    failure_classification: str
    failure_evidence: List[str]
    risk_level: str
    execution_summary: ExecutionSummary
    business_rules: List[str]
    acceptance_criteria: List[str]
    ambiguities: List[str]


def get_workflow() -> Iterator[QAWorkflow]:
    # yield-dependency: FastAPI runs the code after `yield` once the response is sent, so the
    # LLM session opened for this request is always closed (dependency_overrides still work).
    with QAWorkflow() as workflow:
        yield workflow


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": settings.app_name, "environment": settings.app_env, "status": "running"}


@app.get("/health")
def health_check() -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "llm_enabled": settings.llm_enabled,
    }
    client = build_llm_client()
    try:
        llm = check_llm_health(client, timeout=settings.llm_healthcheck_timeout_seconds)
    finally:
        close_llm(client)
    if llm is not None:
        body["llm"] = llm  # already contains "model"
        if not (llm.get("reachable") and llm.get("model_available")):
            body["status"] = "degraded"
    return body


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_requirement(payload: RequirementRequest, workflow: QAWorkflow = Depends(get_workflow)) -> dict[str, Any]:
    try:
        state = workflow.run(payload.requirement)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    body = asdict(state)
    body.pop("requirement")
    body["generated_scenarios"] = state.generated_scenarios
    body["scenario_count"] = len(state.generated_scenarios)
    return body
