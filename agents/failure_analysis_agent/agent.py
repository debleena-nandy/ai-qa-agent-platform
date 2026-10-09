# File: agents/failure_analysis_agent/agent.py
# Description: Classifies test execution outcomes and summarizes failures.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field

from agents.failure_analysis_agent.rules import RuleSignal, rule_signal
from core.llm.agent_support import fallback_enabled, generate, run_llm_step
from core.llm.base import StructuredLLM
from core.llm.prompt_safety import wrap_untrusted
from core.models.execution_result import ExecutionResult, ExecutionStepResult
from core.models.qa_report import FailureCategory, StepClassification
from tools.rag_tools.knowledge_base import KnowledgeBase

MAX_CONFIDENCE = 0.9

CLASSIFY_PROMPT = """You are a senior QA failure analyst. Classify ONE failed test into exactly one category:
application_defect, automation_defect, test_data_issue, environment_issue, dependency_issue,
network_issue, configuration_issue, unknown.
Rules:
- Decide only from the evidence block. Copy the exact evidence lines you relied on into evidence_used.
- If the application never answered (transport error, no HTTP status) it cannot be an application_defect.
- Prefer 'unknown' with low confidence over guessing. Confidence must not exceed 0.9.
- 'deterministic_signal' is a hint from evidence rules; you may disagree, but explain why in the rationale.
- 'known_failure_patterns' are retrieved from past failures; use them only if the evidence matches.

{evidence}
"""


class FailureClassificationLLM(BaseModel):
    category: FailureCategory
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(max_length=600)
    evidence_used: List[str] = Field(default_factory=list, max_length=8)


@dataclass
class FailureReport:
    classification: str
    evidence: List[str] = field(default_factory=list)
    risk_level: str = "medium"
    step_classifications: List[StepClassification] = field(default_factory=list)
    dominant_category: Optional[str] = None


class FailureAnalysisAgent:
    """Failure Analysis Agent: the LLM classifies each failure from structured runner evidence
    (assertions, HTTP statuses, console/network logs, artifacts, RAG patterns), checked by guardrails."""

    def __init__(self, llm: StructuredLLM, knowledge: KnowledgeBase | None = None, use_fallback: bool | None = None):
        self.llm = llm
        self.knowledge = knowledge
        self.use_fallback = fallback_enabled(use_fallback)

    def analyze(self, execution: ExecutionResult) -> FailureReport:
        if execution.total_scenarios == 0:
            return FailureReport("not_executed", ["No scenarios were generated or executed."])
        failed = [step for step in execution.steps if step.status == "failed"]
        if failed:
            classes = [self.classify_step(step) for step in failed]
            categories = [c.category for c in classes]
            return FailureReport(
                classification="failures_detected",
                evidence=[f"{s.scenario}: {s.details or 'no details'}" for s in failed],
                risk_level="high" if "application_defect" in categories else "medium",
                step_classifications=classes,
                dominant_category=max(set(categories), key=categories.count),
            )
        if execution.passed and execution.not_executed == 0:
            note = [f"{execution.skipped} scenarios were excluded by the reviewer."] if execution.skipped else []
            return FailureReport("baseline_validation_pass", note, "low")
        if execution.attempted == 0:
            return FailureReport("not_executed", [
                f"{execution.total_scenarios} scenarios planned; none executed "
                f"(skipped={execution.skipped}, not_executed={execution.not_executed})."])
        return FailureReport("partially_executed", [
            f"passed={execution.passed}, skipped={execution.skipped}, not_executed={execution.not_executed}"])

    def classify_step(self, step: ExecutionStepResult) -> StepClassification:
        signal = rule_signal(step)
        payload = self._payload(step, signal)
        fallback = (lambda: self._from_signal(step, signal)) if self.use_fallback else None
        result, _ = run_llm_step("failure_analysis_agent", lambda: self._with_llm(step, signal, payload), fallback)
        return result

    def _with_llm(self, step: ExecutionStepResult, signal: RuleSignal, payload: str) -> StepClassification:
        answer = generate(self.llm, CLASSIFY_PROMPT.format(evidence=wrap_untrusted(payload, "evidence")),
                          FailureClassificationLLM)
        category, confidence = answer.category, min(answer.confidence, MAX_CONFIDENCE)
        notes: List[str] = []
        if step.execution_type == "api" and not step.http_statuses and category == "application_defect":
            category, confidence = signal.category, min(confidence, signal.confidence)
            notes.append("Guardrail: the application never responded, so an application defect was not accepted.")
        grounded = [q.strip() for q in answer.evidence_used if q.strip() and q.strip()[:80] in payload]
        if not grounded:
            confidence = round(confidence * 0.5, 2)
            notes.append("Guardrail: the model cited no verifiable evidence; confidence halved.")
        return StepClassification(
            scenario_id=step.scenario_id or step.scenario.split(" ")[0], scenario=step.scenario,
            category=category, confidence=round(confidence, 2),
            rationale=" ".join([answer.rationale.strip(), *notes]).strip(),
            evidence=(grounded or step.evidence)[:8], classification_source="llm",
        )

    def _payload(self, step: ExecutionStepResult, signal: RuleSignal) -> str:
        patterns = []
        if self.knowledge is not None:
            query = " ".join([signal.category, signal.rationale, step.details or ""])
            patterns = [f"{h.chunk.title}: {h.chunk.text[:300]}"
                        for h in self.knowledge.search(query, top_k=2, kind="failure_pattern")]
        return json.dumps({
            "scenario": step.scenario, "scenario_type": step.scenario_type, "runner": step.execution_type,
            "error_type": step.error_type, "http_statuses": step.http_statuses,
            "failed_assertions": [
                {"kind": a.kind, "expected_statuses": a.expected_statuses, "actual_status": a.actual_status,
                 "json_path": a.json_path, "path_missing": a.path_missing, "description": a.description}
                for a in step.assertions if not a.passed],
            "evidence": step.evidence[:15], "console_errors": step.console_errors[:5], "artifacts": step.artifacts,
            "deterministic_signal": {"category": signal.category, "rationale": signal.rationale},
            "known_failure_patterns": patterns,
        }, indent=1)

    @staticmethod
    def _from_signal(step: ExecutionStepResult, signal: RuleSignal) -> StepClassification:
        return StepClassification(
            scenario_id=step.scenario_id or step.scenario.split(" ")[0], scenario=step.scenario,
            category=signal.category, confidence=signal.confidence,  
            rationale=signal.rationale, evidence=step.evidence[:8], classification_source="rule_based")