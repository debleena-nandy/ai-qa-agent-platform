# File: agents/root_cause_agent/agent.py
# Description: Assesses likely root causes for reported test failures.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import json
from collections import Counter
from typing import List, cast

from pydantic import BaseModel, Field

from agents.failure_analysis_agent.agent import FailureReport
from core.llm.agent_support import fallback_enabled, generate, run_llm_step
from core.llm.base import StructuredLLM
from core.llm.prompt_safety import wrap_untrusted
from core.models.execution_result import ExecutionResult
from core.models.qa_report import FailureCategory, RootCauseAssessment
from tools.rag_tools.knowledge_base import KnowledgeBase

MAX_CONFIDENCE = 0.9

ROOT_CAUSE_PROMPT = """You are a principal QA engineer performing root-cause analysis across ALL failed tests
of one run (classified failures and retrieved known issues are in the data block).
Return:
- category: the single most probable underlying failure category (must be one of the observed categories)
- hypothesis: 2-4 sentences naming the probable root cause, citing scenario ids. Do not claim certainty.
- suggested_investigation: up to 5 concrete steps
- suggested_remediation: up to 5 concrete fixes for the team that owns the cause
- confidence (0-0.9) and confidence_basis: why the evidence supports that confidence

{evidence}
"""


class RootCauseLLM(BaseModel):
    category: FailureCategory
    hypothesis: str = Field(max_length=1200)
    suggested_investigation: List[str] = Field(default_factory=list, max_length=5)
    suggested_remediation: List[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(ge=0, le=1)
    confidence_basis: str = Field(max_length=400)


class RootCauseAgent:
    """Root Cause Analysis Agent (LLM). Guardrails: the category must match observed evidence, and the
    confidence is capped by how many failures share that category."""

    def __init__(self, llm: StructuredLLM, knowledge: KnowledgeBase | None = None, use_fallback: bool | None = None):
        self.llm = llm
        self.knowledge = knowledge
        self.use_fallback = fallback_enabled(use_fallback)

    def analyze(self, execution: ExecutionResult, failure: FailureReport) -> RootCauseAssessment:
        if execution.total_scenarios == 0 or execution.attempted == 0:
            return RootCauseAssessment(
                category="unknown", evidence=failure.evidence, confidence=None,
                hypothesis="No scenario execution evidence is available; root cause cannot be assessed.",
                suggested_investigation=["Configure a test target, design actions and approve the plan."],
                confidence_basis="No tests were executed.")
        if not failure.step_classifications:
            return RootCauseAssessment(
                category="unknown", evidence=failure.evidence, confidence=None,
                hypothesis="No failures were reported; no root-cause hypothesis is applicable.",
                confidence_basis="No failed scenario was reported.")
        similar = self._similar(failure)
        fallback = (lambda: self._by_vote(failure, similar)) if self.use_fallback else None
        result, _ = run_llm_step("root_cause_agent", lambda: self._with_llm(failure, similar), fallback)
        return result

    def _with_llm(self, failure: FailureReport, similar: List[str]) -> RootCauseAssessment:
        classes = failure.step_classifications
        payload = json.dumps({
            "classified_failures": [{"scenario": c.scenario, "category": c.category, "confidence": c.confidence,
                                     "rationale": c.rationale, "evidence": c.evidence[:4]} for c in classes],
            "known_issues": similar,
        }, indent=1)
        answer = generate(self.llm, ROOT_CAUSE_PROMPT.format(evidence=wrap_untrusted(payload, "evidence")),
                          RootCauseLLM)
        counts = Counter(c.category for c in classes)
        category= answer.category if answer.category in counts else counts.most_common(1)[0][0]
        agreement = counts[category] / len(classes)
        confidence = round(min(answer.confidence, agreement, MAX_CONFIDENCE), 2)
        basis = answer.confidence_basis.strip()
        if confidence < answer.confidence:
            basis += f" (capped: {counts[category]}/{len(classes)} failures support {category})"
        return RootCauseAssessment(
            category=cast(FailureCategory, category),
            hypothesis=answer.hypothesis.strip(),
            evidence=[f"{c.scenario_id}: {c.rationale}" for c in classes][:10],
            suggested_investigation=answer.suggested_investigation,
            suggested_remediation=answer.suggested_remediation,
            similar_known_issues=similar, confidence=confidence, confidence_basis=basis,
            hypothesis_source="llm")

    def _similar(self, failure: FailureReport) -> List[str]:
        if self.knowledge is None:
            return []
        query = " ".join(f"{c.category} {c.rationale}" for c in failure.step_classifications[:5])
        return [f"{h.chunk.title} (score {h.score})" for h in self.knowledge.search(query, top_k=3,
                                                                                    kind="failure_pattern")]

    @staticmethod
    def _by_vote(failure: FailureReport, similar: List[str]) -> RootCauseAssessment:
        classes = failure.step_classifications
        category, votes = Counter(c.category for c in classes).most_common(1)[0]
        mean = sum(c.confidence for c in classes if c.category == category) / votes
        return RootCauseAssessment(
            category=cast(FailureCategory, category), 
            hypothesis=f"{votes}/{len(classes)} failures point to {category.replace('_', ' ')}.",
            evidence=[f"{c.scenario_id}: {c.rationale}" for c in classes][:10],
            suggested_investigation=[f"Inspect evidence for {c.scenario_id}." for c in classes[:5]],
            similar_known_issues=similar, confidence=round(mean * votes / len(classes), 2),
            confidence_basis="Rule-based vote (LLM fallback).", hypothesis_source="rule_based")