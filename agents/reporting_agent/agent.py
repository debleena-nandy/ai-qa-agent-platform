# File: agents/reporting_agent/agent.py
# Description: Builds structured QA reports from execution and analysis results.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import json
from typing import Dict, List

from pydantic import BaseModel, Field

from agents.failure_analysis_agent.agent import FailureReport
from core.llm.agent_support import fallback_enabled, generate, run_llm_step
from core.llm.base import StructuredLLM
from core.llm.prompt_safety import wrap_untrusted
from core.models.execution_result import ExecutionResult
from core.models.qa_report import DefectCandidate, QAReport, RootCauseAssessment
from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario
from core.models.test_plan import TestPlan
from evaluation.evaluators.traceability import TraceabilityEvaluator

DEFECT_THRESHOLD = 0.45
Lines = List[str]
ScenarioIndex = Dict[str, QAScenario]

REPORT_PROMPT = """You are a QA lead writing the summary of a test run for a release decision.
All numbers in the data block are computed by the platform: use them exactly, never recalculate or invent.
Return:
- executive_summary: 3-5 sentences (outcome, main risk, release recommendation, what still needs a human)
- risk_areas: up to 8 business or technical areas at risk, each citing scenario or AC ids
- regression_scope: scenario ids that should be re-run after a fix (failed ones plus closely related ones)
- recommendations: up to 6 next actions

{run}
"""


class ReportNarrative(BaseModel):
    executive_summary: str = Field(max_length=1500)
    risk_areas: List[str] = Field(default_factory=list, max_length=8)
    regression_scope: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list, max_length=6)


class QAReportAgent:
    """QA Report Agent. Statistics, coverage and defect candidates are computed deterministically; the LLM
    writes the executive summary, risk areas, regression scope and recommendations (validated against ids)."""

    def __init__(self, llm: StructuredLLM | None, traceability: TraceabilityEvaluator | None = None,
                 use_fallback: bool | None = None):
        self.llm = llm
        self.traceability = traceability or TraceabilityEvaluator()
        self.use_fallback = fallback_enabled(use_fallback)

    def generate(self, plan: TestPlan, criteria: List[AcceptanceCriterion], scenarios: List[QAScenario],
                 execution: ExecutionResult, failure: FailureReport, root_cause: RootCauseAssessment,
                 workflow_status: str, execution_status: str) -> QAReport:
        coverage = self.traceability.evaluate(criteria, scenarios)
        by_id = {s.id: s for s in scenarios}
        failed = [step for step in execution.steps if step.status == "failed"]
        candidates = [
            DefectCandidate(scenario_id=c.scenario_id, title=by_id[c.scenario_id].title if c.scenario_id in by_id
                            else c.scenario, category=c.category, confidence=c.confidence, evidence=c.evidence[:5])
            for c in failure.step_classifications
            if c.category == "application_defect" and c.confidence >= DEFECT_THRESHOLD]
        report = QAReport(
            plan_id=plan.id, workflow_status=workflow_status, execution_status=execution_status,
            scenario_count=len(scenarios), criterion_count=coverage.criterion_count,
            covered_criterion_ids=coverage.covered_criterion_ids,
            uncovered_criterion_ids=coverage.uncovered_criterion_ids,
            unknown_criterion_ids=coverage.unknown_criterion_ids,
            execution_summary=execution.summary(),
            pass_rate=round(execution.passed / execution.attempted, 3) if execution.attempted else None,
            failure_classification=failure.classification,
            failed_tests=[f"{s.scenario}: {s.details}" for s in failed],
            step_classifications=failure.step_classifications, root_cause=root_cause,
            defect_candidates=candidates,
            artifacts=sorted({a for s in execution.steps for a in s.artifacts}),
            recommendations=self._baseline_recommendations(plan, execution_status, candidates),
        )
        if execution.attempted == 0 or self.llm is None:
            return report  # nothing executed yet: no narrative to write
        fallback = (lambda: self._rule_based(report, by_id)) if self.use_fallback else None
        narrative, mode = run_llm_step("reporting_agent", lambda: self._with_llm(report, by_id, plan), fallback)
        failed_ids = [s.scenario_id for s in failed]
        scope = list(dict.fromkeys(failed_ids + [sid for sid in narrative.regression_scope if sid in by_id]))
        return report.model_copy(update={
            "executive_summary": narrative.executive_summary.strip(),
            "risk_areas": narrative.risk_areas,
            "recommended_regression_scope": scope,
            "recommendations": list(dict.fromkeys(report.recommendations + narrative.recommendations)),
            "narrative_source": mode,
        })

    def _with_llm(self, report: QAReport, by_id: ScenarioIndex, plan: TestPlan) -> ReportNarrative:
        assert self.llm is not None
        run = {
            "summary": report.execution_summary, "pass_rate": report.pass_rate,
            "coverage": {"covered": report.covered_criterion_ids, "uncovered": report.uncovered_criterion_ids},
            "failed": [{"id": c.scenario_id,
                        "title": by_id[c.scenario_id].title if c.scenario_id in by_id else c.scenario,
                        "category": c.category, "confidence": c.confidence,
                        "criteria": by_id[c.scenario_id].criterion_ids if c.scenario_id in by_id else []}
                       for c in report.step_classifications],
            "root_cause": report.root_cause.model_dump(include={"category", "hypothesis", "confidence"}),
            "defect_candidates": [d.scenario_id for d in report.defect_candidates],
            "not_executed_high_priority": [i.scenario_id for i in plan.scenarios if i.eligibility != "eligible" and i.scenario_id in by_id and by_id[i.scenario_id].priority == "high"],
            "all_scenarios": [{"id": s.id, "title": s.title, "type": s.type, "criteria": s.criterion_ids} for s in by_id.values()],
        }
        return generate(self.llm, REPORT_PROMPT.format(run=wrap_untrusted(json.dumps(run), "run")), ReportNarrative)

    @staticmethod
    def _rule_based(report: QAReport, by_id: ScenarioIndex) -> ReportNarrative:
        risks = [f"{by_id[c.scenario_id].type}: {by_id[c.scenario_id].title} ({c.category})"
                 for c in report.step_classifications if c.scenario_id in by_id]
        if report.uncovered_criterion_ids:
            risks.append(f"Acceptance criteria without scenarios: {', '.join(report.uncovered_criterion_ids)}")
        s = report.execution_summary
        return ReportNarrative(
            executive_summary=f"{s['passed']} passed, {s['failed']} failed, {s['skipped']} skipped. "
                              f"Probable cause: {report.root_cause.category}.",
            risk_areas=risks[:8], regression_scope=[c.scenario_id for c in report.step_classifications])

    @staticmethod
    def _baseline_recommendations(plan: TestPlan, execution_status: str, candidates: List[DefectCandidate]) -> Lines:
        items = []
        if execution_status != "completed":
            items.append("Do not treat this report as an execution result until planned tests have run.")
        if plan.approval_status != "approved":
            items.append("Review and approve the test plan before enabling execution.")
        if any(i.eligibility == "blocked" for i in plan.scenarios):
            items.append("Design actions for blocked scenarios or exclude them with a reason.")
        if candidates:
            items.append("Confirm defect candidates manually before raising defects.")
        return items