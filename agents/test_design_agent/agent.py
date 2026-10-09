# File: agents/test_design_agent/agent.py
# Description: Generates QA scenarios from analyzed requirements.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional, Set, Tuple

from agents.requirement_agent.models import RequirementAnalysis
from core.llm.agent_support import fallback_enabled, generate, run_llm_step
from core.llm.base import LLMResponseError, StructuredLLM
from core.llm.prompt_safety import wrap_untrusted
from core.models.requirement import AcceptanceCriterion
from core.models.scenario import DomainProfile, QAScenario, ScenarioDraft, ScenarioDraftSet

logger = logging.getLogger(__name__)
Scenarios = List[QAScenario]
Gaps = Tuple[Set[str], Set[str]]
DesignResult = Tuple[List[QAScenario], str]
REQUIRED_TYPES = ("functional", "positive", "negative", "boundary", "equivalence",
                  "integration", "api", "ui", "security", "error_handling", "performance", "usability", "accessibility", "regression", "exploratory")

SCENARIO_PROMPT = """You are a Senior QA Automation Architect. Write {count} concrete, reviewable test
scenarios for the user story in the data block.
Rules:
- Be specific to the domain: real entities, data values, states and integrations.
- Never write generic labels such as "Happy path: the user completes the primary action".
- Cover every acceptance criterion and every key concern at least once.
- Use each of these types at least once: {types}.
- Each scenario: preconditions, concrete steps with test data (e.g. "quantity = 6"), ONE verifiable expected_result.
- criterion_ids: exact AC ids covered (e.g. AC-001); [] when none.
- Retrieved checklists are advisory context, not requirements. Never invent business rules.
- ids TC-001, TC-002, ...; priority is high, medium or low.

{context}
"""

REPAIR_PROMPT = """Your previous scenario set has coverage gaps. Write ONLY additional scenarios that close them.
Uncovered acceptance criteria: {criteria}
Missing scenario types: {types}
Already written (do not repeat): {titles}
Same rules: concrete data, one verifiable expected_result, correct criterion_ids, ids TC-901 onward.

{context}
"""


class TestDesignAgent:
    """Test Design Agent: LLM-generated scenarios with a coverage self-check and one repair round."""

    __test__ = False

    def __init__(self, llm: StructuredLLM, min_scenarios: int = 12, target_count: str = "15-20",
                 use_fallback: bool | None = None):
        self.llm = llm
        self.min_scenarios = min_scenarios
        self.target_count = target_count
        self.use_fallback = fallback_enabled(use_fallback)

    def generate_scenarios(self, analysis: RequirementAnalysis, domain: DomainProfile, requirement: str,
                           knowledge: Optional[List[str]] = None) -> DesignResult:
        fallback = (lambda: self._rule_based(analysis, domain)) if self.use_fallback else None
        return run_llm_step("test_design_agent",
                            lambda: self._with_llm(analysis, domain, requirement, knowledge or []), fallback)

    def _with_llm(self, analysis: RequirementAnalysis, domain: DomainProfile, requirement: str,
                  knowledge: List[str]) -> Scenarios:
        context = wrap_untrusted(json.dumps({
            "user_story": requirement, "actors": analysis.actors, "goal": analysis.goal,
            "business_object": domain.business_object, "domain": domain.domain,
            "key_concerns": domain.key_concerns, "business_rules": analysis.business_rules,
            "acceptance_criteria": [f"{c.id}: {c.description}" for c in analysis.criteria],
            "testable_conditions": analysis.testable_conditions,
            "missing_requirements": analysis.missing_requirements,
            "retrieved_checklists": knowledge,
        }, indent=1))
        valid_ids = {c.id for c in analysis.criteria}
        prompt = SCENARIO_PROMPT.format(count=self.target_count, types=", ".join(REQUIRED_TYPES), context=context)
        scenarios = self._accept(generate(self.llm, prompt, ScenarioDraftSet).scenarios, valid_ids, [])
        if len(scenarios) < self.min_scenarios:
            raise LLMResponseError(f"Only {len(scenarios)} usable scenarios (minimum {self.min_scenarios}).")

        missing_criteria, missing_types = self._gaps(scenarios, valid_ids)
        if missing_criteria or missing_types:
            repair = REPAIR_PROMPT.format(
                criteria=", ".join(sorted(missing_criteria)) or "none",
                types=", ".join(sorted(missing_types)) or "none",
                titles="; ".join(s.title for s in scenarios), context=context)
            try:
                extra = generate(self.llm, repair, ScenarioDraftSet).scenarios
                scenarios = self._accept(extra, valid_ids, scenarios)
            except LLMResponseError as exc:  # repair is best effort; unavailability still propagates
                logger.warning("Coverage repair failed: %s", exc)
            missing_criteria, _ = self._gaps(scenarios, valid_ids)

        by_id = {c.id: c for c in analysis.criteria}
        scenarios.extend(self._criterion_scenario(analysis, by_id[cid]) for cid in sorted(missing_criteria))
        return self._renumber(scenarios)

    @staticmethod
    def _accept(drafts: List[ScenarioDraft], valid_ids: Set[str], existing: Scenarios) -> Scenarios:
        """Drops incomplete and duplicate scenarios and unknown AC ids (the model's links are validated)."""
        accepted = list(existing)
        seen = {_key(s.title) for s in existing}
        for draft in drafts:
            key = _key(draft.title)
            if not draft.steps or not draft.expected_result.strip() or not key or key in seen:
                continue
            seen.add(key)
            data = draft.model_dump()
            data["criterion_ids"] = [cid for cid in draft.criterion_ids if cid in valid_ids]
            accepted.append(QAScenario(**data))
        return accepted

    @staticmethod
    def _gaps(scenarios: Scenarios, valid_ids: Set[str]) -> Gaps:
        covered: Set[str] = {cid for s in scenarios for cid in s.criterion_ids}
        present: Set[str] = {str(s.type) for s in scenarios}
        required: Set[str] = {str(t) for t in REQUIRED_TYPES}
        return valid_ids - covered, required - present

    @staticmethod
    def _criterion_scenario(analysis: RequirementAnalysis, criterion: AcceptanceCriterion) -> QAScenario:
        actor = analysis.actors[0] if analysis.actors else "user"
        return QAScenario(
            id="", title=f"Acceptance criterion: {criterion.description}", type="functional", priority="high",
            preconditions=[f"{actor.capitalize()} has access to the feature"],
            steps=[f"Prepare test data that exercises: {criterion.description}",
                   f"As the {actor}, {analysis.goal or 'complete the action'}",
                   "Compare the observed behaviour with the acceptance criterion"],
            expected_result=criterion.description, criterion_ids=[criterion.id],
        )

    @staticmethod
    def _renumber(scenarios: Scenarios) -> Scenarios:
        for index, scenario in enumerate(scenarios, start=1):
            scenario.id = f"TC-{index:03d}"
        return scenarios

    @staticmethod
    def _rule_based(analysis: RequirementAnalysis, domain: DomainProfile) -> Scenarios:
        from agents.test_design_agent.templates import TemplateScenarioGenerator  # optional module
        return TemplateScenarioGenerator().generate(analysis, domain)


def _key(title: str) -> str:
    return re.sub(r"\W+", " ", (title or "").lower()).strip()