# File: agents/requirement_agent/agent.py
# Description: Analyzes requirements and extracts testable acceptance criteria.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from agents.requirement_agent.domain import normalize_domain
from agents.requirement_agent.models import RequirementAnalysis, RequirementExtraction
from core.llm.agent_support import fallback_enabled, generate, run_llm_step
from core.llm.base import LLMResponseError, StructuredLLM
from core.llm.prompt_safety import detect_injection, wrap_untrusted
from core.models.scenario import DomainProfile

logger = logging.getLogger(__name__)
AnalysisResult = Tuple[RequirementAnalysis, DomainProfile, str]
Pair = Tuple[RequirementAnalysis, DomainProfile]
Lines = List[str]
WORD = re.compile(r"[a-z0-9]+")
STOP = {"a", "an", "the", "is", "are", "be", "to", "of", "and", "or", "in", "on", "for", "with", "it", "can", "must"}
GROUNDING_RATIO = 0.7

REQUIREMENT_PROMPT = """You are a senior QA analyst. Analyse the user story in the data block.
Return:
- title, actors, goal, business_object (1-3 words), benefit ("" if no 'so that')
- business_rules: explicit rules only (limits, validations, permissions, states). Never invent rules.
- acceptance_criteria: ONLY criteria written in the story, copied verbatim. A Given/When/Then block is ONE
  criterion joined into one line. Return [] when the story has none.
- missing_requirements: information a tester needs but the story does not give
  (numeric limits, error behaviour, who is allowed, observable outcome)
- ambiguities: vague, subjective or contradictory wording
- testable_conditions: concrete checks derived from the rules, including boundary values
  (e.g. "quantity 5 is accepted and 6 is rejected")
- domain: e-commerce, authentication, account-management, payments, booking, reporting or general
- key_concerns: 5-10 areas a QA analyst must test for this story

{story}
"""


class RequirementAgent:
    """Requirement Analysis Agent: one structured LLM call extracts the story, rules, actors, acceptance
    criteria, missing requirements, ambiguities, testable conditions and the business domain."""

    def __init__(self, llm: StructuredLLM, use_fallback: bool | None = None):
        self.llm = llm
        self.use_fallback = fallback_enabled(use_fallback)

    def analyze(self, requirement_text: str) -> AnalysisResult:
        text = (requirement_text or "").strip()
        if not text:
            raise ValueError("Requirement text cannot be empty.")
        fallback = (lambda: self._rule_based(text)) if self.use_fallback else None
        (analysis, domain), mode = run_llm_step("requirement_agent", lambda: self._with_llm(text), fallback)
        if detect_injection(text):
            analysis.security_flags = ["possible_prompt_injection"]
            analysis.ambiguities.append(
                "Requirement contains instruction-like text aimed at the AI; it was treated as data only.")
        return analysis, domain, mode

    def _with_llm(self, text: str) -> Pair:
        result = generate(self.llm, REQUIREMENT_PROMPT.format(story=wrap_untrusted(text)), RequirementExtraction)
        if not result.goal.strip():
            raise LLMResponseError("Requirement extraction returned no goal.")
        explicit = _explicit_criteria(text)
        criteria: Lines = []
        discarded: Lines = []
        if explicit:
            # Criteria written in the story are the contract: take them verbatim, never from the model.
            criteria = explicit
        else:
            for criterion in _clean(result.acceptance_criteria):
                (criteria if _grounded(criterion, text) else discarded).append(criterion)
        ambiguities = _clean(result.ambiguities)
        if discarded:  # hallucination guardrail: criteria must come from the story
            logger.info("Discarded %d ungrounded acceptance criteria: %s", len(discarded), discarded)
            ambiguities.append(f"{len(discarded)} acceptance criteria proposed by the model were not "
                               "found in the story and were discarded.")
        business_object = result.business_object.strip().lower() or "item"
        analysis = RequirementAnalysis(
            title=(result.title.strip() or text.splitlines()[0])[:120],
            actors=_clean(result.actors),
            goal=result.goal.strip(),
            business_object=business_object,
            benefit=result.benefit.strip(),
            business_rules=_clean(result.business_rules)[:15],
            acceptance_criteria=criteria,
            ambiguities=ambiguities,
            missing_requirements=_clean(result.missing_requirements)[:10],
            testable_conditions=_clean(result.testable_conditions)[:20],
        )
        domain = DomainProfile(domain=normalize_domain(result.domain), business_object=business_object,
                               key_concerns=_clean(result.key_concerns)[:10])
        return analysis, domain

    @staticmethod
    def _rule_based(text: str) -> Pair:
        from agents.requirement_agent.rules import rule_based_analysis  # optional module
        return rule_based_analysis(text)


def _clean(items: List[str]) -> Lines:
    seen: Lines = []
    for item in items:
        value = (item or "").strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def _grounded(criterion: str, source: str) -> bool:
    words = {w for w in WORD.findall(criterion.lower()) if w not in STOP}
    if not words:
        return False
    return len(words & set(WORD.findall(source.lower()))) / len(words) >= GROUNDING_RATIO

def _explicit_criteria(text: str) -> Lines:
    """Acceptance criteria written as list items in the story, parsed deterministically (no LLM)."""
    try:
        from agents.requirement_agent.rules import rule_based_analysis  # optional module
    except ImportError:
        return []
    try:
        return list(rule_based_analysis(text)[0].acceptance_criteria)
    except ValueError:
        return []