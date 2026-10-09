# File: tests/unit/test_test_design_agent.py
# Description: Tests generation of structured QA scenarios from requirements.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import pytest

from agents.requirement_agent.agent import RequirementAgent
from agents.test_design_agent.agent import TestDesignAgent
from core.llm.agent_support import AgentLLMError
from tests.conftest import MIN_SCENARIOS
from tests.fakes import DeadLLM

TOY_STORY = "As a customer, I want to place an order to buy a toy"
TOY_WITH_AC = TOY_STORY + "\nAcceptance criteria:\n- Customer can order at most 5 units per toy"
GENERIC_LABELS = [
    "the user completes the primary business action successfully",
    "invalid input is rejected without breaking the flow",
]


def _design(story: str, llm, fallback: bool = False):
    """Deterministic requirement analysis, then test design with the given LLM."""
    analysis, domain, _ = RequirementAgent(DeadLLM(), use_fallback=True).analyze(story)
    return TestDesignAgent(llm, use_fallback=fallback).generate_scenarios(analysis, domain, story)


# ---- rule-based fallback --------------------------------------------------------------------------
def test_fallback_scenarios_are_concrete_and_domain_specific() -> None:
    scenarios, mode = _design(TOY_STORY, DeadLLM(), fallback=True)
    text = " ".join(s.title + " " + " ".join(s.steps) for s in scenarios).lower()
    assert mode == "rule_based"
    assert len(scenarios) >= MIN_SCENARIOS
    assert "toy" in text
    assert not any(label in text for label in GENERIC_LABELS)


def test_scenarios_have_reviewable_steps_and_sequential_ids() -> None:
    scenarios, _ = _design(TOY_STORY, DeadLLM(), fallback=True)
    for scenario in scenarios:
        assert scenario.steps and scenario.expected_result
    assert [s.id for s in scenarios] == [f"TC-{i:03d}" for i in range(1, len(scenarios) + 1)]


def test_explicit_acceptance_criteria_become_scenarios() -> None:
    scenarios, _ = _design(TOY_WITH_AC, DeadLLM(), fallback=True)
    assert any(s.criterion_ids == ["AC-001"] and s.expected_result == "Customer can order at most 5 units per toy"
               for s in scenarios)


def test_llm_failure_without_fallback_raises() -> None:
    with pytest.raises(AgentLLMError):
        _design(TOY_STORY, DeadLLM(), fallback=False)


# ---- LLM path ---------------------------------------------------------------------------------------
def test_llm_scenarios_are_linked_to_acceptance_criteria(fake_llm_factory) -> None:
    scenarios, mode = _design(TOY_WITH_AC, fake_llm_factory(scenario_count=15))
    assert mode == "llm"
    assert any("AC-001" in s.criterion_ids for s in scenarios)  # added by the coverage guard
    assert all(set(s.criterion_ids) <= {"AC-001"} for s in scenarios)


def test_llm_scenarios_are_renumbered_and_grounded_in_context(fake_llm_factory) -> None:
    llm = fake_llm_factory(scenario_count=15)
    scenarios, mode = _design(TOY_STORY, llm)
    assert mode == "llm"
    assert len(scenarios) == 15
    assert scenarios[0].id == "TC-001"
    assert "toy" in llm.prompts[0] and "e-commerce" in llm.prompts[0]


def test_too_few_llm_scenarios_falls_back(fake_llm_factory) -> None:
    scenarios, mode = _design(TOY_STORY, fake_llm_factory(scenario_count=3), fallback=True)
    assert mode == "rule_based"
    assert len(scenarios) >= MIN_SCENARIOS


def test_llm_error_falls_back(fake_llm_factory) -> None:
    scenarios, mode = _design(TOY_STORY, fake_llm_factory(error=ValueError("invalid JSON")), fallback=True)
    assert mode == "rule_based"
    assert scenarios