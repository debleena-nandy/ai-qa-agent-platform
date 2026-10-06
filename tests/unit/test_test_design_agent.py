from __future__ import annotations

from app.agents.domain_agent import DomainAgent
from app.agents.requirement_agent import RequirementAgent
from app.agents.test_design_agent import TestDesignAgent

TOY_STORY = "As a customer, I want to place an order to buy a toy"
GENERIC_LABELS = [
    "the user completes the primary business action successfully",
    "invalid input is rejected without breaking the flow",
]


def _design(story: str, llm=None):
    analysis = RequirementAgent().analyze(story)
    domain, _ = DomainAgent(llm).extract(analysis, story)
    return TestDesignAgent(llm).generate_scenarios(analysis, domain, story)


def test_fallback_scenarios_are_concrete_and_domain_specific() -> None:
    scenarios, mode = _design(TOY_STORY)
    text = " ".join(s.title + " " + " ".join(s.steps) for s in scenarios).lower()

    assert mode == "rule_based"
    assert len(scenarios) >= 15
    for concern in ["toy", "stock", "payment", "shipping", "quantity", "duplicate"]:
        assert concern in text, f"missing coverage for {concern}"
    assert not any(label in text for label in GENERIC_LABELS)


def test_every_scenario_is_executable() -> None:
    scenarios, _ = _design(TOY_STORY)

    assert {s.type for s in scenarios} == {"functional", "negative", "boundary", "security", "api"}
    for scenario in scenarios:
        assert scenario.steps and scenario.expected_result
    assert [s.id for s in scenarios] == [f"TC-{i:03d}" for i in range(1, len(scenarios) + 1)]


def test_unknown_domain_still_uses_story_details() -> None:
    scenarios, _ = _design("As an admin, I want to export a monthly report")
    text = " ".join(s.title for s in scenarios).lower()

    assert "admin" in text and "report" in text


def test_explicit_acceptance_criteria_become_scenarios() -> None:
    story = TOY_STORY + "\nAcceptance criteria:\n- Customer can order at most 5 units per toy"
    scenarios, _ = _design(story)

    assert scenarios[0].expected_result == "Customer can order at most 5 units per toy"


def test_llm_scenarios_are_returned_intact(fake_llm_factory) -> None:
    llm = fake_llm_factory(scenario_count=12)
    scenarios, mode = _design(TOY_STORY, llm)

    assert mode == "llm"
    assert len(scenarios) == 12  # no truncation
    assert scenarios[0].id == "TC-001"
    assert "toy" in llm.prompts[-1] and "e-commerce" in llm.prompts[-1]


def test_too_few_llm_scenarios_falls_back(fake_llm_factory) -> None:
    scenarios, mode = _design(TOY_STORY, fake_llm_factory(scenario_count=3))

    assert mode == "rule_based"
    assert len(scenarios) >= 15


def test_llm_error_falls_back(fake_llm_factory) -> None:
    scenarios, mode = _design(TOY_STORY, fake_llm_factory(error=ValueError("invalid JSON")))

    assert mode == "rule_based"
    assert scenarios
