from __future__ import annotations

import pytest

from app.agents.requirement_agent import RequirementAgent


def test_extracts_actor_goal_and_business_object() -> None:
    analysis = RequirementAgent().analyze("As a customer, I want to place an order to buy a toy")

    assert analysis.actors == ["customer"]
    assert analysis.goal == "place an order to buy a toy"
    assert analysis.business_object == "toy"


def test_extracts_benefit_and_explicit_acceptance_criteria() -> None:
    story = (
        "As a customer, I want to place an order to buy a toy so that it is delivered to me.\n"
        "Acceptance criteria:\n"
        "- Customer can order at most 5 units per toy\n"
        "- Out-of-stock toys cannot be ordered"
    )
    analysis = RequirementAgent().analyze(story)

    assert analysis.benefit == "it is delivered to me"
    assert len(analysis.acceptance_criteria) == 2
    assert "at most 5 units" in analysis.business_rules[0]
    assert analysis.ambiguities == []


def test_story_is_not_repeated_as_a_business_rule() -> None:
    analysis = RequirementAgent().analyze("As a customer, I want to log in to my account securely.")

    assert analysis.business_rules == []


def test_reports_missing_information_as_ambiguities() -> None:
    analysis = RequirementAgent().analyze("Build the toy ordering feature quickly")
    text = " ".join(analysis.ambiguities).lower()

    assert analysis.actors == []
    assert "actor" in text
    assert "acceptance criteria" in text


def test_cancel_is_not_detected_as_can_rule() -> None:
    analysis = RequirementAgent().analyze("As a customer, I want to cancel an order.\nThe cancel button is red.")

    assert analysis.business_rules == []


def test_rejects_empty_requirement() -> None:
    with pytest.raises(ValueError, match="Requirement text cannot be empty"):
        RequirementAgent().analyze("   ")
