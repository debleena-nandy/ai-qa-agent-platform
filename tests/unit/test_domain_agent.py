from __future__ import annotations

import pytest

from app.agents.domain_agent import DomainAgent
from app.agents.requirement_agent import RequirementAgent


@pytest.mark.parametrize(
    "story, expected_domain",
    [
        ("As a customer, I want to place an order to buy a toy", "e-commerce"),
        ("As a customer, I want to log in to my account securely.", "authentication"),
        ("As a user, I want to upload a profile photo", "account-management"),
        ("As an admin, I want to export a monthly report", "general"),
    ],
)
def test_rule_based_domain_detection(story: str, expected_domain: str) -> None:
    analysis = RequirementAgent().analyze(story)
    profile, mode = DomainAgent().extract(analysis, story)

    assert mode == "rule_based"
    assert profile.domain == expected_domain
    assert profile.key_concerns


def test_uses_llm_when_available(fake_llm_factory) -> None:
    story = "As a customer, I want to place an order to buy a toy"
    profile, mode = DomainAgent(fake_llm_factory()).extract(RequirementAgent().analyze(story), story)

    assert mode == "llm"
    assert profile.business_object == "toy"


def test_falls_back_when_llm_fails(fake_llm_factory) -> None:
    story = "As a customer, I want to place an order to buy a toy"
    llm = fake_llm_factory(error=TimeoutError("ollama timeout"))
    profile, mode = DomainAgent(llm).extract(RequirementAgent().analyze(story), story)

    assert mode == "rule_based"
    assert profile.domain == "e-commerce"
