# File: tests/unit/test_domain_agent.py
# Description: Tests rule-based and LLM domain detection (part of the Requirement Analysis Agent).
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import pytest

from agents.requirement_agent.agent import RequirementAgent
from agents.requirement_agent.domain import normalize_domain
from core.llm.agent_support import AgentLLMError
from tests.fakes import DeadLLM

STORY = "As a customer, I want to place an order to buy a toy"


@pytest.mark.parametrize(
    "story, expected_domain",
    [
        (STORY, "e-commerce"),
        ("As a customer, I want to log in to my account securely.", "authentication"),
        ("As a user, I want to upload a profile photo", "account-management"),
        ("As an admin, I want to export a monthly report", "general"),
    ],
)
def test_rule_based_domain_detection(story: str, expected_domain: str) -> None:
    _, profile, mode = RequirementAgent(DeadLLM(), use_fallback=True).analyze(story)
    assert mode == "rule_based"
    assert profile.domain == expected_domain
    assert profile.key_concerns


def test_uses_llm_when_available(fake_llm_factory) -> None:
    _, profile, mode = RequirementAgent(fake_llm_factory()).analyze(STORY)
    assert mode == "llm"
    assert profile.domain == "e-commerce"
    assert profile.business_object == "toy"


def test_falls_back_when_llm_fails_and_fallback_enabled(fake_llm_factory) -> None:
    llm = fake_llm_factory(error=TimeoutError("ollama timeout"))
    _, profile, mode = RequirementAgent(llm, use_fallback=True).analyze(STORY)
    assert mode == "rule_based"
    assert profile.domain == "e-commerce"


def test_llm_failure_is_raised_when_fallback_disabled(fake_llm_factory) -> None:
    llm = fake_llm_factory(error=TimeoutError("ollama timeout"))
    with pytest.raises(AgentLLMError):
        RequirementAgent(llm, use_fallback=False).analyze(STORY)


@pytest.mark.parametrize(
    "raw, expected",
    [("E-Commerce", "e-commerce"), ("online shopping", "e-commerce"), ("Auth", "authentication"),
     ("user_profile", "account-management"), ("", "general"), ("Booking System", "booking-system")],
)
def test_normalize_domain(raw: str, expected: str) -> None:
    assert normalize_domain(raw) == expected