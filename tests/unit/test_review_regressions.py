# File: tests/unit/test_review_regressions.py
# Description: Tests regression cases for previously identified QA workflow issues.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient

from agents.execution_agent.routing import execution_type_for
from agents.requirement_agent.rules import rule_based_analysis
from api.main import app, get_llm
from config.settings import settings
from core.llm.agent_support import AgentLLMError
from core.llm.base import LLMUnavailableError, is_llm_unavailable
from core.llm.ollama_client import OllamaClient
from core.models.scenario import DomainProfile, QAScenario
from graph.workflow import QAWorkflow
from tests.fakes import DeadLLM

TOY = "As a customer, I want to place an order to buy a toy"


# ---- requirement parsing: wrapped stories --------------------------------------------------------
def test_rule_line_is_not_swallowed_into_goal() -> None:
    analysis, _ = rule_based_analysis("As a customer, I want to buy a toy\nquantity must be at most 5")
    assert analysis.goal == "buy a toy"
    assert analysis.business_rules == ["quantity must be at most 5"]


def test_wrapped_story_with_benefit_is_joined() -> None:
    analysis, _ = rule_based_analysis("As a customer,\nI want to buy a toy\nso that my kid is happy")
    assert (analysis.actors, analysis.goal, analysis.benefit) == (["customer"], "buy a toy", "my kid is happy")


def test_wrapped_goal_is_joined() -> None:
    assert rule_based_analysis("As a customer, I want to buy\na wooden toy")[0].goal == "buy a wooden toy"


# ---- domain detection ---------------------------------------------------------------------------------
@pytest.mark.parametrize("story", ["As a user, I want to keep shopping later", "As a user, I want to see what I paid"])
def test_irregular_ecommerce_forms(story: str) -> None:
    assert rule_based_analysis(story)[1].domain == "e-commerce"


# ---- executor routing ---------------------------------------------------------------------------------
def _sc(steps: List[str], kind: Any = "functional") -> QAScenario:
    return QAScenario(id="1", title="t", type=kind, steps=steps, expected_result="r")


@pytest.mark.parametrize(
    "steps, expected",
    [
        (["Request GET /orders/<id> with the customer's token"], "api"),
        (["PUT the profile with valid data", "GET the profile"], "api"),
        (["Send an update request for the other user's id"], "api"),
        (["Open the API docs page in the browser"], "browser"),
        (["Put the item in the cart"], "browser"),
        (["Proceed to checkout"], "browser"),
    ],
)
def test_execution_routing(steps: List[str], expected: str) -> None:
    assert execution_type_for(_sc(steps)) == expected


# ---- LLM availability ---------------------------------------------------------------------------------
def test_unreachable_llm_fails_fast_without_fallback() -> None:
    with pytest.raises(AgentLLMError):
        QAWorkflow(DeadLLM(), use_fallback=False).run(TOY)


def test_unreachable_llm_uses_rule_based_path_with_fallback() -> None:
    state = QAWorkflow(DeadLLM(), use_fallback=True).run(TOY)
    assert (state.domain_mode, state.generation_mode) == ("rule_based", "rule_based")
    assert state.domain == "e-commerce"


def test_unavailable_classification() -> None:
    assert is_llm_unavailable(LLMUnavailableError("x"))
    assert is_llm_unavailable(TimeoutError())
    assert not is_llm_unavailable(ValueError())


def test_ollama_wraps_connection_errors() -> None:
    client = OllamaClient("http://x", "m")
    client._session = MagicMock()
    client._session.post.side_effect = requests.ConnectionError("refused")
    with pytest.raises(LLMUnavailableError, match="not reachable"):
        client.generate_structured("p", DomainProfile)


# ---- API: sessions are closed, health is honest ------------------------------------------------------
class ClosableFake:
    """Real class (not MagicMock): runtime_checkable protocols inspect attributes statically."""

    def __init__(self, health: Any = None) -> None:
        self.closed = 0
        self._health = health

    def generate_structured(self, prompt: str, schema: type) -> Any:
        raise LLMUnavailableError("not used")

    def health(self, timeout: float = 3) -> Any:
        return self._health

    def close(self) -> None:
        self.closed += 1


def test_llm_dependency_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ClosableFake()
    monkeypatch.setattr("api.main.build_llm_client", lambda: fake)
    gen = get_llm()
    assert next(gen) is fake
    with pytest.raises(StopIteration):
        next(gen)
    assert fake.closed == 1


def test_health_degraded_and_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ClosableFake({"reachable": True, "model_available": False, "model": "qwen2.5:3b"})
    monkeypatch.setattr("api.main.build_llm_client", lambda: fake)
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["llm"]["model"] == "qwen2.5:3b"
    assert fake.closed == 1


def test_oversized_requirement_returns_422(client: TestClient) -> None:
    payload = {"requirement": "As a user, " + "x" * (settings.max_requirement_chars + 1)}
    assert client.post("/analyze", json=payload).status_code == 422
