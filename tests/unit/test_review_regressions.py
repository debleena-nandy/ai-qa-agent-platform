from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest
import requests
from fastapi.testclient import TestClient

from app.agents.domain_agent import DomainAgent
from app.agents.requirement_agent import RequirementAgent
from app.api.main import app, get_workflow
from app.config.settings import settings
from app.execution.executor import execution_type_for
from app.graph.workflow import QAWorkflow
from app.llm.base import LLMUnavailableError, is_llm_unavailable
from app.llm.ollama_client import OllamaClient
from app.models.scenario import DomainProfile, QAScenario, ScenarioSet

TOY = "As a customer, I want to place an order to buy a toy"


# ---- requirement_agent: wrapped stories --------------------------------
def test_rule_line_is_not_swallowed_into_goal() -> None:
    analysis = RequirementAgent().analyze("As a customer, I want to buy a toy\nquantity must be at most 5")
    assert analysis.goal == "buy a toy"
    assert analysis.business_rules == ["quantity must be at most 5"]


def test_wrapped_story_with_benefit_is_joined() -> None:
    analysis = RequirementAgent().analyze("As a customer,\nI want to buy a toy\nso that my kid is happy")
    assert (analysis.actors, analysis.goal, analysis.benefit) == (["customer"], "buy a toy", "my kid is happy")


def test_wrapped_goal_is_joined() -> None:
    assert RequirementAgent().analyze("As a customer, I want to buy\na wooden toy").goal == "buy a wooden toy"


# ---- domain_agent ------------------------------------------------------
@pytest.mark.parametrize("story", ["As a user, I want to keep shopping later", "As a user, I want to see what I paid"])
def test_irregular_ecommerce_forms(story: str) -> None:
    analysis = RequirementAgent().analyze(story)
    assert DomainAgent().extract(analysis, story)[0].domain == "e-commerce"


# ---- executor routing --------------------------------------------------
def _sc(steps: List[str], kind: Any = "security") -> QAScenario:
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
    assert execution_type_for(_sc(steps, "functional")) == expected


# ---- workflow: LLM skip only when unreachable --------------------------
class ScriptedLLM:
    def __init__(self, domain_error: Exception) -> None:
        self.domain_error = domain_error
        self.calls: List[type] = []

    def generate_structured(self, prompt: str, schema: type) -> Any:
        self.calls.append(schema)
        if schema is DomainProfile:
            raise self.domain_error
        return ScenarioSet(scenarios=[_sc(["s"], "api") for _ in range(10)])


def test_unreachable_llm_is_called_once() -> None:
    llm = ScriptedLLM(LLMUnavailableError("down"))
    state = QAWorkflow(llm_client=llm).run(TOY)
    assert llm.calls == [DomainProfile]
    assert state.generation_mode == "rule_based"


def test_bad_domain_answer_still_tries_scenarios() -> None:
    llm = ScriptedLLM(ValueError("invalid JSON"))
    state = QAWorkflow(llm_client=llm).run(TOY)
    assert llm.calls == [DomainProfile, ScenarioSet]
    assert (state.domain_mode, state.generation_mode) == ("rule_based", "llm")


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


# ---- API: sessions are closed, health is honest -----------------------
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


def test_workflow_dependency_closes_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ClosableFake()
    monkeypatch.setattr("app.graph.workflow.build_llm_client", lambda: fake)
    gen = get_workflow()
    next(gen)
    with pytest.raises(StopIteration):
        next(gen)
    assert fake.closed == 1


def test_health_degraded_and_closes_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ClosableFake({"reachable": True, "model_available": False, "model": "llama3.1"})
    monkeypatch.setattr(settings, "llm_enabled", True)
    monkeypatch.setattr("app.api.main.build_llm_client", lambda: fake)
    body = TestClient(app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["llm"]["model"] == "llama3.1"
    assert fake.closed == 1


def test_oversized_requirement_returns_422(client: TestClient) -> None:
    payload = {"requirement": "As a user, " + "x" * (settings.max_requirement_chars + 1)}
    assert client.post("/analyze", json=payload).status_code == 422
