from __future__ import annotations

from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config.settings import settings
from app.models.scenario import DomainProfile, QAScenario, ScenarioSet


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests must never call a real LLM, even if .env enables it.
    monkeypatch.setattr(settings, "llm_enabled", False)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class FakeLLM:
    """Returns canned structured responses and records the prompts it received."""

    def __init__(self, responses: Dict[type, Any] | None = None, error: Exception | None = None):
        self.responses = responses or {}
        self.error = error
        self.prompts: List[str] = []

    def generate_structured(self, prompt: str, schema: type) -> Any:
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return self.responses[schema]


def make_llm_scenarios(count: int) -> ScenarioSet:
    return ScenarioSet(scenarios=[
        QAScenario(
            id=f"X{i}", title=f"LLM scenario {i} for toy order", type="functional",
            steps=["step"], expected_result="result",
        )
        for i in range(count)
    ])


@pytest.fixture
def fake_llm_factory():
    def _factory(scenario_count: int = 12, error: Exception | None = None) -> FakeLLM:
        return FakeLLM(
            responses={
                DomainProfile: DomainProfile(domain="e-commerce", business_object="toy", key_concerns=["payment"]),
                ScenarioSet: make_llm_scenarios(scenario_count),
            },
            error=error,
        )
    return _factory
