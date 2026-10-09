# File: tests/conftest.py
# Description: Defines shared pytest fixtures for application and agent tests.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import socket
from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from time import monotonic, sleep
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import pytest
import uvicorn
from fastapi.testclient import TestClient

from agents.test_design_agent.agent import REQUIRED_TYPES
from api.main import app, get_llm, get_run_repository
from config.settings import settings
from core.models.scenario import ScenarioDraft, ScenarioDraftSet
from core.persistence.run_repository import RunRepository
from demo_target.main import create_demo_app
from tests.fakes import DemoFakeLLM

UrlIterator = Iterator[str]
ClientIterator = Iterator[TestClient]
DEMO_AUTH = {"Authorization": "Bearer demo-customer-token"}
MIN_SCENARIOS = 8  # mirrors TEST_DESIGN_MIN_SCENARIOS


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Tests never depend on a developer's .env, Ollama, or a real target."""
    values: Dict[str, Any] = {
        "qa_api_base_url": "", "qa_approval_token": "", "browser_enabled": False,
        "analyze_requires_token": False, "rule_based_fallback": False, "llm_check_on_startup": False,
        "rag_use_embeddings": False, "llm_max_retries": 1, "qa_artifacts_dir": str(tmp_path / "artifacts"),
        "knowledge_dir": "tools/rag_tools/corpus", "qa_allow_private_targets": True,
        "test_design_min_scenarios": MIN_SCENARIOS,
    }
    for name, value in values.items():
        monkeypatch.setattr(settings, name, value)


@pytest.fixture
def fake_llm() -> DemoFakeLLM:
    return DemoFakeLLM()


@pytest.fixture
def client(fake_llm: DemoFakeLLM, tmp_path: Path) -> ClientIterator:
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_run_repository] = lambda: RunRepository(tmp_path / "runs.sqlite3")
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


# ---- configurable fake LLM ------------------------------------------------------------------------
class ConfigurableFakeLLM:
    """Delegates to DemoFakeLLM, but controls the scenario count and can simulate a failing model."""

    def __init__(self, scenario_count: int = 15, error: Optional[Exception] = None):
        self.scenario_count = scenario_count
        self.error = error
        self.prompts: List[str] = []
        self._delegate = DemoFakeLLM()

    def generate_structured(self, prompt: str, schema: type) -> Any:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        if schema is ScenarioDraftSet:
            return make_llm_scenarios(self.scenario_count)
        return self._delegate.generate_structured(prompt, schema)


def make_llm_scenarios(count: int) -> ScenarioDraftSet:
    """`count` unique drafts that rotate through every required scenario type."""
    types = list(REQUIRED_TYPES)
    return ScenarioDraftSet(scenarios=[
        ScenarioDraft(
            id=f"X{i}", title=f"LLM scenario {i} for toy order ({types[i % len(types)]})",
            type=types[i % len(types)], steps=["Open the toy page", "Place the order"],
            expected_result="The order behaves as specified",
        )
        for i in range(count)
    ])


@pytest.fixture
def fake_llm_factory() -> Callable[..., ConfigurableFakeLLM]:
    def _factory(scenario_count: int = 15, error: Optional[Exception] = None) -> ConfigurableFakeLLM:
        return ConfigurableFakeLLM(scenario_count=scenario_count, error=error)
    return _factory


# ---- demo target ------------------------------------------------------------------------------------
@contextmanager
def run_demo(defects: Optional[Set[str]] = None) -> UrlIterator:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_demo_app(defects or set()), log_level="critical", lifespan="off"))
    thread = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    deadline = monotonic() + 5
    while not server.started and thread.is_alive() and monotonic() < deadline:
        sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        listener.close()
        raise RuntimeError("Demo target failed to start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()


@pytest.fixture
def demo_target() -> UrlIterator:
    with run_demo() as url:
        yield url