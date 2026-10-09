# File: tests/unit/test_llm_base.py
# Description: Tests the shared structured language model interfaces, errors, and startup gate.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from core.llm import ollama_client
from core.llm.agent_support import ensure_llm_ready
from core.llm.base import (
    ClosableLLM, HealthCheckableLLM, LLMClient, LLMError, LLMResponseError, LLMUnavailableError, StructuredLLM,
    check_llm_health, close_llm,
)
from core.llm.ollama_client import OllamaClient, build_llm_client

MODEL = "qwen2.5:3b"


class FakeLLM:
    def generate_structured(self, prompt: str, schema: type) -> Any:
        return schema()


class NotAnLLM:
    pass


def test_ollama_client_satisfies_all_protocols() -> None:
    client = OllamaClient("http://x", MODEL)
    assert isinstance(client, StructuredLLM)
    assert isinstance(client, HealthCheckableLLM)
    assert isinstance(client, ClosableLLM)


def test_build_llm_client_always_returns_a_client() -> None:
    client = build_llm_client()
    try:
        assert isinstance(client, OllamaClient)
    finally:
        client.close()


def test_fake_llm_only_needs_generate_structured() -> None:
    assert isinstance(FakeLLM(), StructuredLLM)
    assert not isinstance(FakeLLM(), HealthCheckableLLM)
    assert not isinstance(NotAnLLM(), StructuredLLM)
    assert LLMClient is StructuredLLM


def test_error_hierarchy_is_shared_with_ollama_client() -> None:
    assert ollama_client.LLMResponseError is LLMResponseError
    assert issubclass(LLMResponseError, LLMError)
    assert issubclass(LLMError, RuntimeError)


def test_check_llm_health() -> None:
    assert check_llm_health(None) is None
    fake_health = check_llm_health(FakeLLM())
    assert fake_health is not None and fake_health["reachable"] is True
    client = OllamaClient("http://x", MODEL)
    client.health = MagicMock(return_value={"reachable": False, "model_available": False})  # type: ignore[method-assign]
    assert check_llm_health(client, timeout=1) == {"reachable": False, "model_available": False}
    client.health.assert_called_once_with(timeout=1)


def test_ensure_llm_ready_gate() -> None:
    client = OllamaClient("http://x", MODEL)
    client.health = MagicMock(return_value={"reachable": True, "model_available": False})  # type: ignore[method-assign]
    with pytest.raises(LLMUnavailableError, match="not ready"):
        ensure_llm_ready(client, timeout=1)
    client.health = MagicMock(return_value={"reachable": True, "model_available": True})  # type: ignore[method-assign]
    assert ensure_llm_ready(client, timeout=1)["model_available"] is True


def test_close_llm_is_safe_for_any_client() -> None:
    close_llm(None)
    close_llm(FakeLLM())
    client = OllamaClient("http://x", MODEL)
    client._session = MagicMock()
    close_llm(client)
    client._session.close.assert_called_once()