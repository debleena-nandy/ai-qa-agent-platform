from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


from app.llm import ollama_client
from app.llm.base import (
    ClosableLLM, HealthCheckableLLM, LLMClient, LLMError, LLMResponseError, StructuredLLM,
    check_llm_health, close_llm,
)
from app.llm.ollama_client import OllamaClient


class FakeLLM:
    def generate_structured(self, prompt: str, schema: type) -> Any:
        return schema()


class NotAnLLM:
    pass


def test_ollama_client_satisfies_all_protocols() -> None:
    client = OllamaClient("http://x", "llama3.1")
    assert isinstance(client, StructuredLLM)
    assert isinstance(client, HealthCheckableLLM)
    assert isinstance(client, ClosableLLM)


def test_fake_llm_only_needs_generate_structured() -> None:
    fake = FakeLLM()
    assert isinstance(fake, StructuredLLM)
    assert not isinstance(fake, HealthCheckableLLM)
    assert not isinstance(NotAnLLM(), StructuredLLM)
    assert LLMClient is StructuredLLM


def test_error_hierarchy_is_shared_with_ollama_client() -> None:
    assert ollama_client.LLMResponseError is LLMResponseError
    assert issubclass(LLMResponseError, LLMError)
    assert issubclass(LLMError, RuntimeError)


def test_check_llm_health() -> None:
    assert check_llm_health(None) is None
    fake_health = check_llm_health(FakeLLM())
    assert fake_health is not None
    assert fake_health["reachable"] is True
    client = OllamaClient("http://x", "llama3.1")
    client.health = MagicMock(return_value={"reachable": False, "model_available": False})  # type: ignore[method-assign]
    assert check_llm_health(client, timeout=1) == {"reachable": False, "model_available": False}
    client.health.assert_called_once_with(timeout=1)


def test_close_llm_is_safe_for_any_client() -> None:
    close_llm(None)
    close_llm(FakeLLM())
    client = OllamaClient("http://x", "llama3.1")
    client._session = MagicMock()
    close_llm(client)
    client._session.close.assert_called_once()
