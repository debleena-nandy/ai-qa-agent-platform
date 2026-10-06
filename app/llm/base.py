from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "T",
    "LLMError",
    "LLMResponseError",
    "LLMUnavailableError",
    "is_llm_unavailable",
    "StructuredLLM",
    "LLMClient",
    "HealthCheckableLLM",
    "ClosableLLM",
    "check_llm_health",
    "close_llm",
]


class LLMError(RuntimeError):
    """Base class for all LLM client errors."""


class LLMResponseError(LLMError):
    """Raised when the provider answers with an error status or an unexpected payload."""


class LLMUnavailableError(LLMError):
    """Raised when the provider cannot be reached or does not answer in time."""


def is_llm_unavailable(exc: BaseException) -> bool:
    # What: Tells "server down / timed out" apart from "server answered badly".
    # Why: Only an unreachable server justifies skipping the next LLM call in the same request;
    #      one invalid JSON answer does not mean the scenario call will fail too.
    return isinstance(exc, (LLMUnavailableError, ConnectionError, TimeoutError))


@runtime_checkable
class StructuredLLM(Protocol):
    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        """Return the model's answer parsed and validated as an instance of `schema`."""
        ...


LLMClient = StructuredLLM


@runtime_checkable
class HealthCheckableLLM(Protocol):
    def health(self, timeout: float = 3) -> Dict[str, Any]:
        """Return at least {"reachable": bool, "model_available": bool}."""
        ...


@runtime_checkable
class ClosableLLM(Protocol):
    def close(self) -> None:
        """Release network resources (sessions, sockets)."""
        ...


def check_llm_health(llm: Optional[StructuredLLM], timeout: float = 3) -> Optional[Dict[str, Any]]:
    if llm is None:
        return None
    if isinstance(llm, HealthCheckableLLM):
        return llm.health(timeout=timeout)
    return {"reachable": True, "model_available": True, "note": "client does not support health checks"}


def close_llm(llm: Optional[StructuredLLM]) -> None:
    if isinstance(llm, ClosableLLM):
        llm.close()
