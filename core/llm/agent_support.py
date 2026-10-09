# File: core/llm/agent_support.py
# Description: Shared LLM helpers for agents (self-correcting generation, fallback, startup gate).
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Literal, Tuple, TypeVar

from pydantic import BaseModel

from config.settings import settings
from core.llm.base import LLMResponseError, LLMUnavailableError, StructuredLLM, check_llm_health

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
StepMode = Literal["llm", "rule_based"]
StepOutcome = Tuple[Any, StepMode]
Health = Dict[str, Any]


class AgentLLMError(RuntimeError):
    """An LLM step failed and the optional rule-based fallback is disabled."""


def fallback_enabled(override: bool | None) -> bool:
    return settings.rule_based_fallback if override is None else override


def generate(llm: StructuredLLM, prompt: str, schema: type[T], retries: int | None = None) -> T:
    """Structured call with self-correction: a rejected answer is fed back to the model with the error."""
    attempts = 1 + (settings.llm_max_retries if retries is None else retries)
    feedback = ""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return llm.generate_structured(prompt + feedback, schema)
        except LLMUnavailableError:
            raise
        except (LLMResponseError, ValueError) as exc:  # pydantic ValidationError is a ValueError
            last_error = exc
            feedback = (f"\n\nYour previous answer was rejected: {str(exc)[:300]}\n"
                        "Return only JSON that matches the schema.")
    raise LLMResponseError(f"{schema.__name__}: no valid answer after {attempts} attempts ({last_error}).")


def run_llm_step(step: str, llm_call: Callable[[], Any], fallback: Callable[[], Any] | None = None) -> StepOutcome:
    """Runs an LLM step. The fallback is used only when the caller supplied one (RULE_BASED_FALLBACK=true)."""
    try:
        return llm_call(), "llm"
    except Exception as exc:
        if fallback is not None:
            logger.warning("%s: LLM step failed (%s); using optional rule-based fallback.", step, exc)
            return fallback(), "rule_based"
        raise AgentLLMError(
            f"{step}: LLM step failed ({type(exc).__name__}: {str(exc)[:200]}). "
            "Rule-based fallback is disabled (RULE_BASED_FALLBACK=false)."
        ) from exc


def ensure_llm_ready(llm: StructuredLLM, timeout: float | None = None) -> Health:
    """Startup gate: the platform does not run without a reachable model."""
    health = check_llm_health(llm, timeout=timeout or settings.llm_healthcheck_timeout_seconds) or {}
    if not (health.get("reachable") and health.get("model_available")):
        raise LLMUnavailableError(
            f"LLM is mandatory but not ready ({health}). Start Ollama and run: ollama pull {settings.ollama_model}"
        )
    return health