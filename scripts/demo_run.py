# File: scripts/demo_run.py
# Description: Runs the QA design workflow against a synthetic requirement and prints the result.
# Author Name: Debleena Nandy
# Date: 08-10-2026
"""Print one complete QA analysis for a sample requirement.

Usage (from the project root):
    python scripts/demo_run.py              # real LLM (Ollama must be running with the configured model)
    python scripts/demo_run.py --fallback   # real LLM, rule-based fallback if an LLM step fails
    python scripts/demo_run.py --offline    # no Ollama: deterministic fake LLM from tests/fakes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/demo_run.py` from the project root (otherwise only scripts/ is on sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.encoders import jsonable_encoder  # noqa: E402

from config.settings import Settings, settings  # noqa: E402
from core.llm.agent_support import AgentLLMError, ensure_llm_ready  # noqa: E402
from core.llm.base import LLMResponseError, LLMUnavailableError, StructuredLLM, close_llm  # noqa: E402
from core.llm.ollama_client import build_llm_client  # noqa: E402
from graph.workflow import QAWorkflow  # noqa: E402

REQUIREMENT = (
    "As a customer, I want to place an order to buy a toy so that I can receive it at home.\n"
    "Acceptance criteria:\n"
    "- Customer can order at most 5 units per toy\n"
    "- Out-of-stock toys cannot be ordered"
)
LLM_FAILURES = (AgentLLMError, LLMUnavailableError, LLMResponseError)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QA workflow on a sample requirement.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true",
                      help="Use the deterministic fake LLM (no Ollama, no network).")
    mode.add_argument("--fallback", action="store_true",
                      help="Use Ollama, but fall back to rule-based logic when an LLM step fails.")
    parser.add_argument("--requirement", default=REQUIREMENT, help="Requirement text to analyse.")
    return parser.parse_args()


def build_llm(offline: bool) -> StructuredLLM:
    if offline:
        from tests.fakes import DemoFakeLLM  # test helper; imported lazily so normal runs don't need it
        return DemoFakeLLM()
    return build_llm_client()


def demo_config(offline: bool, fallback: bool) -> Settings:
    """The demo only designs tests; it never calls a real target or (offline) Ollama embeddings."""
    updates: dict[str, Any] = {"qa_api_base_url": ""}
    if offline or fallback:
        updates["rag_use_embeddings"] = False  # embeddings need a live Ollama
    return settings.model_copy(update=updates)


def main() -> int:
    args = parse_args()
    llm = build_llm(args.offline)
    try:
        if not (args.offline or args.fallback):
            ensure_llm_ready(llm)  # fail fast with a clear message instead of timing out mid-workflow
        workflow = QAWorkflow(llm, config=demo_config(args.offline, args.fallback),
                              use_fallback=True if args.fallback else None)
        result = workflow.run(args.requirement)
    except LLM_FAILURES as exc:
        print(f"LLM error: {exc}\nTip: start Ollama, or rerun with --fallback or --offline.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Invalid requirement: {exc}", file=sys.stderr)
        return 1
    finally:
        close_llm(llm)

    # jsonable_encoder handles the dataclass and the nested pydantic models (QAScenario, TestPlan, QAReport).
    print(json.dumps(jsonable_encoder(result), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
