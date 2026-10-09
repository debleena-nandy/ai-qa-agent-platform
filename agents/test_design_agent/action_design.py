"""Validate explicit API actions before planning or test execution."""
# File: agents/test_design_agent/action_design.py
# Description: Ensures proposed API actions are restricted to documented target operations.
# Author Name: Debleena Nandy
# Date: 07-10-2026

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from config.settings import settings
from core.llm.agent_support import AgentLLMError, fallback_enabled, generate
from core.llm.base import LLMResponseError, LLMUnavailableError, StructuredLLM
from core.llm.prompt_safety import wrap_untrusted
from core.models.actions import ApiRequestAction
from core.models.scenario import QAScenario
from tools.api_tools.openapi import OpenApiSpec

logger = logging.getLogger(__name__)
ToolName = Literal["list_endpoints", "get_endpoint_schema"]
ToolOutput = Tuple[str, bool]


class ToolCall(BaseModel):
    name: ToolName  # only these two read-only tools exist; anything else fails validation
    arguments: Dict[str, str] = Field(default_factory=dict)


class ActionDesignStep(BaseModel):
    thought: str = Field(default="", max_length=500)
    tool_call: Optional[ToolCall] = None
    api_actions: Optional[List[ApiRequestAction]] = None
    cannot_automate: Optional[str] = Field(default=None, description="Reason when no safe API test exists")


@dataclass
class ToolTraceEntry:
    scenario_id: str
    step: int
    tool: str
    arguments: Dict[str, str]
    ok: bool
    note: str = ""


@dataclass
class ActionDesignReport:
    llm: int = 0
    rule_based: int = 0
    unresolved: List[str] = field(default_factory=list)
    tool_trace: List[ToolTraceEntry] = field(default_factory=list)
    spec_error: Optional[str] = None


DesignOutput = Tuple[List[QAScenario], ActionDesignReport]

ACTION_PROMPT = """You are a QA automation engineer. Convert ONE test scenario into executable HTTP API actions
for the target described by its OpenAPI document. You may call read-only tools to inspect the API.

Tools:
- list_endpoints(): METHOD /path - summary
- get_endpoint_schema(method, path): parameters, JSON body schema, documented response codes

Rules:
- Respond with exactly one of: tool_call, api_actions, cannot_automate.
- Use only endpoints from the OpenAPI document. Paths are relative, e.g. /orders.
- Every action needs at least one assertion (status_code, status_in, json_equals, json_exists, json_not_exists).
- To reuse a response value, set capture {{"order_id": "id"}} and reference it as {{{{order_id}}}}.
- Authentication header for the test user: {auth_header}. Never invent other credentials.
- If the scenario needs a browser, external systems or data the API cannot produce, use cannot_automate.

{scenario}

Tool results so far:
{transcript}
"""


class ActionDesignAgent:
    """Tool-calling agent: grounds scenarios in the target's OpenAPI spec and returns validated actions.
    It can only READ the API description; execution happens later and only after human approval."""

    def __init__(self, llm: StructuredLLM, auth_header: str = "Authorization: Bearer demo-customer-token",
                 max_tool_steps: int | None = None, use_fallback: bool | None = None):
        self.llm = llm
        self.auth_header = auth_header
        self.max_tool_steps = max_tool_steps or settings.llm_max_tool_steps
        self.use_fallback = fallback_enabled(use_fallback)

    def design(self, scenarios: List[QAScenario], spec: OpenApiSpec) -> DesignOutput:
        report = ActionDesignReport()
        library = self._library() if self.use_fallback else None
        llm_available = True
        designed: List[QAScenario] = []
        for scenario in scenarios:
            if scenario.api_actions or scenario.browser_actions:
                designed.append(scenario)
                continue
            result: QAScenario | None = None
            if llm_available and scenario.type != "ui":
                try:
                    result = self._design_with_llm(scenario, spec, report)
                except LLMUnavailableError as exc:
                    if library is None:
                        raise AgentLLMError(f"action_design: LLM unavailable ({exc}); fallback disabled.") from exc
                    llm_available = False
                except LLMResponseError as exc:
                    report.tool_trace.append(ToolTraceEntry(scenario.id, 0, "final", {}, False, str(exc)[:200]))
            if result is not None:
                report.llm += 1
            elif library is not None:
                result = library.design(scenario, spec)
                if result is not None:
                    report.rule_based += 1
            if result is None:
                report.unresolved.append(scenario.id)
                designed.append(scenario)
            else:
                designed.append(result)
        return designed, report

    def _design_with_llm(self, scenario: QAScenario, spec: OpenApiSpec, report: ActionDesignReport) -> QAScenario | None:
        transcript: List[str] = []
        scenario_block = wrap_untrusted(json.dumps({
            "title": scenario.title, "type": scenario.type, "steps": scenario.steps,
            "expected_result": scenario.expected_result}, indent=1), "scenario")
        for step in range(1, self.max_tool_steps + 1):
            prompt = ACTION_PROMPT.format(scenario=scenario_block, auth_header=self.auth_header,
                                          transcript="\n".join(transcript) or "none")
            decision = generate(self.llm, prompt, ActionDesignStep)
            if decision.tool_call is not None:
                output, ok = self._run_tool(decision.tool_call, spec)
                report.tool_trace.append(ToolTraceEntry(scenario.id, step, decision.tool_call.name,
                                                        decision.tool_call.arguments, ok))
                transcript.append(f"[{decision.tool_call.name}({decision.tool_call.arguments})] -> {output[:1500]}")
                continue
            if decision.cannot_automate:
                report.tool_trace.append(ToolTraceEntry(scenario.id, step, "final", {}, True,
                                                        decision.cannot_automate[:200]))
                return None
            actions = decision.api_actions or []
            unknown = [f"{a.method} {a.path}" for a in actions if not spec.has(a.method, a.path)]
            if not actions or unknown:  # guardrail against hallucinated endpoints
                note = f"rejected: endpoints not in spec {unknown}" if unknown else "rejected: no actions"
                report.tool_trace.append(ToolTraceEntry(scenario.id, step, "final", {}, False, note))
                transcript.append(f"Your answer was {note}. Use list_endpoints.")
                continue
            report.tool_trace.append(ToolTraceEntry(scenario.id, step, "final", {}, True, f"{len(actions)} actions"))
            return scenario.model_copy(update={"api_actions": actions, "action_source": "llm"})
        return None

    @staticmethod
    def _run_tool(call: ToolCall, spec: OpenApiSpec) -> ToolOutput:
        if call.name == "list_endpoints":
            return "\n".join(spec.list_endpoints()), True
        method, path = call.arguments.get("method", ""), call.arguments.get("path", "")
        if not method or not path:
            return "error: get_endpoint_schema requires method and path", False
        return json.dumps(spec.endpoint_schema(method, path))[:3000], True

    @staticmethod
    def _library() -> Any:
        from agents.test_design_agent.action_library import DemoShopActionLibrary  # optional module
        return DemoShopActionLibrary()