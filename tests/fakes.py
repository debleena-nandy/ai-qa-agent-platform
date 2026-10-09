# File: tests/fakes.py
# Description: Defines fake implementations for testing purposes.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List

from pydantic import BaseModel

from agents.failure_analysis_agent.agent import FailureClassificationLLM
from agents.reporting_agent.agent import ReportNarrative
from agents.requirement_agent.models import RequirementExtraction
from agents.requirement_agent.rules import rule_based_analysis
from agents.root_cause_agent.agent import RootCauseLLM
from agents.test_design_agent.action_design import ActionDesignStep, ToolCall
from agents.test_design_agent.action_library import DemoShopActionLibrary
from agents.test_design_agent.templates import TemplateScenarioGenerator
from agents.test_planning_agent.agent import PrioritisedPlan
from core.llm.base import LLMUnavailableError
from core.models.scenario import QAScenario, ScenarioDraft, ScenarioDraftSet
from demo_target.main import create_demo_app
from tools.api_tools.openapi import OpenApiSpec

APPROVAL_TOKEN = "test-token"
Handler = Callable[[str], BaseModel]
Texts = List[str]
DRAFT_FIELDS = set(ScenarioDraft.model_fields)
PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}


def untrusted(prompt: str, label: str) -> str:
    match = re.search(rf"<untrusted_{label}>\n(.*)\n</untrusted_{label}>", prompt, re.DOTALL)
    if match is None:
        raise AssertionError(f"Prompt has no <untrusted_{label}> block.")
    return match.group(1)


class DemoFakeLLM:
    """Deterministic stand-in for the local model, dispatching on the requested schema."""

    def __init__(self, tool_calls: bool = True):
        self.calls: Texts = []
        self.prompts: Texts = []
        self.tool_calls = tool_calls
        self.spec = OpenApiSpec(create_demo_app(set()).openapi())
        self.library = DemoShopActionLibrary()
        self.handlers: Dict[str, Handler] = {
            "RequirementExtraction": self._requirement,
            "ScenarioDraftSet": self._scenarios,
            "ActionDesignStep": self._action,
            "PrioritisedPlan": self._plan,
            "FailureClassificationLLM": self._classify,
            "RootCauseLLM": self._root_cause,
            "ReportNarrative": self._report,
        }

    def generate_structured(self, prompt: str, schema: type) -> Any:
        self.calls.append(schema.__name__)
        self.prompts.append(prompt)
        return self.handlers[schema.__name__](prompt)

    def _requirement(self, prompt: str) -> BaseModel:
        story = untrusted(prompt, "requirement")
        analysis, domain = rule_based_analysis(story)
        return RequirementExtraction(
            title=analysis.title, actors=analysis.actors, goal=analysis.goal or "complete the action",
            business_object=analysis.business_object or "item", benefit=analysis.benefit,
            business_rules=analysis.business_rules, acceptance_criteria=analysis.acceptance_criteria,
            missing_requirements=analysis.missing_requirements, ambiguities=analysis.ambiguities,
            testable_conditions=analysis.testable_conditions, domain=domain.domain, key_concerns=domain.key_concerns)

    def _scenarios(self, prompt: str) -> BaseModel:
        if "coverage gaps" in prompt:
            return ScenarioDraftSet(scenarios=[])
        story = json.loads(untrusted(prompt, "requirement"))["user_story"]
        analysis, domain = rule_based_analysis(story)
        generated = TemplateScenarioGenerator().generate(analysis, domain)
        return ScenarioDraftSet(scenarios=[ScenarioDraft(**s.model_dump(include=DRAFT_FIELDS)) for s in generated])

    def _action(self, prompt: str) -> BaseModel:
        if self.tool_calls and "Tool results so far:\nnone" in prompt:
            return ActionDesignStep(tool_call=ToolCall(name="list_endpoints"))
        data = json.loads(untrusted(prompt, "scenario"))
        candidate = QAScenario(id="TC-X", title=data["title"], type=data["type"], steps=data["steps"],
                               expected_result=data["expected_result"])
        designed = self.library.design(candidate, self.spec)
        if designed is None or not designed.api_actions:
            return ActionDesignStep(cannot_automate="The target exposes no API for this scenario.")
        return ActionDesignStep(api_actions=designed.api_actions)

    def _plan(self, prompt: str) -> BaseModel:
        items = json.loads(untrusted(prompt, "plan"))
        ordered = sorted(items, key=lambda item: PRIORITY_RANK.get(item["priority"], 1))
        order = [item["id"] for item in ordered]
        return PrioritisedPlan(order=order, risk_notes={sid: "High business risk." for sid in order[:2]})

    def _classify(self, prompt: str) -> BaseModel:
        payload = json.loads(untrusted(prompt, "evidence"))
        signal = payload["deterministic_signal"]
        return FailureClassificationLLM(category=signal["category"], confidence=0.8,
                                        rationale=f"Evidence indicates: {signal['rationale']}",
                                        evidence_used=payload["evidence"][:2])

    def _root_cause(self, prompt: str) -> BaseModel:
        payload = json.loads(untrusted(prompt, "evidence"))
        categories = [item["category"] for item in payload["classified_failures"]]
        top = max(set(categories), key=categories.count)
        return RootCauseLLM(category=top, hypothesis=f"Most failures point to {top}.",
                            suggested_investigation=["Reproduce the first failing request."],
                            suggested_remediation=["Fix the underlying rule and re-run the regression scope."],
                            confidence=0.85, confidence_basis="Consistent evidence across failures.")

    def _report(self, prompt: str) -> BaseModel:
        run = json.loads(untrusted(prompt, "run"))
        failed = [item["id"] for item in run["failed"]]
        summary = run["summary"]
        return ReportNarrative(
            executive_summary=f"{summary['passed']} passed and {summary['failed']} failed.",
            risk_areas=[f"Failures in {', '.join(failed)}"] if failed else [],
            regression_scope=failed + ["TC-999"],  # TC-999 does not exist and must be filtered out
            recommendations=["Re-run the regression scope after fixes."])


class ScriptedLLM:
    """Returns (or raises) prepared responses in order."""

    def __init__(self, *responses: Any):
        self.responses = list(responses)
        self.calls: Texts = []
        self.prompts: Texts = []

    def generate_structured(self, prompt: str, schema: type) -> Any:
        self.calls.append(schema.__name__)
        self.prompts.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class DeadLLM:
    def generate_structured(self, prompt: str, schema: type) -> Any:
        raise LLMUnavailableError("Ollama is not reachable (test).")