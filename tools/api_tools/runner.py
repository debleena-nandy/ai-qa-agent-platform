# File: tools/api_tools/runner.py
# Description: Executes API request actions and assertions for a QA scenario.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

from core.models.actions import ApiAssertion, ApiRequestAction
from core.models.execution_result import AssertionOutcome, ExecutionResult, ExecutionStepResult
from core.models.scenario import QAScenario
from tools.api_tools.client import ApiTool
from tools.api_tools.target_policy import ResolvedTarget

PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]{0,39})\}\}")
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
Substituted = Tuple[str, Optional[Dict[str, Any]]]

class ApiScenarioRunner:
    """Executes validated API actions against one configured, policy-checked origin."""

    def __init__(
        self,
        base_url: str,
        allowed_hosts: set | None = None,
        allow_private: bool = False,
        api_tool: ApiTool | None = None,
        timeout_seconds: float = 10,
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
        ):
            raise ValueError("API target must be a plain HTTP origin without credentials, path, query, or fragment.")
        hosts = allowed_hosts if allowed_hosts is not None else {parsed.hostname.lower()}
        self._owns_tool = api_tool is None
        self.api_tool = api_tool or ApiTool(hosts, allow_private=allow_private)
        self.timeout_seconds = timeout_seconds
        try:
            self.target: ResolvedTarget = self.api_tool.validate_target_url(f"http://{parsed.netloc}")
        except ValueError:
            self.close()
            raise

    def close(self) -> None:
        if self._owns_tool:
            self.api_tool.close()

    def execute(self, scenario: QAScenario) -> ExecutionResult:
        step = ExecutionStepResult(
            scenario=f"{scenario.id} {scenario.title}", scenario_id=scenario.id,
            scenario_type=scenario.type, execution_type="api",
        )
        if not scenario.api_actions:
            step.status, step.details = "not_executed", "No structured API actions are available."
            return self._wrap(step)

        variables: Dict[str, Any] = {}
        failures: List[str] = []
        for index, action in enumerate(scenario.api_actions, start=1):
            try:
                path, body = self._substitute(action, variables)
            except KeyError as exc:
                step.evidence.append(f"Action {index}: unresolved variable {exc}.")
                failures.append(f"Action {index} depends on a value that an earlier action did not capture.")
                step.error_type = step.error_type or "UnresolvedVariable"
                break
            response = self.api_tool.request(action.method, self.target, path, timeout=self.timeout_seconds,
                                             headers=action.headers, json=body)
            if response["status_code"] in REDIRECT_STATUSES:
                response = {**response, "status_code": None, "error_type": "RedirectNotAllowed"}
            if response["status_code"] is None:
                error_type = response.get("error_type", "RequestError")
                step.error_type = step.error_type or error_type
                step.evidence.append(f"Action {index}: {action.method} {path} transport error ({error_type}).")
                failures.append(f"Action {index} could not get a usable response from the configured target.")
                break
            step.http_statuses.append(response["status_code"])
            step.duration_ms += response["elapsed_ms"]
            step.evidence.append(
                f"Action {index}: {action.method} {path} -> HTTP {response['status_code']} in {response['elapsed_ms']} ms."
            )
            for number, assertion in enumerate(action.assertions, start=1):
                outcome = self._check(assertion, response, variables)
                step.assertions.append(outcome)
                step.evidence.append(
                    f"Action {index}, assertion {number}: {'passed' if outcome.passed else 'failed'} ({outcome.description})."
                )
                if not outcome.passed:
                    failures.append(f"Action {index}, assertion {number}: {outcome.description}.")
            for name, json_path in action.capture.items():
                try:
                    variables[name] = ApiTool.json_path_value(response["json"], json_path)
                except (KeyError, ValueError, TypeError):
                    step.evidence.append(f"Action {index}: could not capture '{name}' from '{json_path}'.")

        if failures:
            step.status, step.details = "failed", " ".join(failures)
        else:
            step.status, step.details = "passed", "All API assertions passed."
        return self._wrap(step)

    @staticmethod
    def _substitute(action: ApiRequestAction, variables: Dict[str, Any]) -> Substituted:
        path = PLACEHOLDER.sub(lambda m: quote(str(variables[m.group(1)]), safe=""), action.path)
        body = action.json_body
        if body is not None:
            encoded = json.dumps(body)
            for name in PLACEHOLDER.findall(encoded):
                if name not in variables:
                    raise KeyError(name)
            body = json.loads(PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), encoded))
        return path, body

    @staticmethod
    def _check(assertion: ApiAssertion, response: Dict[str, Any], variables: Dict[str, Any]) -> AssertionOutcome:
        actual_status = response["status_code"]
        if assertion.type in ("status_code", "status_in"):
            expected = assertion.expected_statuses
            shown = expected if len(expected) > 1 else expected[0]
            return AssertionOutcome(
                kind=assertion.type, passed=actual_status in expected, expected_statuses=expected,
                actual_status=actual_status, description=f"HTTP status expected {shown}, received {actual_status}",
            )
        path = assertion.json_path or ""
        try:
            actual = ApiTool.json_path_value(response["json"], path)
            missing = False
        except (KeyError, ValueError, TypeError):
            actual, missing = None, True
        if assertion.type == "json_not_exists":
            return AssertionOutcome(kind=assertion.type, passed=missing, json_path=path, actual_status=actual_status,
                                    description=f"JSON path '{path}' {'is absent' if missing else 'is unexpectedly present'}")
        if missing:
            return AssertionOutcome(kind=assertion.type, passed=False, json_path=path, path_missing=True,
                                    actual_status=actual_status, description=f"JSON path '{path}' does not exist")
        if assertion.type == "json_exists":
            return AssertionOutcome(kind=assertion.type, passed=True, json_path=path, actual_status=actual_status,
                                    description=f"JSON path '{path}' exists")
        expected_value = assertion.expected
        if isinstance(expected_value, str):
            match = PLACEHOLDER.fullmatch(expected_value)
            if match and match.group(1) in variables:
                expected_value = variables[match.group(1)]
        passed = actual == expected_value
        # Values are never echoed into evidence: responses may contain secrets or personal data.
        return AssertionOutcome(kind=assertion.type, passed=passed, json_path=path, actual_status=actual_status,
                                description=f"JSON value at '{path}' {'matched' if passed else 'did not match'}")

    @staticmethod
    def _wrap(step: ExecutionStepResult) -> ExecutionResult:
        result = ExecutionResult()
        result.add_step(step)
        return result