# File: agents/failure_analysis_agent/rules.py
# Description: Classifies test execution outcomes and summarizes failures.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

from dataclasses import dataclass

from core.models.execution_result import ExecutionStepResult
from core.models.qa_report import FailureCategory

NETWORK_ERRORS = {"ConnectTimeout", "ReadTimeout", "Timeout", "TimeoutError", "SSLError"}
ENVIRONMENT_ERRORS = {"ConnectionError", "NavigationError", "ProxyError"}
CONFIGURATION_ERRORS = {"RedirectNotAllowed", "InvalidURL", "MissingSchema"}


@dataclass(frozen=True)
class RuleSignal:
    category: FailureCategory
    confidence: float
    rationale: str


def rule_signal(step: ExecutionStepResult) -> RuleSignal:
    error = step.error_type or ""
    if error in NETWORK_ERRORS and step.execution_type == "api":
        return RuleSignal("network_issue", 0.6, f"Transport timeout ({error}) before the application answered.")
    if error in ENVIRONMENT_ERRORS:
        return RuleSignal("environment_issue", 0.75, f"Target unreachable ({error}); the application was not exercised.")
    if error in CONFIGURATION_ERRORS:
        return RuleSignal("configuration_issue", 0.6, f"{error}: the target or base URL is likely misconfigured.")
    if error == "UnresolvedVariable":
        return RuleSignal("automation_defect", 0.6, "An action used a value an earlier action did not capture.")
    if step.execution_type == "browser":
        if error == "TimeoutError":
            if any(code >= 500 for code in step.http_statuses):
                return RuleSignal("application_defect", 0.55, "Browser timed out after a 5xx response.")
            return RuleSignal("automation_defect", 0.5, "A selector was not found in time; locator drift likely.")
        if error == "AssertionError":
            confidence = 0.6 if step.console_errors else 0.5
            return RuleSignal("application_defect", confidence, "The page rendered an unexpected result.")
        return RuleSignal("unknown", 0.2, "Browser failure without a recognised signature.")
    failed = [a for a in step.assertions if not a.passed]
    for outcome in (a for a in failed if a.kind in ("status_code", "status_in")):
        actual, expected = outcome.actual_status or 0, outcome.expected_statuses
        expected_success = any(200 <= code < 300 for code in expected)
        expected_rejection = all(code >= 400 for code in expected)
        if actual == 500:
            return RuleSignal("application_defect", 0.7, "HTTP 500: unhandled server error.")
        if actual in (502, 503, 504):
            return RuleSignal("dependency_issue", 0.6, f"HTTP {actual}: an upstream dependency failed.")
        if expected_rejection and 200 <= actual < 300:
            return RuleSignal("application_defect", 0.75,
                              f"Request that should be rejected ({expected}) was accepted with HTTP {actual}.")
        if expected_success and actual in (401, 403):
            return RuleSignal("configuration_issue", 0.65, f"HTTP {actual}: test credentials or roles are wrong.")
        if expected_success and actual in (404, 409):
            return RuleSignal("test_data_issue", 0.55, f"HTTP {actual}: required test data is missing or exhausted.")
        if expected_success and actual in (400, 422):
            return RuleSignal("automation_defect", 0.45, f"HTTP {actual}: the request may not match the API contract.")
        if expected_rejection and actual >= 400:
            return RuleSignal("application_defect", 0.45, f"Rejected with HTTP {actual} instead of {expected}.")
    if any(a.path_missing for a in failed):
        return RuleSignal("automation_defect", 0.45, "Expected JSON field missing: contract drift or wrong assertion.")
    if any(a.kind == "json_not_exists" for a in failed):
        return RuleSignal("application_defect", 0.65, "Response exposed data that must not be returned.")
    if any(a.kind == "json_equals" for a in failed):
        return RuleSignal("application_defect", 0.6, "Response value differs from the expected business result.")
    return RuleSignal("unknown", 0.2, "Failure evidence does not match any known signature.")