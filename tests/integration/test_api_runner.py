# File: tests/integration/test_api_runner.py
# Description: Tests API scenario execution against a local HTTP server.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Iterator

import pytest

from agents.execution_agent.agent import ExecutionAgent
from agents.test_planning_agent.agent import TestPlanningAgent
from core.models.actions import ApiAssertion, ApiRequestAction, HttpMethod
from core.models.scenario import QAScenario
from demo_target.scenarios import demo_scenarios
from tools.api_tools.runner import ApiScenarioRunner

LOOPBACK = {"127.0.0.1"}


def _runner(base_url: str) -> ApiScenarioRunner:
    return ApiScenarioRunner(base_url, LOOPBACK, True, timeout_seconds=5)


def _scenario(sid: str, method: HttpMethod, path: str, *assertions: ApiAssertion, body: dict | None = None) -> QAScenario:
    return QAScenario(
        id=sid, title=f"{method} {path}", type="api", steps=[f"{method} {path}"], expected_result="Checked",
        api_actions=[ApiRequestAction(method=method, path=path, json_body=body, assertions=list(assertions))],
    )


class DemoHandler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/orders":
            length = int(self.headers.get("Content-Length", "0"))
            request_body = json.loads(self.rfile.read(length))
            self._json(201, {"id": "order-1", "quantity": request_body["quantity"]})
            return
        self.send_error(404)

    def do_GET(self) -> None:
        if self.path == "/openapi.json":
            self._json(200, {"openapi": "3.0.0", "paths": {"/orders": {"post": {}}, "/redirect": {"get": {}}}})
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.com/")
            self.end_headers()
            return
        self.send_error(404)

    def log_message(self, _format: str, *args: object) -> None:
        pass


@pytest.fixture
def local_target() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), DemoHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_approved_api_plan_executes_against_controlled_demo(demo_target: str) -> None:
    scenario = demo_scenarios()[0]  # POST /orders qty 2 -> 201, status confirmed
    plan = TestPlanningAgent(None, demo_target, False).create_plan([scenario])
    assert plan.scenarios[0].eligibility == "eligible"
    approved_plan = plan.model_copy(update={"approval_status": "approved"})
    runner = _runner(demo_target)
    try:
        execution = ExecutionAgent(runner, None).execute(approved_plan, [scenario])
    finally:
        runner.close()
    assert execution.total_scenarios == 1
    assert execution.passed == 1, execution.steps[0].details


def test_unapproved_plan_is_never_executed(demo_target: str) -> None:
    scenario = demo_scenarios()[0]
    plan = TestPlanningAgent(None, demo_target, False).create_plan([scenario])
    runner = _runner(demo_target)
    try:
        execution = ExecutionAgent(runner, None).execute(plan, [scenario])
    finally:
        runner.close()
    assert execution.not_executed == 1
    assert execution.passed == 0


def test_api_runner_executes_request_and_checks_response(local_target: str) -> None:
    scenario = _scenario(
        "TC-API-001", "POST", "/orders",
        ApiAssertion(type="status_code", expected=201),
        ApiAssertion(type="json_equals", json_path="id", expected="order-1"),
        ApiAssertion(type="json_equals", json_path="quantity", expected=2),
        body={"quantity": 2},
    )
    runner = _runner(local_target)
    try:
        result = runner.execute(scenario)
    finally:
        runner.close()
    assert result.passed == 1
    assert result.steps[0].evidence


def test_api_runner_fails_mismatched_assertion_without_leaking_body(local_target: str) -> None:
    scenario = _scenario(
        "TC-API-002", "POST", "/orders",
        ApiAssertion(type="json_equals", json_path="id", expected="secret-value"),
        body={"quantity": 2},
    )
    runner = _runner(local_target)
    try:
        result = runner.execute(scenario)
    finally:
        runner.close()
    assert result.failed == 1
    assert "secret-value" not in " ".join(result.steps[0].evidence + [result.steps[0].details or ""])


def test_api_runner_does_not_follow_redirects(local_target: str) -> None:
    scenario = _scenario("TC-API-003", "GET", "/redirect", ApiAssertion(type="status_code", expected=200))
    runner = _runner(local_target)
    try:
        result = runner.execute(scenario)
    finally:
        runner.close()
    assert result.failed == 1
    assert result.steps[0].error_type == "RedirectNotAllowed" or \
        "RedirectNotAllowed" in " ".join(result.steps[0].evidence)


def test_api_runner_rejects_host_outside_allow_list() -> None:
    with pytest.raises(ValueError):
        ApiScenarioRunner("http://example.com", LOOPBACK, False)


@pytest.mark.parametrize("url", ["https://127.0.0.1", "http://127.0.0.1/api", "http://user:pw@127.0.0.1"])
def test_api_runner_rejects_non_plain_origins(url: str) -> None:
    with pytest.raises(ValueError, match="plain HTTP origin"):
        ApiScenarioRunner(url, LOOPBACK, True)