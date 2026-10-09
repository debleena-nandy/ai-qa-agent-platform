"""Validate reviewed API actions against a target's published OpenAPI operations.
Optional deterministic action library for the bundled demo target (RULE_BASED_FALLBACK=true only).
It maps scenario text to validated actions only when the target's OpenAPI document actually exposes every
endpoint used, so it never invents calls against an unknown API. Unmatched scenarios stay blocked and must be
automated by the LLM tool-calling agent, a reviewer (manual actions), or excluded from the plan.
"""
# File: agents/test_design_agent/action_library.py
# Description: Rejects API actions that are not defined by the reviewed target OpenAPI contract.
# Author Name: Debleena Nandy
# Date: 07-10-2026
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from core.models.actions import ApiAssertion, ApiRequestAction, BrowserAction
from core.models.scenario import QAScenario
from tools.api_tools.openapi import OpenApiSpec

IN_STOCK, OUT_OF_STOCK, LAST_UNIT = "toy-001", "toy-002", "toy-003"
UNIT_PRICE = 19.99
DECLINED_CARD = "4000000000000002"

ActionList = List[ApiRequestAction]
BrowserSteps = List[BrowserAction]
Assertions = List[ApiAssertion]
Headers = Dict[str, str]
ApiBuilder = Callable[[QAScenario], ActionList]
UiBuilder = Callable[[QAScenario], BrowserSteps]
ApiRule = Tuple[re.Pattern, ApiBuilder]
UiRule = Tuple[re.Pattern, UiBuilder]
ApiRules = List[ApiRule]
UiRules = List[UiRule]


@dataclass(frozen=True)
class TargetCredentials:
    token: str = "demo-customer-token"
    other_token: str = "demo-other-token"


def _status(code: int) -> ApiAssertion:
    return ApiAssertion(type="status_code", expected=code)


def _eq(path: str, value: Any) -> ApiAssertion:
    return ApiAssertion(type="json_equals", json_path=path, expected=value)


def _pattern(text: str) -> re.Pattern:
    return re.compile(text, re.IGNORECASE)


class DemoShopActionLibrary:
    def __init__(self, credentials: TargetCredentials | None = None):
        self.credentials = credentials or TargetCredentials()

    # ---- helpers -----------------------------------------------------------------------
    def _auth(self, other: bool = False) -> Headers:
        token = self.credentials.other_token if other else self.credentials.token
        return {"Authorization": f"Bearer {token}"}

    def _order(
        self,
        quantity: int = 1,
        product: str = IN_STOCK,
        assertions: Assertions | None = None,
        auth: bool = True,
        capture: Headers | None = None,
        headers: Headers | None = None,
        **extra: Any,
    ) -> ApiRequestAction:
        body = {"product_id": product, "quantity": quantity, **extra}
        return ApiRequestAction(
            method="POST",
            path="/orders",
            json_body=body,
            headers={**(self._auth() if auth else {}), **(headers or {})},
            assertions=assertions or [_status(201)],
            capture=capture or {},
        )

    def _cancel(self, assertions: Assertions) -> ApiRequestAction:
        return ApiRequestAction(method="POST", path="/orders/{{order_id}}/cancel", headers=self._auth(),
                                assertions=assertions)

    def _idempotent(self) -> ActionList:
        key = {"Idempotency-Key": f"qa-{uuid.uuid4()}"}
        return [
            self._order(headers=key, capture={"order_id": "id"}),
            self._order(headers=key, assertions=[_status(201), _eq("id", "{{order_id}}")]),
        ]

    # ---- API rules (first match wins; most specific first) ---------------------------------
    def _rules(self) -> ApiRules:
        created = {"order_id": "id"}
        return [
            (_pattern(r"another customer|another user'?s order|other customer"), lambda s: [
                self._order(capture=created),
                ApiRequestAction(method="GET", path="/orders/{{order_id}}", headers=self._auth(other=True),
                                 assertions=[_status(404),
                                             ApiAssertion(type="json_not_exists", json_path="product_id")]),
            ]),
            (_pattern(r"cancel.*restor|stock.*cancel|reduces stock"), lambda s: [
                self._order(capture=created),
                self._cancel([_status(200), _eq("status", "cancelled")]),
                self._cancel([_status(409)]),
            ]),
            (_pattern(r"\bcancel"), lambda s: [
                self._order(capture=created),
                self._cancel([_status(200), _eq("status", "cancelled")]),
            ]),
            (_pattern(r"duplicate|double-click|twice|idempot"), lambda s: self._idempotent()),
            (_pattern(r"out[- ]of[- ]stock|stock = 0|stock changes to zero"), lambda s: [
                self._order(product=OUT_OF_STOCK, assertions=[_status(409)]),
            ]),
            (_pattern(r"declin"), lambda s: [
                self._order(payment_card=DECLINED_CARD, assertions=[_status(402)]),
            ]),
            (_pattern(r"price .*cannot be changed|tamper|price .*from the client"), lambda s: [
                self._order(unit_price=0.01, assertions=[_status(201), _eq("unit_price", UNIT_PRICE)]),
            ]),
            (_pattern(r"equivalence|partition"), lambda s: [
                self._order(3),
                self._order(-2, assertions=[_status(422)]),
                self._order(9, assertions=[_status(422)]),
            ]),
            (_pattern(r"maximum allowed quantity|maximum \+ 1|at most 5|more than 5|limit of 5"), lambda s: [
                self._order(5, assertions=[_status(201), _eq("quantity", 5)]),
                self._order(6, assertions=[_status(422)]),
            ]),
            (_pattern(r"quantity (?:of [a-z ]+ )?(?:set to |= ?)0\b|quantity = 0"), lambda s: [
                self._order(0, assertions=[_status(422)]),
            ]),
            (_pattern(r"last available unit"), lambda s: [
                self._order(product=LAST_UNIT),
                self._order(product=LAST_UNIT, assertions=[_status(409)]),
            ]),
            (_pattern(r"guest|not logged in|unauthenticated|without a token|must log in"), lambda s: [
                self._order(auth=False, assertions=[_status(401)]),
            ]),
            (_pattern(r"unknown product|does-not-exist"), lambda s: [
                self._order(product="does-not-exist", assertions=[_status(404)]),
            ]),
            (_pattern(r"invalid payload|quantity = -1|missing product id"), lambda s: [
                ApiRequestAction(method="POST", path="/orders", headers=self._auth(), json_body={"quantity": -1},
                                 assertions=[_status(422)]),
            ]),
            (_pattern(r"multiple units"), lambda s: [
                self._order(2, assertions=[_status(201), _eq("quantity", 2)]),
            ]),
            (_pattern(r"total is calculated|order total"), lambda s: [
                self._order(3, assertions=[_status(201), _eq("total", round(UNIT_PRICE * 3, 2))]),
            ]),
            (_pattern(r"create order api returns 201|in-stock .*confirmation|place an order|orders? an? in-stock"),
             lambda s: [
                 self._order(assertions=[_status(201), ApiAssertion(type="json_exists", json_path="id"),
                                         _eq("status", "confirmed")]),
             ]),
        ]

    # ---- UI rules ----------------------------------------------------------------------
    def _ui_rules(self) -> UiRules:
        return [
            (_pattern(r"more than the allowed|quantity 6|error"), lambda s: [
                BrowserAction(type="goto", path="/shop"),
                BrowserAction(type="fill", selector="#quantity", value="6"),
                BrowserAction(type="click", selector="#place-order"),
                BrowserAction(type="expect_text", selector="#result", value="Error"),
            ]),
            (_pattern(r"confirmation|places an order|order"), lambda s: [
                BrowserAction(type="goto", path="/shop"),
                BrowserAction(type="select", selector="#product", value=IN_STOCK),
                BrowserAction(type="fill", selector="#quantity", value="1"),
                BrowserAction(type="click", selector="#place-order"),
                BrowserAction(type="expect_text", selector="#result", value="confirmed"),
            ]),
        ]

    # ---- public --------------------------------------------------------------------------
    def design(self, scenario: QAScenario, spec: OpenApiSpec) -> QAScenario | None:
        text = " ".join([scenario.title, *scenario.steps, scenario.expected_result])
        if scenario.type == "ui":
            if not spec.has("GET", "/shop"):
                return None
            for pattern, build_ui in self._ui_rules():
                if pattern.search(text):
                    return scenario.model_copy(update={"browser_actions": build_ui(scenario),
                                                       "action_source": "rule_based"})
            return None
        for pattern, build_api in self._rules():
            if pattern.search(text):
                actions = build_api(scenario)
                if all(spec.has(action.method, action.path) for action in actions):
                    return scenario.model_copy(update={"api_actions": actions, "action_source": "rule_based"})
                return None
        return None