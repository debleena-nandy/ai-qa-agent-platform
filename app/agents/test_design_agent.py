from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from app.agents.requirement_agent import RequirementAnalysis
from app.llm.base import StructuredLLM
from app.models.scenario import DomainProfile, Priority, QAScenario, ScenarioSet, ScenarioType

logger = logging.getLogger(__name__)

MIN_LLM_SCENARIOS = 8
# A criterion counts as covered when this share of its key words appears in one scenario.
CRITERION_COVERAGE_RATIO = 0.6
COVERAGE_STOP_WORDS = {
    "a", "an", "the", "is", "are", "be", "can", "cannot", "must", "should", "to", "of", "and",
    "or", "per", "for", "in", "on", "with", "it", "not", "at", "most", "least",
}

SCENARIO_PROMPT = """You are a Senior QA Automation Architect. Write 15-20 concrete, executable test scenarios
for the user story below.

User story: {requirement}
Actor(s): {actors}
Goal: {goal}
Business object: {business_object}
Domain: {domain}
Key concerns to cover: {concerns}
Explicit business rules: {rules}
Explicit acceptance criteria: {criteria}

Rules:
- Be specific to this domain. Name real entities, data values, states and integrations
  (for an order: stock level, cart, quantity, price, payment, shipping address, confirmation).
- NEVER write generic labels such as "Happy path: the user completes the primary action".
- Cover every key concern and every acceptance criterion at least once.
- Mix types: functional, negative, boundary, security, api.
- Each scenario needs preconditions, numbered concrete steps, and ONE verifiable expected_result.
- Use concrete test data in steps (e.g. "quantity = 0", "card number 4000 0000 0000 0002").
- Use ids TC-001, TC-002, ... ; priority is high, medium or low.
"""

# Template format: (type, priority, title, preconditions, steps, expected_result)
# Placeholders: {actor}, {obj}, {goal}
Template = Tuple[ScenarioType, Priority, str, List[str], List[str], str]

DOMAIN_TEMPLATES: Dict[str, List[Template]] = {
    "e-commerce": [
        ("functional", "high", "{actor} orders an in-stock {obj} and receives an order confirmation",
         ["{actor} is logged in", "{obj} has stock >= 1"],
         ["Open the {obj} product page", "Add 1 {obj} to the cart", "Proceed to checkout", "Enter a valid shipping address", "Pay with a valid card"],
         "Order is created with status 'confirmed' and a confirmation number is shown"),
        ("functional", "high", "{actor} orders multiple units and different items in one cart",
         ["{actor} is logged in", "Two products, including {obj}, are in stock"],
         ["Add 2 units of {obj} and 1 other product to the cart", "Complete checkout"],
         "One order is created containing all line items with correct quantities and total"),
        ("functional", "high", "Order total is calculated correctly for {obj} with tax and shipping",
         ["{obj} has a known unit price"],
         ["Add 3 units of {obj} to the cart", "Proceed to checkout and view the summary"],
         "Total equals unit price x 3 + tax + shipping, matching the pricing rules"),
        ("negative", "high", "{actor} cannot order an out-of-stock {obj}",
         ["{obj} has stock = 0"],
         ["Open the {obj} product page", "Try to add it to the cart"],
         "Add to cart is disabled or rejected with an 'Out of stock' message; no order is created"),
        ("negative", "high", "Stock changes to zero while {obj} is in the cart",
         ["{obj} is in the cart", "Stock is set to 0 by another purchase"],
         ["Proceed to checkout and submit payment"],
         "Checkout is blocked with a stock message and the customer is not charged"),
        ("negative", "high", "Order is not placed when payment is declined",
         ["{obj} is in the cart"],
         ["Checkout with a declined test card (e.g. 4000 0000 0000 0002)"],
         "Payment error is shown, order stays unpaid/not created, and stock is not reduced"),
        ("negative", "medium", "Checkout is blocked for an invalid shipping address",
         ["{obj} is in the cart"],
         ["Enter a shipping address with an empty postcode", "Submit the order"],
         "Field-level validation error is shown and the order is not submitted"),
        ("negative", "medium", "{actor} cannot check out with an empty cart",
         ["Cart is empty"],
         ["Open the cart page", "Try to start checkout"],
         "Checkout button is disabled or a 'Your cart is empty' message is shown"),
        ("negative", "medium", "Invalid or expired coupon is rejected at checkout",
         ["{obj} is in the cart"],
         ["Apply coupon code 'EXPIRED10'"],
         "Coupon is rejected with a clear message and the total is unchanged"),
        ("boundary", "medium", "Quantity of {obj} set to 0 is rejected",
         ["{obj} is in stock"],
         ["Set quantity = 0 in the cart"],
         "Quantity is rejected or the line item is removed; checkout total is not negative"),
        ("boundary", "medium", "Maximum allowed quantity of {obj} per order is enforced",
         ["A maximum quantity per order is configured"],
         ["Set quantity to the maximum", "Set quantity to maximum + 1"],
         "Maximum is accepted; maximum + 1 is rejected with a limit message"),
        ("boundary", "low", "Ordering the last available unit of {obj}",
         ["{obj} has stock = 1"],
         ["Order 1 unit of {obj}"],
         "Order succeeds and the {obj} then shows as out of stock"),
        ("security", "high", "Guest {actor} must log in or use guest checkout before paying",
         ["User is not logged in", "{obj} is in the cart"],
         ["Proceed to checkout"],
         "User is redirected to login or guest checkout; no order is created anonymously"),
        ("security", "high", "{actor} cannot view or modify another customer's order",
         ["Order A belongs to another customer"],
         ["Request GET /orders/<order A id> with the {actor}'s token"],
         "API returns 403 or 404 and no order data is exposed"),
        ("security", "high", "Price of {obj} cannot be changed from the client",
         ["{obj} is in the cart"],
         ["Intercept the checkout request and change the unit price to 0.01", "Submit the order"],
         "Server recalculates the price from the catalogue and rejects or corrects the tampered value"),
        ("api", "high", "Double-clicking 'Place order' does not create duplicate orders",
         ["{obj} is in the cart"],
         ["Send the same create-order request twice within 1 second"],
         "Only one order and one payment charge are created"),
        ("api", "medium", "Create order API returns 201 with order id, items and total",
         ["Valid auth token", "{obj} is in stock"],
         ["POST /orders with a valid {obj} line item"],
         "Response is 201 and the body contains order id, status, line items and total matching the request"),
        ("api", "medium", "Create order API rejects an invalid payload with 400",
         ["Valid auth token"],
         ["POST /orders with quantity = -1 and a missing product id"],
         "Response is 400 with field-level validation errors"),
        ("functional", "medium", "{actor} can cancel the {obj} order before it ships",
         ["An order for {obj} exists with status 'confirmed'"],
         ["Open order history", "Cancel the order"],
         "Order status becomes 'cancelled', payment is refunded and stock is restored"),
    ],
    "authentication": [
        ("functional", "high", "{actor} signs in with valid credentials and reaches the {obj} area",
         ["An active {actor} account exists"],
         ["Open the sign-in page", "Enter a valid email and password", "Submit"],
         "{actor} is signed in and redirected to the {obj} page"),
        ("negative", "high", "Sign-in fails with a wrong password",
         ["An active {actor} account exists"],
         ["Enter a valid email and a wrong password", "Submit"],
         "A generic 'Invalid email or password' message is shown; no session is created"),
        ("negative", "medium", "Sign-in fails for an unregistered email",
         [], ["Enter an unregistered email and any password", "Submit"],
         "The same generic error is shown, so account existence is not revealed"),
        ("negative", "medium", "Sign-in is blocked when required fields are empty",
         [], ["Leave email and password empty", "Submit"],
         "Required-field errors are shown and no request is sent"),
        ("security", "high", "Account is locked after repeated failed attempts",
         ["A lockout threshold is configured"],
         ["Enter a wrong password until the threshold is reached", "Try the correct password"],
         "Account is locked and access is denied even with the correct password"),
        ("security", "high", "Disabled {actor} account cannot sign in",
         ["The {actor} account is disabled"],
         ["Sign in with the correct credentials"],
         "Access is denied with an 'account disabled' message"),
        ("security", "high", "Password is never returned or logged in plain text",
         [], ["Sign in", "Inspect the API response and application logs"],
         "Password does not appear in responses, URLs or logs"),
        ("security", "medium", "Session expires after the idle timeout",
         ["{actor} is signed in"],
         ["Stay idle longer than the session timeout", "Open the {obj} page"],
         "{actor} is redirected to sign-in"),
        ("security", "medium", "Signing out invalidates the session token",
         ["{actor} is signed in"],
         ["Sign out", "Reuse the old token to call a protected API"],
         "The API returns 401"),
        ("boundary", "low", "Email and password fields handle maximum length input",
         [], ["Enter a 255-character email and a 129-character password", "Submit"],
         "Input is validated gracefully with no server error"),
        ("api", "high", "Login API returns a token for valid credentials",
         ["An active account exists"],
         ["POST /auth/login with valid credentials"],
         "Response is 200 with an access token and expiry; no password in the body"),
        ("api", "medium", "Login API rejects a malformed payload",
         [], ["POST /auth/login with a missing password field"],
         "Response is 400/422 with a validation error"),
    ],
    "account-management": [
        ("functional", "high", "{actor} updates the {obj} with valid data",
         ["{actor} is signed in"],
         ["Open the {obj} page", "Change a field to a valid value", "Save"],
         "A success message is shown and the change persists after reload"),
        ("negative", "high", "Saving the {obj} with an empty required field is blocked",
         ["{actor} is signed in"],
         ["Clear a required field", "Save"],
         "A field-level error is shown and nothing is saved"),
        ("negative", "medium", "Invalid format values are rejected for the {obj}",
         ["{actor} is signed in"],
         ["Enter an invalid email/phone format", "Save"],
         "A format validation error is shown"),
        ("negative", "medium", "Unsupported file type or size is rejected for the {obj}",
         ["{actor} is signed in"],
         ["Upload a .exe file", "Upload a file larger than the size limit"],
         "Both uploads are rejected with clear messages"),
        ("boundary", "medium", "Field length limits for the {obj} are enforced",
         ["{actor} is signed in"],
         ["Enter the maximum allowed characters", "Enter maximum + 1 characters"],
         "The maximum is accepted; maximum + 1 is rejected"),
        ("security", "high", "{actor} cannot edit another user's {obj}",
         ["Another user's id is known"],
         ["Send an update request for the other user's id"],
         "Response is 403 and no data is changed"),
        ("security", "medium", "Unauthenticated users cannot access the {obj}",
         ["User is signed out"],
         ["Open the {obj} URL directly"],
         "User is redirected to sign-in"),
        ("security", "medium", "Script input in {obj} fields is not executed",
         ["{actor} is signed in"],
         ["Enter <script>alert(1)</script> in a text field", "Save and reload"],
         "The value is escaped and shown as text; no script runs"),
        ("api", "medium", "Update API returns the saved {obj} data",
         ["Valid auth token"],
         ["PUT the {obj} with valid data", "GET the {obj}"],
         "PUT returns 200 and GET returns the updated values"),
        ("functional", "low", "Changes to the {obj} are recorded in the audit trail",
         ["Audit logging is enabled"],
         ["Update the {obj}"],
         "An audit entry with user, time and changed fields is created"),
    ],
    "general": [
        ("functional", "high", "{actor} can {goal} with valid data",
         ["{actor} has the required access"],
         ["Open the feature to {goal}", "Provide valid data", "Submit"],
         "The {obj} is processed successfully and a confirmation is shown"),
        ("functional", "medium", "Result of the '{goal}' action persists",
         ["{actor} has completed the action"],
         ["Reload the page or sign in again", "Open the {obj}"],
         "The saved {obj} data is shown unchanged"),
        ("negative", "high", "Invalid data for the {obj} is rejected",
         ["{actor} has the required access"],
         ["Provide invalid or malformed values for the {obj}", "Submit"],
         "Clear validation errors are shown and nothing is saved"),
        ("negative", "medium", "Required data for the {obj} cannot be skipped",
         [], ["Leave required fields empty", "Submit"],
         "Required-field errors are shown"),
        ("negative", "medium", "Downstream failure while processing the {obj} is handled",
         ["A dependent service is unavailable"],
         ["Try to {goal}"],
         "A friendly error is shown, no partial data is saved, and the error is logged"),
        ("boundary", "medium", "Minimum and maximum values for the {obj} are enforced",
         [], ["Submit minimum, maximum, and maximum + 1 values"],
         "Minimum and maximum are accepted; out-of-range values are rejected"),
        ("security", "high", "Users without permission cannot {goal}",
         ["User lacks the {actor} role"],
         ["Try to {goal}"],
         "Access is denied with 403 and nothing is changed"),
        ("security", "medium", "Unauthenticated requests to act on the {obj} are rejected",
         [], ["Call the API without a token"],
         "Response is 401"),
        ("api", "medium", "API for the {obj} returns the documented success contract",
         ["Valid auth token"],
         ["Send a valid request for the {obj}"],
         "Status code and response body match the API contract"),
        ("api", "medium", "API for the {obj} rejects an invalid payload",
         ["Valid auth token"],
         ["Send a payload with missing and wrongly typed fields"],
         "Response is 400/422 with field-level errors"),
    ],
}


# What: Generates concrete, domain-aware QA scenarios.
# Why: Generic category labels cannot be executed or reviewed; scenarios need real entities, data, steps, and outcomes.
# How: Uses the LLM with a strict prompt when available; otherwise renders domain templates
#      with the extracted actor/object and adds one scenario per explicit acceptance criterion.
class TestDesignAgent:
    """Converts requirement analysis into concrete test scenarios."""

    __test__ = False  # stops pytest from treating this class as a test class

    def __init__(self, llm_client: Optional[StructuredLLM] = None):
        self.llm = llm_client

    def generate_scenarios(
        self,
        analysis: RequirementAnalysis,
        domain: DomainProfile,
        requirement: str,
        use_llm: bool = True,
    ) -> Tuple[List[QAScenario], str]:
        # use_llm=False lets the workflow skip a second slow call when the LLM already failed for the domain.
        llm = self.llm
        if llm is not None and use_llm:
            try:
                scenarios = self._generate_with_llm(analysis, domain, requirement, llm)
                if len(scenarios) >= MIN_LLM_SCENARIOS:
                    scenarios.extend(self._missing_criteria_scenarios(analysis, scenarios))
                    return self._renumber(scenarios), "llm"
                logger.warning("LLM returned only %d scenarios, using rule-based fallback", len(scenarios))
            except Exception as exc:  # network, timeout, invalid JSON
                logger.warning("LLM scenario generation failed, using rule-based fallback: %s", exc)
        return self._rule_based(analysis, domain), "rule_based"

    def _generate_with_llm(
        self, analysis: RequirementAnalysis, domain: DomainProfile, requirement: str, llm: StructuredLLM
    ) -> List[QAScenario]:
        prompt = SCENARIO_PROMPT.format(
            requirement=requirement,
            actors=", ".join(analysis.actors) or "not specified",
            goal=analysis.goal or "not specified",
            business_object=domain.business_object,
            domain=domain.domain,
            concerns="; ".join(domain.key_concerns) or "infer from the story",
            rules="; ".join(analysis.business_rules) or "none stated",
            criteria="; ".join(analysis.acceptance_criteria) or "none stated",
        )
        scenarios = llm.generate_structured(prompt, ScenarioSet).scenarios
        # Drop incomplete items: a scenario without steps or an expected result is not executable.
        return [s for s in scenarios if s.steps and s.expected_result.strip()]

    @staticmethod
    def _key_words(text: str) -> Set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in COVERAGE_STOP_WORDS}

    def _missing_criteria_scenarios(
        self, analysis: RequirementAnalysis, scenarios: List[QAScenario]
    ) -> List[QAScenario]:
        # What: Adds a scenario for every acceptance criterion the LLM did not cover.
        # Why: The prompt asks for full coverage, but LLM output is not guaranteed; criteria are the story's contract.
        # How: Word overlap between the criterion and each scenario's title + expected result.
        scenario_words = [self._key_words(f"{s.title} {s.expected_result}") for s in scenarios]
        missing = []
        for criterion in analysis.acceptance_criteria:
            words = self._key_words(criterion)
            if words and not any(len(words & sw) / len(words) >= CRITERION_COVERAGE_RATIO for sw in scenario_words):
                missing.append(criterion)
        if missing:
            logger.info("LLM missed %d acceptance criteria; adding rule-based scenarios", len(missing))
        return [self._criterion_scenario(analysis, criterion) for criterion in missing]

    def _criterion_scenario(self, analysis: RequirementAnalysis, criterion: str) -> QAScenario:
        actor = analysis.actors[0] if analysis.actors else "user"
        goal = analysis.goal or "complete the action"
        return QAScenario(
            id="", title=f"Acceptance criterion: {criterion}", type="functional", priority="high",
            preconditions=[f"{self._capitalize(actor)} has access to the feature"],
            steps=[
                f"Prepare test data that exercises: {criterion}",
                f"As the {actor}, {goal}",
                "Compare the observed behaviour with the acceptance criterion",
            ],
            expected_result=criterion,
        )

    def _rule_based(self, analysis: RequirementAnalysis, domain: DomainProfile) -> List[QAScenario]:
        values = {
            "actor": analysis.actors[0] if analysis.actors else "user",
            "obj": domain.business_object or analysis.business_object or "item",
            "goal": analysis.goal or "complete the action",
        }
        # Explicit acceptance criteria come first: they are the contract of the story.
        scenarios = [self._criterion_scenario(analysis, c) for c in analysis.acceptance_criteria]

        templates = DOMAIN_TEMPLATES.get(domain.domain, DOMAIN_TEMPLATES["general"])
        for scenario_type, priority, title, pre, steps, expected in templates:
            scenarios.append(QAScenario(
                id="",
                title=self._capitalize(self._fill(title, values)),
                type=scenario_type,
                priority=priority,
                preconditions=[self._capitalize(self._fill(p, values)) for p in pre],
                steps=[self._capitalize(self._fill(s, values)) for s in steps],
                expected_result=self._capitalize(self._fill(expected, values)),
            ))
        return self._renumber(scenarios)

    @staticmethod
    def _fill(text: str, values: Dict[str, str]) -> str:
        return text.format(**values)

    @staticmethod
    def _capitalize(text: str) -> str:
        return text[:1].upper() + text[1:] if text else text

    @staticmethod
    def _renumber(scenarios: List[QAScenario]) -> List[QAScenario]:
        for index, scenario in enumerate(scenarios, start=1):
            scenario.id = f"TC-{index:03d}"
        return scenarios
