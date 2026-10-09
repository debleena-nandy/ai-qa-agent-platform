# File: agents/requirement_agent/rules.py
# Description: Detects and normalizes the requirement agent rules.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from agents.requirement_agent.models import RequirementAnalysis
from core.models.scenario import DomainProfile

# Optional deterministic fallback for the Requirement Analysis Agent.
# Loaded only when RULE_BASED_FALLBACK=true; the LLM path in agent.py is the default.

Lines = List[str]
StorySplit = Tuple[str, List[str]]
AnalysisPair = Tuple[RequirementAnalysis, DomainProfile]
CatalogEntry = Tuple[str, List[str], List[str]]

RULE_KEYWORDS = re.compile(
    r"\b(should|must|shall|can(?:not)?|allow(?:ed|s)?|require[sd]?|only|at least|at most|maximum|minimum)\b",
    re.IGNORECASE,
)
# Groups: 1 = actor, 2 = goal, 3 = benefit (optional)
STORY_PATTERN = re.compile(
    r"^\s*as\s+(?:a|an|the)\s+([a-z][a-z\s'-]*?)\s*,?\s*I\s+(?:want|need|would like|'d like|wish)\s+(?:to\s+)?"
    r"(.+?)(?:\s*,?\s+so\s+that\s+(.+?))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
CRITERIA_LINE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)]|AC\s*\d+\s*[:.)-]?|AC\s*[:.)-]|given\b|when\b|then\b|and\b|but\b)\s*",
    re.IGNORECASE,
)
GHERKIN_START = re.compile(r"^\s*given\b", re.IGNORECASE)
GHERKIN_CONTINUE = re.compile(r"^\s*(?:when|then|and|but)\b", re.IGNORECASE)
BENEFIT_START = re.compile(r"^(?:so\s+that|in\s+order)\b", re.IGNORECASE)
LIMIT = re.compile(
    r"\b(at most|at least|maximum(?: of)?|minimum(?: of)?|up to|no more than|no less than)\s+(\d+)",
    re.IGNORECASE,
)
UPPER_LIMITS = {"at most", "maximum", "maximum of", "up to", "no more than"}
NEGATIVE_RULE = re.compile(r"\b(cannot|can not|must not|should not|shall not|only)\b", re.IGNORECASE)
NUMERIC_HINT = re.compile(r"\b(quantity|amount|price|age|length|size|count|number|limit|units?)\b", re.IGNORECASE)
ERROR_HINT = re.compile(r"\b(error|fail|invalid|reject|declin|denied|message)\w*", re.IGNORECASE)
PERMISSION_HINT = re.compile(r"\b(only|permission|role|admin|owner|authori[sz]ed|logged in|signed in)\b",
                             re.IGNORECASE)
OUTCOME_HINT = re.compile(r"\b(confirm|notif|email|message|shown|display|receive)\w*", re.IGNORECASE)

CRITERIA_HEADERS = {"acceptance criteria", "acceptance criteria (ac)", "ac"}
OBJECT_STOP_WORDS = {
    "to", "for", "with", "from", "and", "so", "that", "using", "in", "on", "at", "by", "via",
    "into", "of", "when", "if", "which", "without", "per",
}


class RuleBasedRequirementParser:
    """Parses user stories with regular expressions (fallback for the LLM Requirement Analysis Agent)."""

    def analyze(self, requirement_text: str) -> RequirementAnalysis:
        normalized = (requirement_text or "").strip()
        if not normalized:
            raise ValueError("Requirement text cannot be empty.")
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        story_line, body = self._split_story(lines)
        match = STORY_PATTERN.search(story_line)

        actor = self._clean(match.group(1)) if match else ""
        goal = self._clean(match.group(2)) if match else ""
        benefit = self._clean(match.group(3)) if match and match.group(3) else ""

        acceptance_criteria = self._extract_acceptance_criteria(body)
        rule_lines = [line for line in body if not CRITERIA_LINE.match(line)]
        business_rule_lines = [*rule_lines, *acceptance_criteria]
        if not match:
            business_rule_lines.insert(0, story_line)
        business_rules = self._extract_business_rules(business_rule_lines)

        return RequirementAnalysis(
            title=self._extract_title(story_line),
            actors=[actor] if actor else [],
            goal=goal,
            business_object=self._extract_business_object(goal) if goal else "",
            benefit=benefit,
            business_rules=business_rules,
            acceptance_criteria=acceptance_criteria,
            ambiguities=self._detect_ambiguities(match is not None, actor, benefit, business_rules,
                                                 acceptance_criteria),
            missing_requirements=self._detect_missing(normalized, goal),
            testable_conditions=self._testable_conditions(business_rules, acceptance_criteria),
        )

    # ---- story splitting --------------------------------------------------------------
    def _split_story(self, lines: Lines) -> StorySplit:
        """Joins a story that wraps over several lines; a rule-bearing line starts the body instead."""
        story_end = 1
        matched = False
        for end in range(1, len(lines) + 1):
            line = lines[end - 1]
            if end > 1:
                previous_ends_sentence = lines[end - 2].endswith((".", "!", "?"))
                starts_rule = matched and RULE_KEYWORDS.search(line) and not BENEFIT_START.match(line)
                if (previous_ends_sentence or self._is_header(line) or CRITERIA_LINE.match(line)
                        or not self._is_continuation(line) or starts_rule):
                    break
            if STORY_PATTERN.search(" ".join(lines[:end])):
                story_end = end
                matched = True
        return " ".join(lines[:story_end]), lines[story_end:]

    @staticmethod
    def _is_continuation(line: str) -> bool:
        return line[:1].islower() or bool(re.match(r"^(?:I\s|so\s+that\b|in\s+order\b)", line, re.IGNORECASE))

    @staticmethod
    def _clean(value: str | None) -> str:
        return re.sub(r"\s+", " ", (value or "")).strip(" ,.")

    @staticmethod
    def _is_header(line: str) -> bool:
        return line.lower().strip().rstrip(":").strip() in CRITERIA_HEADERS

    @staticmethod
    def _extract_title(story_line: str) -> str:
        title = story_line.rstrip(".")
        return title if len(title) <= 120 else title[:117].rstrip() + "..."

    @staticmethod
    def _extract_business_object(goal: str) -> str:
        phrases = re.findall(r"(?=\b(?:a|an|the|my|our|their|some)\s+([a-z][a-z\s-]*))", goal, flags=re.IGNORECASE)
        if not phrases:
            words = goal.split()
            return words[-1].lower() if words else ""
        cut: Lines = []
        for word in phrases[-1].split():
            if word.lower() in OBJECT_STOP_WORDS:
                break
            cut.append(word)
        while len(cut) > 1 and cut[-1].lower().endswith("ly"):
            cut.pop()
        return " ".join(cut).lower().strip()

    # ---- rules and acceptance criteria -------------------------------------------------
    def _extract_business_rules(self, lines: Lines) -> Lines:
        rules: Lines = []
        for line in lines:
            if self._is_header(line):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                sentence = CRITERIA_LINE.sub("", sentence).strip()
                if sentence and RULE_KEYWORDS.search(sentence) and sentence not in rules:
                    rules.append(sentence)
        return rules[:10]

    def _extract_acceptance_criteria(self, lines: Lines) -> Lines:
        """A Given/When/Then/And block is ONE criterion; bullets and numbered lines are one each."""
        criteria: Lines = []
        in_gherkin = False
        for line in lines:
            if self._is_header(line):
                in_gherkin = False
                continue
            if GHERKIN_START.match(line):
                criteria.append(line.strip())
                in_gherkin = True
                continue
            if in_gherkin and GHERKIN_CONTINUE.match(line):
                criteria[-1] = f"{criteria[-1]} {line.strip()}"
                continue
            in_gherkin = False
            if CRITERIA_LINE.match(line):
                cleaned = CRITERIA_LINE.sub("", line).strip()
                if cleaned:
                    criteria.append(cleaned)
        return criteria

    # ---- QA insights -------------------------------------------------------------------
    @staticmethod
    def _detect_ambiguities(is_story_format: bool, actor: str, benefit: str, business_rules: Lines,
                            acceptance_criteria: Lines) -> Lines:
        ambiguities = []
        if not is_story_format:
            ambiguities.append("Requirement is not in 'As a <actor>, I want <goal>' format; "
                               "actor and goal may be inaccurate.")
        if not actor:
            ambiguities.append("No actor/persona is identified.")
        if not benefit:
            ambiguities.append("Business value ('so that ...') is not stated.")
        if not business_rules:
            ambiguities.append("No explicit business rules (limits, validations, permissions) are defined.")
        if not acceptance_criteria:
            ambiguities.append("No acceptance criteria are provided; expected results are inferred from the domain.")
        return ambiguities

    @staticmethod
    def _detect_missing(text: str, goal: str) -> Lines:
        missing = []
        if NUMERIC_HINT.search(text) and not LIMIT.search(text):
            missing.append("A numeric field is mentioned but no minimum/maximum limit is specified.")
        if not ERROR_HINT.search(text):
            missing.append("Expected behaviour for invalid input or failures (error messages, rejection) "
                           "is not specified.")
        if not PERMISSION_HINT.search(text):
            missing.append("Who is allowed to perform the action (authentication/authorization) is not specified.")
        if goal and not OUTCOME_HINT.search(text):
            missing.append("The observable outcome for the actor (confirmation, notification, displayed result) "
                           "is not specified.")
        return missing

    @staticmethod
    def _testable_conditions(rules: Lines, criteria: Lines) -> Lines:
        conditions: Lines = []
        for text in [*criteria, *rules]:
            limits = LIMIT.findall(text)
            for kind, number in limits:
                value = int(number)
                if kind.lower() in UPPER_LIMITS:
                    conditions.append(f"Boundary: {value} is accepted and {value + 1} is rejected ({text}).")
                else:
                    conditions.append(f"Boundary: {value} is accepted and {max(value - 1, 0)} is rejected ({text}).")
            if NEGATIVE_RULE.search(text):
                conditions.append(f"Negative: the disallowed case is rejected with a clear message ({text}).")
            elif not limits:
                conditions.append(f"Positive: {text}.")
        unique: Lines = []
        for condition in conditions:
            if condition not in unique:
                unique.append(condition)
        return unique[:15]


# ---- domain detection (fallback for the LLM domain field) ---------------------------------
GENERAL_CONCERNS = ["valid input", "invalid input", "boundaries", "permissions", "error handling"]

DOMAIN_CATALOG: List[CatalogEntry] = [
    (
        "e-commerce",
        ["order", "buy", "purchase", "checkout", "cart", "basket", "payment", "pay", "paid", "shop", "shopping",
         "shipping", "delivery"],
        ["product availability and inventory", "cart and quantity", "pricing and discounts", "payment",
         "shipping address", "order confirmation", "duplicate submission", "order cancellation"],
    ),
    (
        "authentication",
        ["log in", "login", "sign in", "sign-in", "signin", "log out", "logout", "password", "authenticate",
         "credentials", "2fa", "otp"],
        ["valid credentials", "invalid credentials", "account lockout", "session management", "password reset",
         "brute-force protection", "secure transport"],
    ),
    (
        "account-management",
        ["profile", "account settings", "update my", "edit my", "personal details", "preferences", "photo",
         "avatar"],
        ["required fields", "field format validation", "data persistence", "authorization to edit", "audit trail"],
    ),
]
CATALOG_CONCERNS: Dict[str, Lines] = {name: concerns for name, _, concerns in DOMAIN_CATALOG}


def _keyword_pattern(keyword: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(keyword)}(?:s|es|d|ed|ing)?\b")


COMPILED_CATALOG = [
    (domain, [_keyword_pattern(keyword) for keyword in keywords], concerns)
    for domain, keywords, concerns in DOMAIN_CATALOG
]


def detect_domain(analysis: RequirementAnalysis, requirement: str) -> DomainProfile:
    text = requirement.lower()
    best_domain, best_concerns, best_score = "general", GENERAL_CONCERNS, 0
    for domain, patterns, concerns in COMPILED_CATALOG:
        score = sum(1 for pattern in patterns if pattern.search(text))
        if score > best_score:
            best_domain, best_concerns, best_score = domain, concerns, score
    return DomainProfile(domain=best_domain, business_object=analysis.business_object or "item",
                         key_concerns=list(best_concerns))


def rule_based_analysis(text: str) -> AnalysisPair:
    """Entry point used by RequirementAgent when RULE_BASED_FALLBACK=true."""
    analysis = RuleBasedRequirementParser().analyze(text)
    return analysis, detect_domain(analysis, text)