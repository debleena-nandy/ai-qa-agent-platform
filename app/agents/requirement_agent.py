from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

RULE_KEYWORDS = re.compile(
    r"\b(should|must|shall|can(?:not)?|allow(?:ed|s)?|require[sd]?|only|at least|at most|maximum|minimum)\b",
    re.IGNORECASE,
)
STORY_PATTERN = re.compile(
    r"^\s*as\s+(?:a|an|the)\s+(?P<actor>[a-z][a-z\s'-]*?)\s*,?\s*I\s+(?:want|need|would like|'d like|wish)\s+(?:to\s+)?"
    r"(?P<goal>.+?)(?:\s*,?\s+so\s+that\s+(?P<benefit>.+?))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
CRITERIA_LINE = re.compile(
    r"^\s*(?:[-*•]|\d+[.)]|AC\s*\d+\s*[:.)-]?|AC\s*[:.)-]|given\b|when\b|then\b|and\b)\s*",
    re.IGNORECASE,
)
BENEFIT_START = re.compile(r"^(?:so\s+that|in\s+order)\b", re.IGNORECASE)
CRITERIA_HEADERS = {"acceptance criteria", "acceptance criteria (ac)", "ac"}
OBJECT_STOP_WORDS = {
    "to", "for", "with", "from", "and", "so", "that", "using", "in", "on", "at", "by", "via",
    "into", "of", "when", "if", "which", "without", "per",
}


@dataclass
class RequirementAnalysis:
    title: str
    actors: List[str] = field(default_factory=list)
    goal: str = ""
    business_object: str = ""
    benefit: str = ""
    business_rules: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)


class RequirementAgent:
    """Parses user stories into structured testable requirements."""

    def analyze(self, requirement_text: str) -> RequirementAnalysis:
        normalized = (requirement_text or "").strip()
        if not normalized:
            raise ValueError("Requirement text cannot be empty.")

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        story_line, body = self._split_story(lines)
        match = STORY_PATTERN.search(story_line)

        actor = self._clean(match.group("actor")) if match else ""
        goal = self._clean(match.group("goal")) if match else ""
        benefit = self._clean(match.group("benefit")) if match and match.group("benefit") else ""

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
            ambiguities=self._detect_ambiguities(match is not None, actor, benefit, business_rules, acceptance_criteria),
        )

    def _split_story(self, lines: List[str]) -> Tuple[str, List[str]]:
        # What: Joins a story that wraps over several lines ("As a customer,\nI want to ...").
        # How: A line continues the story only while the previous line does not end a sentence and the line
        #      looks like a continuation. Once the story already matches, a rule-bearing line
        #      ("quantity must be at most 5") starts the body instead of being swallowed into the goal.
        story_end = 1
        matched = False
        for end in range(1, len(lines) + 1):
            line = lines[end - 1]
            if end > 1:
                if (
                    lines[end - 2].endswith((".", "!", "?"))
                    or self._is_header(line)
                    or CRITERIA_LINE.match(line)
                    or not self._is_continuation(line)
                    or (matched and RULE_KEYWORDS.search(line) and not BENEFIT_START.match(line))
                ):
                    break
            if STORY_PATTERN.search(" ".join(lines[:end])):
                story_end = end
                matched = True
        return " ".join(lines[:story_end]), lines[story_end:]

    @staticmethod
    def _is_continuation(line: str) -> bool:
        return line[:1].islower() or bool(re.match(r"^(?:I\s|so\s+that\b|in\s+order\b)", line, re.IGNORECASE))

    @staticmethod
    def _clean(value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", (value or "")).strip(" ,.")

    @staticmethod
    def _is_header(line: str) -> bool:
        return line.lower().strip().rstrip(":").strip() in CRITERIA_HEADERS

    def _extract_title(self, story_line: str) -> str:
        title = story_line.rstrip(".")
        return title if len(title) <= 120 else title[:117].rstrip() + "..."

    def _extract_business_object(self, goal: str) -> str:
        phrases = re.findall(r"(?=\b(?:a|an|the|my|our|their|some)\s+([a-z][a-z\s-]*))", goal, flags=re.IGNORECASE)
        if not phrases:
            words = goal.split()
            return words[-1].lower() if words else ""
        cut: List[str] = []
        for word in phrases[-1].split():
            if word.lower() in OBJECT_STOP_WORDS:
                break
            cut.append(word)
        while len(cut) > 1 and cut[-1].lower().endswith("ly"):
            cut.pop()
        return " ".join(cut).lower().strip()

    def _extract_business_rules(self, lines: List[str]) -> List[str]:
        rules: List[str] = []
        for line in lines:
            if self._is_header(line):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                sentence = CRITERIA_LINE.sub("", sentence).strip()
                if sentence and RULE_KEYWORDS.search(sentence) and sentence not in rules:
                    rules.append(sentence)
        return rules[:10]

    def _extract_acceptance_criteria(self, lines: List[str]) -> List[str]:
        criteria: List[str] = []
        for line in lines:
            if self._is_header(line):
                continue
            if CRITERIA_LINE.match(line):
                cleaned = CRITERIA_LINE.sub("", line).strip()
                if cleaned:
                    criteria.append(cleaned)
        return criteria

    def _detect_ambiguities(
        self,
        is_story_format: bool,
        actor: str,
        benefit: str,
        business_rules: List[str],
        acceptance_criteria: List[str],
    ) -> List[str]:
        ambiguities = []
        if not is_story_format:
            ambiguities.append("Requirement is not in 'As a <actor>, I want <goal>' format; actor and goal may be inaccurate.")
        if not actor:
            ambiguities.append("No actor/persona is identified.")
        if not benefit:
            ambiguities.append("Business value ('so that ...') is not stated.")
        # business_rules already contains rule-bearing acceptance criteria (see analyze()).
        if not business_rules:
            ambiguities.append("No explicit business rules (limits, validations, permissions) are defined.")
        if not acceptance_criteria:
            ambiguities.append("No acceptance criteria are provided; expected results are inferred from the domain.")
        return ambiguities
