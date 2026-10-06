from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from app.agents.requirement_agent import RequirementAnalysis
from app.llm.base import StructuredLLM, is_llm_unavailable
from app.models.scenario import DomainProfile

logger = logging.getLogger(__name__)

GENERAL_CONCERNS = ["valid input", "invalid input", "boundaries", "permissions", "error handling"]

DOMAIN_CATALOG: List[Tuple[str, List[str], List[str]]] = [
    (
        "e-commerce",
        ["order", "buy", "purchase", "checkout", "cart", "basket", "payment", "pay", "paid", "shop", "shopping", "shipping", "delivery"],
        ["product availability and inventory", "cart and quantity", "pricing and discounts",
         "payment", "shipping address", "order confirmation", "duplicate submission", "order cancellation"],
    ),
    (
        "authentication",
        ["log in", "login", "sign in", "sign-in", "signin", "log out", "logout", "password", "authenticate", "credentials", "2fa", "otp"],
        ["valid credentials", "invalid credentials", "account lockout", "session management",
         "password reset", "brute-force protection", "secure transport"],
    ),
    (
        "account-management",
        ["profile", "account settings", "update my", "edit my", "personal details", "preferences", "photo", "avatar"],
        ["required fields", "field format validation", "data persistence", "authorization to edit", "audit trail"],
    ),
]
CATALOG_CONCERNS: Dict[str, List[str]] = {name: concerns for name, _, concerns in DOMAIN_CATALOG}

DOMAIN_ALIASES: Dict[str, str] = {
    "ecommerce": "e-commerce", "e commerce": "e-commerce", "online shopping": "e-commerce",
    "shopping": "e-commerce", "retail": "e-commerce", "orders": "e-commerce", "ordering": "e-commerce",
    "auth": "authentication", "login": "authentication", "identity": "authentication",
    "user authentication": "authentication",
    "account management": "account-management", "user management": "account-management",
    "profile management": "account-management", "user profile": "account-management",
}

DOMAIN_PROMPT = """You are a senior QA analyst. Identify the business domain of this user story.

User story: {requirement}
Actor: {actors}
Goal: {goal}

Return:
- domain: short business domain name (e.g. e-commerce, authentication, booking, payments, account-management)
- business_object: the main thing the actor acts on, in 1-3 words
- key_concerns: 5-10 areas a QA analyst must test for this story (data, states, integrations, limits, permissions)
"""


def normalize_domain(name: str) -> str:
    key = re.sub(r"[_\s]+", " ", (name or "").strip().lower())
    if key.replace(" ", "-") in CATALOG_CONCERNS:
        return key.replace(" ", "-")
    return DOMAIN_ALIASES.get(key, key.replace(" ", "-") or "general")


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(keyword)}(?:s|es|d|ed|ing)?\b")


COMPILED_CATALOG = [
    (domain, [_keyword_pattern(kw) for kw in keywords], concerns) for domain, keywords, concerns in DOMAIN_CATALOG
]


class DomainAgent:
    def __init__(self, llm_client: Optional[StructuredLLM] = None):
        self.llm: Optional[StructuredLLM] = llm_client
        # True only when the last LLM call failed because the server was unreachable or timed out.
        self.llm_unavailable = False

    def extract(self, analysis: RequirementAnalysis, requirement: str) -> Tuple[DomainProfile, str]:
        self.llm_unavailable = False
        llm = self.llm  # local variable: same narrowing pattern as TestDesignAgent
        if llm is not None:
            try:
                prompt = DOMAIN_PROMPT.format(
                    requirement=requirement,
                    actors=", ".join(analysis.actors) or "not specified",
                    goal=analysis.goal or "not specified",
                )
                profile = llm.generate_structured(prompt, DomainProfile)
                profile.domain = normalize_domain(profile.domain)
                if not profile.business_object:
                    profile.business_object = analysis.business_object or "item"
                if not profile.key_concerns:
                    profile.key_concerns = list(CATALOG_CONCERNS.get(profile.domain, GENERAL_CONCERNS))
                return profile, "llm"
            except Exception as exc:  # network, timeout, invalid JSON
                self.llm_unavailable = is_llm_unavailable(exc)
                logger.warning("LLM domain extraction failed, using rule-based fallback: %s", exc)
        return self._rule_based(analysis, requirement), "rule_based"

    def _rule_based(self, analysis: RequirementAnalysis, requirement: str) -> DomainProfile:
        text = requirement.lower()
        best_domain, best_concerns, best_score = "general", GENERAL_CONCERNS, 0
        for domain, patterns, concerns in COMPILED_CATALOG:
            score = sum(1 for pattern in patterns if pattern.search(text))
            if score > best_score:
                best_domain, best_concerns, best_score = domain, concerns, score
        return DomainProfile(
            domain=best_domain,
            business_object=analysis.business_object or "item",
            key_concerns=list(best_concerns),
        )
