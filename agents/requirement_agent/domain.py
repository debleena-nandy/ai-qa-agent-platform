# File: agents/requirement_agent/domain.py
# Description: Detects and normalizes the business domain of a requirement.
# Author Name: Debleena Nandy
# Date: 2026-10-07
# Time: 11:41:01 +05:30

from __future__ import annotations

import re

DOMAIN_ALIASES = {
    "ecommerce": "e-commerce", "e commerce": "e-commerce", "online shopping": "e-commerce",
    "shopping": "e-commerce", "retail": "e-commerce", "orders": "e-commerce", "ordering": "e-commerce",
    "auth": "authentication", "login": "authentication", "identity": "authentication",
    "user authentication": "authentication",
    "account management": "account-management", "user management": "account-management",
    "profile management": "account-management", "user profile": "account-management",
    "payment": "payments", "fintech": "payments",
}


def normalize_domain(name: str) -> str:
    """Maps the model's free-text domain to a stable slug used by RAG, templates and evaluation."""
    key = re.sub(r"[_\s]+", " ", (name or "").strip().lower())
    return DOMAIN_ALIASES.get(key, key.replace(" ", "-") or "general")