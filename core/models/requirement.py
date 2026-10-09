# File: core/models/requirement.py
# Description: Defines the acceptance criterion model for requirements.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

from pydantic import BaseModel, Field


class AcceptanceCriterion(BaseModel):
    id: str = Field(pattern=r"^AC-\d{3}$", description="Requirement-local id such as AC-001")
    description: str = Field(min_length=1, description="A verifiable acceptance criterion")
