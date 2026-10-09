# File: core/models/execution_result.py
# Description: Defines individual step results and aggregated execution outcomes.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# What: The only statuses a step may have once it is recorded.
# Why: An unknown status (e.g. "pending", "error") was silently not counted, so the summary no longer added up.
VALID_STATUSES = ("passed", "failed", "skipped", "not_executed")

ExecutionStatus = Literal["not_started", "partial", "completed"]


@dataclass
class AssertionOutcome:
    """Structured result of one check, so failure analysis never parses free text."""

    kind: str  # status_code | status_in | json_equals | json_exists | json_not_exists | browser
    passed: bool
    expected_statuses: List[int] = field(default_factory=list)
    actual_status: Optional[int] = None
    json_path: Optional[str] = None
    path_missing: bool = False
    description: str = ""


@dataclass
class ExecutionStepResult:
    """Detailed result for one QA scenario."""

    scenario: str
    status: str = "not_executed"  # passed | failed | skipped | not_executed
    execution_type: str = "generic"
    evidence: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    details: Optional[str] = None
    response_data: Any = None
    scenario_id: str = ""
    scenario_type: str = ""
    error_type: Optional[str] = None
    http_statuses: List[int] = field(default_factory=list)
    assertions: List[AssertionOutcome] = field(default_factory=list)
    console_errors: List[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class ExecutionResult:
    """Aggregate result for a set of QA scenarios."""

    total_scenarios: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    not_executed: int = 0
    steps: List[ExecutionStepResult] = field(default_factory=list)

    def add_step(self, step: ExecutionStepResult) -> None:
        if step.status not in VALID_STATUSES:
            raise ValueError(f"Unknown step status '{step.status}'; expected one of {VALID_STATUSES}.")
        self.steps.append(step)
        self.total_scenarios = len(self.steps)
        setattr(self, step.status, getattr(self, step.status) + 1)

    def extend(self, other: ExecutionResult) -> None:
        for step in other.steps:
            self.add_step(step)

    def summary(self) -> Dict[str, int]:
        return {
            "total_scenarios": self.total_scenarios,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "not_executed": self.not_executed,
        }

    @property
    def attempted(self) -> int:
        return self.passed + self.failed


# FIX: was an indented @staticmethod inside ExecutionResult, so
# `from core.models.execution_result import derive_execution_status` failed
# (Pylance error + ImportError at startup in api/main.py and graph/nodes.py).
def derive_execution_status(approved: bool, execution: ExecutionResult) -> ExecutionStatus:
    """Single source of truth for not_started / partial / completed (used by the graph and the API)."""
    if not approved or execution.total_scenarios == 0 or execution.attempted == 0:
        return "not_started"
    if execution.not_executed == 0:
        return "completed"
    return "partial"