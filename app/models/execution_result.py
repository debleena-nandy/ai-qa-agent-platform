from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

# What: The only statuses a step may have once it is recorded.
# Why: An unknown status (e.g. "pending", "error") was silently not counted, so the summary no longer added up.
VALID_STATUSES = ("passed", "failed", "skipped", "not_executed")


@dataclass
class ExecutionStepResult:
    """Detailed result for one QA scenario."""

    scenario: str
    status: str = "not_executed"  # passed | failed | skipped | not_executed
    execution_type: str = "generic"
    evidence: List[str] = field(default_factory=list)
    details: Optional[str] = None
    response_data: Any = None


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

    def summary(self) -> dict:
        return {
            "total_scenarios": self.total_scenarios,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "not_executed": self.not_executed,
        }
