# File: evaluation/evaluators/traceability.py
# Description: Evaluates traceability coverage between acceptance criteria and QA scenarios.
# Author Name: Debleena Nandy
# Date: 2026-10-07
# Time: 11:41:01 +05:30
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario


@dataclass(frozen=True)
class CriterionCoverage:
    criterion_count: int
    covered_criterion_ids: List[str]
    uncovered_criterion_ids: List[str]
    unknown_criterion_ids: List[str]


class TraceabilityEvaluator:
    """Measures criterion-to-scenario links and flags references to unknown criteria."""

    def evaluate(
        self,
        criteria: List[AcceptanceCriterion],
        scenarios: List[QAScenario],
    ) -> CriterionCoverage:
        expected_ids = {criterion.id for criterion in criteria}
        linked_ids = {criterion_id for scenario in scenarios for criterion_id in scenario.criterion_ids}
        covered = expected_ids & linked_ids
        return CriterionCoverage(
            criterion_count=len(expected_ids),
            covered_criterion_ids=sorted(covered),
            uncovered_criterion_ids=sorted(expected_ids - linked_ids),
            unknown_criterion_ids=sorted(linked_ids - expected_ids),
        )
