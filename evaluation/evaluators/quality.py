# File: evaluation/evaluators/quality.py
# Description: Measures scenario quantity, coverage, quality fields, and type diversity.
# Author Name: Debleena Nandy
# Date: 07-10-2026
"""Deterministic quality metrics for generated QA scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario
from evaluation.evaluators.traceability import TraceabilityEvaluator

REQUIRED_SCENARIO_TYPES = {
    "functional",
    "positive",
    "negative",
    "boundary",
    "equivalence",
    "integration",
    "api",
    "ui",
    "security",
    "error_handling",
}


@dataclass(frozen=True)
class ScenarioQuality:
    scenario_count: int
    scenario_count_score: float
    type_diversity: float
    criterion_coverage: float
    complete_scenarios: int
    total_scenarios: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ScenarioQualityEvaluator:
    """Scores scenario outputs against explicit, reviewable quality dimensions."""

    def evaluate(
        self,
        criteria: Sequence[AcceptanceCriterion],
        scenarios: Sequence[QAScenario],
    ) -> ScenarioQuality:
        types = {scenario.type for scenario in scenarios}
        coverage = TraceabilityEvaluator().evaluate(list(criteria), list(scenarios))
        criterion_coverage = (
            len(coverage.covered_criterion_ids) / coverage.criterion_count
            if coverage.criterion_count
            else 1.0
        )
        complete = sum(
            bool(scenario.title.strip() and scenario.steps and scenario.expected_result.strip())
            for scenario in scenarios
        )
        return ScenarioQuality(
            scenario_count=len(scenarios),
            scenario_count_score=min(len(scenarios) / 15, 1.0),
            type_diversity=len(types & REQUIRED_SCENARIO_TYPES) / len(REQUIRED_SCENARIO_TYPES),
            criterion_coverage=criterion_coverage,
            complete_scenarios=complete,
            total_scenarios=len(scenarios),
        )
