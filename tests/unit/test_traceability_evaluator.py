# File: tests/unit/test_traceability_evaluator.py
# Description: Tests acceptance criterion coverage evaluation.
# Author Name: Debleena Nandy
# Date: 2026-10-07
# Time: 11:41:01 +05:30
from __future__ import annotations

from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario
from evaluation.evaluators.traceability import TraceabilityEvaluator


def test_evaluator_reports_missing_and_unknown_criterion_links() -> None:
    criteria = [
        AcceptanceCriterion(id="AC-001", description="Order quantity is limited"),
        AcceptanceCriterion(id="AC-002", description="Out of stock orders are rejected"),
    ]
    scenarios = [
        QAScenario(
            id="TC-001",
            title="Check quantity limit",
            type="boundary",
            steps=["Submit limit + 1"],
            expected_result="The quantity is rejected",
            criterion_ids=["AC-001", "AC-999"],
        )
    ]

    result = TraceabilityEvaluator().evaluate(criteria, scenarios)

    assert result.criterion_count == 2
    assert result.covered_criterion_ids == ["AC-001"]
    assert result.uncovered_criterion_ids == ["AC-002"]
    assert result.unknown_criterion_ids == ["AC-999"]
