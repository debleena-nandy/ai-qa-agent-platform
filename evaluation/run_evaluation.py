# File: evaluation/run_evaluation.py
# Description: Runs deterministic scenario quality gates against a versioned evaluation dataset.
# Author Name: Debleena Nandy
# Date: 07-10-2026
"""Run offline QA scenario quality evaluation and write a JSON report."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agents.requirement_agent.agent import RequirementAgent
from agents.test_design_agent.agent import TestDesignAgent
from core.llm.base import LLMUnavailableError
from evaluation.evaluators.quality import ScenarioQualityEvaluator

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "datasets" / "scenarios.json"
BASELINE_PATH = ROOT / "evaluation" / "datasets" / "baseline.json"
REPORT_PATH = ROOT / "evaluation" / "reports" / "latest.json"

Case = Dict[str, Any]


class OfflineLLM:
    """Always unavailable: forces the deterministic rule-based path so CI gates are reproducible."""

    def generate_structured(self, prompt: str, schema: type) -> Any:
        raise LLMUnavailableError("Offline evaluation: no LLM is used.")


def _threshold(case: Case, baseline: Case, case_key: str, baseline_key: str) -> float:
    """A case may tighten a gate; otherwise the versioned baseline applies."""
    return case.get(case_key, baseline[baseline_key])


def run_evaluation(update_baseline: bool = False) -> Dict[str, Any]:
    dataset: List[Case] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    baseline: Case = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    llm = OfflineLLM()
    requirement_agent = RequirementAgent(llm, use_fallback=True)
    test_design_agent = TestDesignAgent(llm, use_fallback=True)
    evaluator = ScenarioQualityEvaluator()
    cases: List[Case] = []
    failures: List[str] = []

    for case in dataset:
        requirement = case["requirement"]
        analysis, domain, analysis_mode = requirement_agent.analyze(requirement)
        scenarios, mode = test_design_agent.generate_scenarios(analysis, domain, requirement)
        metrics = evaluator.evaluate(analysis.criteria, scenarios).to_dict()
        gates = {
            "domain": domain.domain == case["expected_domain"],
            "scenario_count": metrics["scenario_count"] >= _threshold(
                case, baseline, "minimum_scenarios", "minimum_scenario_count"),
            "type_diversity": metrics["type_diversity"] >= _threshold(
                case, baseline, "minimum_type_diversity", "minimum_type_diversity"),
            "criterion_coverage": metrics["criterion_coverage"] >= _threshold(
                case, baseline, "minimum_criterion_coverage", "minimum_criterion_coverage"),
            "scenario_completeness": (
                metrics["complete_scenarios"] / max(metrics["total_scenarios"], 1)
                >= baseline["minimum_complete_scenario_ratio"]
            ),
        }
        cases.append({
            "id": case["id"], "domain": domain.domain, "analysis_mode": analysis_mode,
            "generation_mode": mode, "metrics": metrics, "gates": gates,
        })
        failures.extend(f"{case['id']}:{name}" for name, passed in gates.items() if not passed)

    if update_baseline and cases:
        # Baselines only ever move down to the observed minimum; review the diff before committing.
        baseline["minimum_scenario_count"] = min(c["metrics"]["scenario_count"] for c in cases)
        baseline["minimum_type_diversity"] = min(c["metrics"]["type_diversity"] for c in cases)
        baseline["minimum_criterion_coverage"] = min(c["metrics"]["criterion_coverage"] for c in cases)
        BASELINE_PATH.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_PATH.relative_to(ROOT).as_posix(),
        "mode": "offline_rule_based",
        "cases": cases,
        "passed": not failures,
        "failures": failures,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline QA scenario quality gates.")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Lower baseline.json to the observed minimums (review before committing).")
    args = parser.parse_args()
    report = run_evaluation(args.update_baseline)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())