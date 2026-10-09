# File: tests/unit/test_artifact_tools.py
# Description: Tests QA report serialization and safe artifact file storage.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations

import json

import pytest

from agents.execution_agent.agent import ExecutionAgent
from agents.failure_analysis_agent.agent import FailureAnalysisAgent
from agents.reporting_agent.agent import QAReportAgent
from agents.root_cause_agent.agent import RootCauseAgent
from agents.test_planning_agent.agent import TestPlanningAgent
from core.models.execution_result import derive_execution_status
from core.models.requirement import AcceptanceCriterion
from core.models.scenario import QAScenario
from tests.fakes import DeadLLM
from tools.filesystem_tools.store import WorkspaceFileStore
from tools.reporting_tools.writer import QAReportWriter


def test_report_writer_writes_json_under_configured_root(tmp_path) -> None:
    llm = DeadLLM()
    scenario = QAScenario(id="TC-001", title="Verify order limit", type="boundary",
                          steps=["Submit maximum quantity"], expected_result="Limit is enforced",
                          criterion_ids=["AC-001"])
    criteria = [AcceptanceCriterion(id="AC-001", description="Orders are limited")]
    plan = TestPlanningAgent(None).create_plan([scenario])
    execution = ExecutionAgent().execute(plan, [scenario])
    failure = FailureAnalysisAgent(llm, use_fallback=True).analyze(execution)
    root_cause = RootCauseAgent(llm, use_fallback=True).analyze(execution, failure)
    report = QAReportAgent(llm, use_fallback=True).generate(
        plan=plan, criteria=criteria, scenarios=[scenario], execution=execution, failure=failure,
        root_cause=root_cause, workflow_status="completed",
        execution_status=derive_execution_status(False, execution),
    )
    saved_path = QAReportWriter(WorkspaceFileStore(tmp_path)).write_json(report, "runs/report.json")
    saved = json.loads((tmp_path / "runs" / "report.json").read_text(encoding="utf-8"))
    assert saved["plan_id"] == plan.id
    assert saved["execution_status"] == "not_started"
    assert saved_path.endswith("report.json")


@pytest.mark.parametrize("path", ["../outside.json", "nested/../../outside.json"])
def test_file_store_rejects_paths_outside_root(tmp_path, path: str) -> None:
    store = WorkspaceFileStore(tmp_path / "artifacts")
    with pytest.raises(ValueError, match="inside the configured root"):
        store.write_text(path, "not allowed")