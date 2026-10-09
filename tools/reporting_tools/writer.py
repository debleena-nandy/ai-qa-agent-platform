# File: tools/reporting_tools/writer.py
# Description: Serializes QA reports to the configured artifact store.
# Author Name: Debleena Nandy
# Date: 2026-10-07
# Time: 11:41:01 +05:30

from __future__ import annotations

from typing import Any, Dict, List

from core.models.qa_report import QAReport
from tools.filesystem_tools.store import WorkspaceFileStore

Paths = List[str]


class QAReportWriter:
    """Serializes QA reports to JSON and Markdown inside the restricted artifact store."""

    def __init__(self, file_store: WorkspaceFileStore):
        self.file_store = file_store

    def write_json(self, report: QAReport, relative_path: str) -> str:
        return self.file_store.write_text(relative_path, report.model_dump_json(indent=2) + "\n")

    def write_markdown(self, report: QAReport, relative_path: str, run: Dict[str, Any] | None = None) -> str:
        return self.file_store.write_text(relative_path, render_markdown(report, run or {}))

    def write_all(self, report: QAReport, prefix: str, run: Dict[str, Any] | None = None) -> Paths:
        return [self.write_json(report, f"{prefix}/report.json"),
                self.write_markdown(report, f"{prefix}/report.md", run)]


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: QAReport, run: Dict[str, Any]) -> str:
    s = report.execution_summary
    rate = "-" if report.pass_rate is None else f"{report.pass_rate:.0%}"
    lines = [
        f"# QA Report — {_cell(run.get('title', report.plan_id))}",
        "",
        f"- **Run:** {run.get('run_id', '-')}",
        f"- **Plan:** {report.plan_id} ({run.get('test_plan', {}).get('approval_status', '-')})",
        f"- **Domain:** {run.get('domain', '-')}",
        f"- **Execution status:** {report.execution_status}",
        f"- **Classification:** {report.failure_classification}",
        "",
        "## Execution summary",
        "",
        "| Total | Passed | Failed | Skipped | Not executed | Pass rate |",
        "|---|---|---|---|---|---|",
        f"| {s.get('total_scenarios', 0)} | {s.get('passed', 0)} | {s.get('failed', 0)} | {s.get('skipped', 0)} "
        f"| {s.get('not_executed', 0)} | {rate} |",
        "",
        "## Acceptance-criteria coverage",
        "",
        f"Covered: {', '.join(report.covered_criterion_ids) or 'none'} · "
        f"Uncovered: {', '.join(report.uncovered_criterion_ids) or 'none'}",
        "",
    ]
    if report.step_classifications:
        lines += ["## Failed tests", "", "| Scenario | Category | Confidence | Rationale |", "|---|---|---|---|"]
        lines += [f"| {_cell(c.scenario)} | {c.category} | {c.confidence:.2f} | {_cell(c.rationale)} |"
                  for c in report.step_classifications]
        lines.append("")
    rc = report.root_cause
    confidence = "n/a" if rc.confidence is None else f"{rc.confidence:.2f}"
    lines += ["## Root cause", "",
              f"**Category:** {rc.category} · **Confidence:** {confidence} · **Source:** {rc.hypothesis_source}", "",
              rc.hypothesis, "", f"_Basis: {rc.confidence_basis}_", ""]
    sections = (
        ("Suggested investigation", rc.suggested_investigation),
        ("Suggested remediation", rc.suggested_remediation),
        ("Similar known issues (RAG)", rc.similar_known_issues),
        ("Defect candidates (require human confirmation)",
         [f"{d.scenario_id} {d.title} — {d.category} ({d.confidence:.2f})" for d in report.defect_candidates]),
        ("Risk areas", report.risk_areas),
        ("Recommended regression scope", report.recommended_regression_scope),
        ("Artifacts", report.artifacts),
        ("Recommendations", report.recommendations),
    )
    for title, items in sections:
        if items:
            lines += [f"## {title}", ""] + [f"- {_cell(item)}" for item in items] + [""]
    return "\n".join(lines)