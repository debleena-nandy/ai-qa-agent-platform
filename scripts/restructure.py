"""Move the application into the root package layout used by the portfolio."""
# File: scripts/restructure.py
# Description: Migrates application modules and rewrites imports to the portfolio layout.
# Author Name: Debleena Nandy
# Date: 07-10-2026

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MOVES = {
    "app/agents/requirement_agent.py": "agents/requirement_agent/agent.py",
    "app/agents/domain_agent.py": "agents/requirement_agent/domain.py",
    "app/agents/test_design_agent.py": "agents/test_design_agent/agent.py",
    "app/agents/test_planning_agent.py": "agents/test_planning_agent/agent.py",
    "app/agents/execution_agent.py": "agents/execution_agent/agent.py",
    "app/execution/executor.py": "agents/execution_agent/routing.py",
    "app/agents/failure_analysis_agent.py": "agents/failure_analysis_agent/agent.py",
    "app/agents/root_cause_agent.py": "agents/root_cause_agent/agent.py",
    "app/agents/reporting_agent.py": "agents/reporting_agent/agent.py",
    "app/graph/state.py": "graph/state.py",
    "app/graph/nodes.py": "graph/nodes.py",
    "app/graph/workflow.py": "graph/workflow.py",
    "app/tools/api_tools.py": "tools/api_tools/client.py",
    "app/execution/api_runner.py": "tools/api_tools/runner.py",
    "app/tools/browser_tools.py": "tools/browser_tools/playwright_tool.py",
    "app/tools/filesystem_tools.py": "tools/filesystem_tools/store.py",
    "app/tools/reporting_tools.py": "tools/reporting_tools/writer.py",
    "app/api/main.py": "api/main.py",
    "app/config/settings.py": "config/settings.py",
    "app/llm/base.py": "core/llm/base.py",
    "app/llm/ollama_client.py": "core/llm/ollama_client.py",
    "app/models/actions.py": "core/models/actions.py",
    "app/models/execution_result.py": "core/models/execution_result.py",
    "app/models/qa_report.py": "core/models/qa_report.py",
    "app/models/requirement.py": "core/models/requirement.py",
    "app/models/scenario.py": "core/models/scenario.py",
    "app/models/test_plan.py": "core/models/test_plan.py",
    "app/persistence/run_repository.py": "core/persistence/run_repository.py",
    "app/demo/main.py": "demo_target/main.py",
    "app/demo/scenarios.py": "demo_target/scenarios.py",
    "extract_code.py": "scripts/extract_code.py",
}

MODULES = {
    "app.agents.requirement_agent": "agents.requirement_agent.agent",
    "app.agents.domain_agent": "agents.requirement_agent.domain",
    "app.agents.test_design_agent": "agents.test_design_agent.agent",
    "app.agents.test_planning_agent": "agents.test_planning_agent.agent",
    "app.agents.execution_agent": "agents.execution_agent.agent",
    "app.execution.executor": "agents.execution_agent.routing",
    "app.agents.failure_analysis_agent": "agents.failure_analysis_agent.agent",
    "app.agents.root_cause_agent": "agents.root_cause_agent.agent",
    "app.agents.reporting_agent": "agents.reporting_agent.agent",
    "app.graph.state": "graph.state",
    "app.graph.nodes": "graph.nodes",
    "app.graph.workflow": "graph.workflow",
    "app.tools.api_tools": "tools.api_tools.client",
    "app.execution.api_runner": "tools.api_tools.runner",
    "app.tools.browser_tools": "tools.browser_tools.playwright_tool",
    "app.tools.filesystem_tools": "tools.filesystem_tools.store",
    "app.tools.reporting_tools": "tools.reporting_tools.writer",
    "app.api.main": "api.main",
    "app.config.settings": "config.settings",
    "app.llm.base": "core.llm.base",
    "app.llm.ollama_client": "core.llm.ollama_client",
    "app.models.actions": "core.models.actions",
    "app.models.execution_result": "core.models.execution_result",
    "app.models.qa_report": "core.models.qa_report",
    "app.models.requirement": "core.models.requirement",
    "app.models.scenario": "core.models.scenario",
    "app.models.test_plan": "core.models.test_plan",
    "app.persistence.run_repository": "core.persistence.run_repository",
    "app.demo.main": "demo_target.main",
    "app.demo.scenarios": "demo_target.scenarios",
}
PACKAGES = {
    "app.agents": "agents",
    "app.api": "api",
    "app.config": "config",
    "app.execution": "agents.execution_agent",
    "app.graph": "graph",
    "app.llm": "core.llm",
    "app.models": "core.models",
    "app.persistence": "core.persistence",
    "app.tools": "tools",
    "app.demo": "demo_target",
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache"}


def _move(source: Path, destination: Path, dry_run: bool) -> None:
    if not source.exists():
        return
    if destination.exists():
        raise FileExistsError(f"Migration destination already exists: {destination.relative_to(ROOT)}")
    print(f"move   {source.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def _ensure_packages(dry_run: bool) -> None:
    package_dirs = {
        (ROOT / destination).parent
        for destination in MOVES.values()
        if destination.endswith(".py") and destination.startswith(("agents/", "graph/", "tools/", "api/", "config/", "core/", "demo_target/"))
    }
    package_dirs.update(
        ROOT / name
        for name in ("agents", "graph", "tools", "api", "config", "core", "demo_target")
    )
    for directory in sorted(package_dirs):
        while directory != ROOT:
            init_file = directory / "__init__.py"
            if not init_file.exists():
                print(f"create {init_file.relative_to(ROOT)}")
                if not dry_run:
                    directory.mkdir(parents=True, exist_ok=True)
                    init_file.write_text("", encoding="utf-8")
            directory = directory.parent


def _rewrite_imports(dry_run: bool) -> None:
    mapping = {**PACKAGES, **MODULES}
    names = sorted((name for name in mapping if name), key=len, reverse=True)
    pattern = re.compile(r"(?<![\w.])(" + "|".join(map(re.escape, names)) + r")(?!\w)")
    for path in ROOT.rglob("*.py"):
        if SKIP_DIRS.intersection(path.parts) or path == Path(__file__):
            continue
        text = path.read_text(encoding="utf-8")
        updated = pattern.sub(lambda match: mapping[match.group(1)], text)
        header = re.compile(r"^# File: .*$", re.MULTILINE)
        updated = header.sub(f"# File: {path.relative_to(ROOT).as_posix()}", updated, count=1)
        if updated != text:
            print(f"edit   {path.relative_to(ROOT)}")
            if not dry_run:
                path.write_text(updated, encoding="utf-8", newline="")


def _remove_legacy_app(dry_run: bool) -> None:
    legacy_dir = ROOT / "app"
    if not legacy_dir.exists():
        return
    unexpected = [
        path.relative_to(ROOT)
        for path in legacy_dir.rglob("*")
        if path.is_file() and path.suffix != ".pyc" and path.name != "__init__.py"
    ]
    if unexpected:
        raise RuntimeError(f"Legacy app/ contains unmigrated files: {unexpected}")
    print("remove app/ (migrated modules and generated bytecode only)")
    if not dry_run:
        shutil.rmtree(legacy_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="List changes without applying them")
    parser.add_argument("--imports-only", action="store_true", help="Only rewrite import paths and headers")
    args = parser.parse_args()
    if not args.imports_only:
        for source, destination in MOVES.items():
            _move(ROOT / source, ROOT / destination, args.dry_run)
        _ensure_packages(args.dry_run)
    _rewrite_imports(args.dry_run)
    if not args.dry_run and not args.imports_only:
        _remove_legacy_app(False)


if __name__ == "__main__":
    main()
