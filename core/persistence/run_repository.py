# File: core/persistence/run_repository.py
# Description: Persists and retrieves QA run state using SQLite.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal
from uuid import uuid4

from core.models.test_plan import TestPlan

ApprovalDecision = Literal["approved", "rejected"]
FinalStatus = Literal["completed", "failed"]
Payload = Dict[str, Any]
Summaries = List[Payload]
Mutation = Callable[[str, Payload], str]
PlanChange = Callable[[TestPlan, Payload], TestPlan]


class RunNotFoundError(LookupError):
    pass


class RunStateError(ValueError):
    pass


class RunRepository:
    """SQLite persistence. State machine: created -> approved|rejected -> running -> completed|failed."""

    def __init__(self, database_path: str | Path, max_runs: int = 0):
        self.database_path = str(database_path)
        self.max_runs = max_runs
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_runs (
                    run_id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL UNIQUE,
                    run_status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        finally:
            connection.close()

    @staticmethod
    def _encode(payload: Payload) -> str:
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)

    def create_run(self, payload: Payload) -> Payload:
        stored = dict(payload)
        stored["run_id"] = str(uuid4())
        stored["run_status"] = "created"
        plan = TestPlan.model_validate(stored["test_plan"])
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO qa_runs(run_id, plan_id, run_status, payload) VALUES (?, ?, ?, ?)",
                (stored["run_id"], plan.id, "created", self._encode(stored)),
            )
            if self.max_runs > 0:  # retention: keep the newest N runs, never a running one
                connection.execute(
                    """
                    DELETE FROM qa_runs WHERE run_status != 'running' AND run_id NOT IN (
                        SELECT run_id FROM qa_runs ORDER BY created_at DESC, rowid DESC LIMIT ?
                    )
                    """,
                    (self.max_runs,),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return stored

    def _read(self, column: str, value: str) -> sqlite3.Row:
        connection = self._connect()
        try:
            row = connection.execute(
                f"SELECT run_id, run_status, payload FROM qa_runs WHERE {column} = ?", (value,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise RunNotFoundError(f"{'Run' if column == 'run_id' else 'Plan'} '{value}' was not found.")
        return row

    def get_run(self, run_id: str) -> Payload:
        return json.loads(self._read("run_id", run_id)["payload"])

    def get_plan(self, plan_id: str) -> TestPlan:
        return TestPlan.model_validate(json.loads(self._read("plan_id", plan_id)["payload"])["test_plan"])

    def list_runs(self, limit: int = 50) -> Summaries:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT run_id, plan_id, run_status, payload, created_at FROM qa_runs "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            connection.close()
        summaries: Summaries = []
        for row in rows:
            payload = json.loads(row["payload"])
            summaries.append({
                "run_id": row["run_id"], "plan_id": row["plan_id"], "run_status": row["run_status"],
                "title": payload.get("title", ""), "execution_status": payload.get("execution_status", ""),
                "created_at": row["created_at"],
            })
        return summaries

    def _transaction(self, column: str, value: str, mutate: Mutation) -> Payload:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT run_id, run_status, payload FROM qa_runs WHERE {column} = ?", (value,)
            ).fetchone()
            if row is None:
                raise RunNotFoundError(f"{'Run' if column == 'run_id' else 'Plan'} '{value}' was not found.")
            payload = json.loads(row["payload"])
            new_status = mutate(row["run_status"], payload)
            payload["run_status"] = new_status
            connection.execute(
                "UPDATE qa_runs SET run_status = ?, payload = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
                (new_status, self._encode(payload), row["run_id"]),
            )
            connection.commit()
            return payload
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def update_plan(self, plan_id: str, change: PlanChange) -> TestPlan:
        def mutate(status: str, payload: Payload) -> str:
            plan = TestPlan.model_validate(payload["test_plan"])
            if plan.approval_status != "pending" or status != "created":
                raise RunStateError("Only a pending plan can be edited.")
            payload["test_plan"] = change(plan, payload).model_dump(mode="json")
            return status

        return TestPlan.model_validate(self._transaction("plan_id", plan_id, mutate)["test_plan"])

    def decide_plan(self, plan_id: str, decision: ApprovalDecision) -> TestPlan:
        def mutate(status: str, payload: Payload) -> str:
            plan = TestPlan.model_validate(payload["test_plan"])
            if plan.approval_status != "pending" or status != "created":
                raise RunStateError("Only a pending plan can be approved or rejected.")
            if decision == "approved":
                problems = plan.approval_blockers()
                if problems:
                    raise RunStateError(" ".join(problems))
            payload["test_plan"] = plan.model_copy(update={"approval_status": decision}).model_dump(mode="json")
            return decision

        return TestPlan.model_validate(self._transaction("plan_id", plan_id, mutate)["test_plan"])

    def claim_execution(self, run_id: str) -> Payload:
        def mutate(status: str, payload: Payload) -> str:
            if status != "approved":
                raise RunStateError("Only an approved run can be executed, and each run can be executed once.")
            return "running"

        return self._transaction("run_id", run_id, mutate)

    def complete_run(self, run_id: str, updates: Payload) -> Payload:
        return self._finish(run_id, updates, "completed")

    def fail_run(self, run_id: str, message: str) -> Payload:
        return self._finish(run_id, {"execution_error": message}, "failed")

    def _finish(self, run_id: str, updates: Payload, final_status: FinalStatus) -> Payload:
        def mutate(status: str, payload: Payload) -> str:
            if status != "running":
                raise RunStateError("Only a running run can be completed.")
            payload.update(updates)
            return final_status

        return self._transaction("run_id", run_id, mutate)