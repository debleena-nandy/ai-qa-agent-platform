# AI QA Agent Platform

Turns a user story into a domain-aware QA report: actor, goal, business object, domain, key concerns,
ambiguities, and 15-20 concrete, reviewable test scenarios (preconditions, steps, expected result).

## Current Architecture (implemented)
```text
User Story
    |
    v
Requirement Agent      actor, goal, business object, benefit, rules, acceptance criteria with AC ids, ambiguities
    |
    v
Domain Agent           domain + key QA concerns (LLM, or offline domain catalogue)
    |
    v
Test Design Agent      15-20 structured scenarios (LLM with strict prompt, or domain templates)
    |
    v
Test Planning Agent    plan items, criterion links, approval state, and explicit execution blockers
    |
    v
Execution Agent        enforces approval/eligibility; records not_executed until structured actions and runners exist
    |
    v
Failure Analyzer       aggregate execution-status classification
    |
    v
Root Cause Agent       cautious hypothesis; reports insufficient evidence rather than inventing a cause
    |
    v
QA Report Agent        criterion coverage, execution summary, blockers, and recommendations
```
Orchestrated with LangGraph (`graph/workflow.py`). `status` indicates workflow processing completion;
`execution_status` separately reports whether test execution has started. Scenarios link to acceptance criteria
through `criterion_ids`; their natural-language steps are not yet machine-executable actions. Plans are pending
approval and blocked until structured actions and a configured target exist.

Every agent falls back to rule-based logic if the
LLM is disabled, unreachable, or returns invalid output. The response field `generation_mode`
shows which path produced the scenarios.

## Portfolio Architecture Mapping

| Portfolio component | Current code | Status |
|---|---|---|
| Requirement Analysis Agent | `agents/requirement_agent/agent.py`, `core/models/requirement.py` | Parses story fields and assigns requirement-local acceptance-criterion IDs. |
| Test Design Agent | `agents/test_design_agent/agent.py`, `core/models/scenario.py` | LLM and deterministic scenario generation; emits ten scenario types and links scenarios to criteria. |
| Test Planning Agent | `agents/test_planning_agent/agent.py`, `core/models/test_plan.py` | Builds plans, links criteria, and blocks actions without approval, configured targets, and structured actions. |
| Execution Agent | `agents/execution_agent/agent.py`, `tools/api_tools/runner.py`, `tools/browser_tools/runner.py` | Runs approved structured API and browser actions against configured loopback targets. |
| Browser/API testing | `tools/api_tools/`, `tools/browser_tools/`, `demo_target/` | OpenAPI-constrained API actions, Playwright actions, DOM/trace/screenshot/console/network artifacts, and synthetic order API. |
| Failure Analyzer | `agents/failure_analysis_agent/agent.py` | Classifies execution statuses and reports captured evidence. |
| Root Cause Agent | `agents/root_cause_agent/agent.py`, `tools/rag_tools/` | Conservative assessments with related local failure-pattern guidance. |
| QA Report Generator | `agents/reporting_agent/agent.py`, `core/models/qa_report.py` | Produces structured reports and supports JSON/Markdown artifact export. |
| Workflow orchestration | `graph/workflow.py`, `graph/state.py`, `graph/nodes.py`, `graph/engine.py` | Runs the ordered workflow through reporting and execution stages. |
| Shared API and configuration | `api/`, `config/` | FastAPI endpoints and environment-based settings. |
| Persistence | `core/persistence/run_repository.py` | SQLite run history, approval state, and execution results. |
| Evaluation | `evaluation/datasets/`, `evaluation/evaluators/`, `evaluation/run_evaluation.py` | Versioned scenario dataset, traceability/quality metrics, and offline CI gates. |
| Artifact tools | `tools/filesystem_tools/`, `tools/reporting_tools/` | Restricted artifact storage and JSON/Markdown report export. |
| Delivery pipeline | `.github/workflows/ci.yml`, `Dockerfile`, `docker-compose.yml` | GitHub Actions quality gates and Docker build/run configuration. |

## Limitations
- API and browser execution is intentionally restricted to explicit loopback origins.
- Generated LLM scenarios are draft-only; executable actions must be provided and reviewed separately.
- API actions must match the configured target's OpenAPI document.
- The API runner cannot retrieve server-side stack traces from the target.
- Root-cause output is a hypothesis based on collected evidence, not an automatic defect declaration.
- Local retrieval uses deterministic TF-IDF over checked-in guidance; it does not use embeddings or a hosted vector database.
- SQLite is used for local persistence; PostgreSQL is a future deployment option.

## Technology Stack
Python 3.11+, FastAPI, Pydantic, LangGraph, Ollama (optional), pytest, Playwright, ruff, mypy, Docker, GitHub Actions.

## Folder Structure
```text
01-ai-qa-agent-platform/
├── agents/           one package per QA agent
├── graph/            state, nodes, workflow, and engine
├── tools/             API, browser, filesystem, reporting, and RAG tools
├── api/               FastAPI app
├── config/            environment-based settings
├── core/              shared LLM, models, and persistence
├── demo_target/       synthetic local order API
├── evaluation/        datasets, evaluators, reports, and evaluation runner
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── scripts/           restructure, extraction, and demo scripts
├── docs/              architecture and sample reports
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements-dev.txt
├── pyproject.toml
├── pytest.ini
└── requirements.txt
```

## Configuration
Copy `.env.example` to `.env`. The LLM is mandatory; tests and CI use fake LLMs and RULE_BASED_FALLBACK=true, so they stay offline.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_ENABLED` | `false` | Use Ollama for domain extraction and scenario generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model name (`ollama pull qwen2.5:3b`) |
| `QA_API_BASE_URL` | unset | Explicit loopback-only HTTP origin for the API runner; no target is contacted unless a plan is approved |
| `QA_API_TIMEOUT_SECONDS` | `10` | Maximum wait for one API action |
| `QA_BROWSER_BASE_URL` | unset | Explicit loopback-only target for the Playwright runner |
| `QA_BROWSER_TIMEOUT_MS` | `10000` | Maximum wait for an individual browser action |
| `QA_ARTIFACT_DIRECTORY` | `artifacts` | Root for screenshots, DOM snapshots, traces, and logs |
| `QA_DATABASE_PATH` | `data/qa_runs.sqlite3` | SQLite file for persistent runs and reports |
| `QA_APPROVAL_TOKEN` | unset | Required token protecting structured run, plan approval, execution, and retrieval endpoints |

## Local Development
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q
python -m uvicorn api.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs and call `POST /analyze`:
```json
{ "requirement": "As a customer, I want to place an order to buy a toy" }
```
To run the controlled API demo target separately, start it on loopback:
```powershell
python -m uvicorn demo_target.main:app --host 127.0.0.1 --port 8001
```
The API runner accepts only loopback targets. `QA_API_BASE_URL=http://127.0.0.1:8001` selects this demo origin;
generated natural-language scenarios without reviewed structured actions stay blocked and cannot be executed.
Set `QA_APPROVAL_TOKEN` before using `POST /runs`, plan approval, or execution; send it as the
`X-QA-Approval-Token` header. `POST /runs` accepts a requirement and explicit structured scenarios, while
`POST /analyze` remains the LLM/rule-based scenario-design flow. Runs and reports can be retrieved with
`GET /runs/{run_id}` and plans with `GET /plans/{plan_id}`.

## Docker
```bash
docker build -t ai-qa-agent-platform .
docker run --rm -p 8000:8000 ai-qa-agent-platform
docker run --rm ai-qa-agent-platform pytest -q
docker compose up --build          # LLM_ENABLED=true docker compose up to use Ollama on the host
```

## Evaluation
Run the deterministic offline quality dataset and inspect its JSON report:
```powershell
python -m evaluation.run_evaluation
```
The command checks domain, scenario count, all ten scenario types, completeness, and criterion coverage.
Use `--update-baseline` only when intentionally accepting new metric baselines.

## Demo
Run an offline sample requirement through the workflow:
```powershell
python scripts/demo_run.py
```

## CI
GitHub Actions runs Ruff, mypy, unit, integration, end-to-end, and AI evaluation gates, then builds the Docker image.

## Definition of Done
- Application builds and runs in Docker with a health check
- Lint and tests run in CI inside the container
- Test results are published as pipeline artifacts
- Works offline (rule-based) and with a local LLM
