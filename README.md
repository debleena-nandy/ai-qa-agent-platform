# AI QA Agent Platform

Turns a user story into a domain-aware QA report: actor, goal, business object, domain, key concerns,
ambiguities, and 15-20 concrete, executable test scenarios (preconditions, steps, expected result).

## Current Architecture (implemented)
```text
User Story
    |
    v
Requirement Agent      actor, goal, business object, benefit, explicit rules & acceptance criteria, ambiguities
    |
    v
Domain Agent           domain + key QA concerns (LLM, or offline domain catalogue)
    |
    v
Test Design Agent      15-20 structured scenarios (LLM with strict prompt, or domain templates)
    |
    v
Executor               records every scenario as "not_executed" until runners are connected
    |
    v
Failure Analyzer       classification from real step statuses
```
Orchestrated with LangGraph (`app/graph/workflow.py`). Every agent falls back to rule-based logic if the
LLM is disabled, unreachable, or returns invalid output. The response field `generation_mode`
shows which path produced the scenarios.

## Roadmap (not implemented yet)
- Playwright browser runner with screenshot and trace capture
- API test runner using `ApiTool`
- Test Planning and Root Cause agents
- QA report generator and SQLite/PostgreSQL persistence
- GitHub Actions workflow (Jenkins pipeline exists)

## Technology Stack
Python 3.11+, FastAPI, Pydantic, LangGraph, Ollama (optional), pytest, ruff, Docker, Jenkins.

## Folder Structure
```text
01-ai-qa-agent-platform/
├── app/
│   ├── agents/       requirement, domain, test design, failure analysis
│   ├── api/          FastAPI app
│   ├── config/       settings (.env)
│   ├── execution/    executor
│   ├── graph/        LangGraph workflow
│   ├── llm/          Ollama client
│   ├── models/       scenario and execution models
│   └── tools/        API / browser tools
├── tests/
│   ├── unit/
│   └── e2e/
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
├── pytest.ini
└── requirements.txt
```

## Configuration
Copy `.env.example` to `.env`. The LLM is off by default so tests and CI stay offline.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_ENABLED` | `false` | Use Ollama for domain extraction and scenario generation |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `llama3.1` | Model name (`ollama pull llama3.1`) |

## Local Development
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python -m uvicorn app.api.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs and call `POST /analyze`:
```json
{ "requirement": "As a customer, I want to place an order to buy a toy" }
```

## Docker
```bash
docker build -t ai-qa-agent-platform .
docker run --rm -p 8000:8000 ai-qa-agent-platform
docker run --rm ai-qa-agent-platform pytest -q
docker compose up --build          # LLM_ENABLED=true docker compose up to use Ollama on the host
```

## CI (Jenkins)
Builds the image, runs `ruff` and `pytest` inside the container, and publishes `reports/junit.xml`.
The pipeline fails when lint or tests fail.

## Definition of Done
- Application builds and runs in Docker with a health check
- Lint and tests run in CI inside the container
- Test results are published as pipeline artifacts
- Works offline (rule-based) and with a local LLM
