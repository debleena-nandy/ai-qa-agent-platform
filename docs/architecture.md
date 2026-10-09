# Architecture

## Runtime flow

```text
Requirement -> Requirement Agent -> Domain Agent -> Test Design Agent
     -> Test Planning Agent -> human approval -> Execution Agent
     -> Failure Analysis -> Root Cause -> QA Report
```

`graph/engine.py` owns the public workflow lifecycle. `graph/workflow.py` composes
agents as a LangGraph state graph (or executes the same ordered nodes when
LangGraph is unavailable). API and browser runners are optional, independently
configured target integrations.

## Trust boundaries

- Requirement text and other user-controlled prompt fields are HTML-escaped and
  wrapped as untrusted data before being included in LLM prompts.
- LLM test-design output is a scenario draft only; it cannot directly supply
  executable API or browser actions.
- API actions are validated against the configured target's OpenAPI document.
- API and browser targets must be explicit, plain HTTP loopback origins.
- Browser actions are declarative, selector-based operations; navigation and
  requests are restricted to the configured origin.
- Plan execution requires approval. Artifacts are written under the configured
  artifacts directory, and reports avoid recording response bodies or secrets.

## Local retrieval and evaluation

`tools/rag_tools/knowledge_base.py` indexes checked-in Markdown guidance using
deterministic TF-IDF scoring. It uses no hosted service or generated embedding.
`evaluation/run_evaluation.py` runs the versioned JSON scenario dataset offline
and writes `evaluation/reports/latest.json`.

## Persistence and limitations

SQLite stores run and approval state. PostgreSQL is a future deployment option.
The API runner cannot inspect server-side stack traces from a remote process.
Failure classification is based only on evidence captured at the test boundary;
root-cause conclusions remain hypotheses.
