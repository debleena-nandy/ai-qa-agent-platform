# File: config/settings.py
# Description: Defines environment-based application and LLM settings.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations
from typing import Set
from urllib.parse import urlsplit
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# What: Central configuration for the API and the optional LLM provider.
# Why: The same code must run offline in CI (rule-based) and with a local LLM (Ollama) in development.
# How: Values come from environment variables or .env; LLM usage is opt-in via LLM_ENABLED=true.
class Settings(BaseSettings):
    app_name: str = "ai-qa-agent-platform"
    app_env: str = "development"
    api_port: int = 8000
    log_level: str = "INFO"

    # LLM (mandatory). Agents depend only on the StructuredLLM protocol; Ollama is the default provider.
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_num_ctx: int = 8192     # Ollama's default context is too small for scenario prompts  
    ollama_num_predict: int = 3072    # cap on output tokens per call # Ollama's default context is too small for scenario prompts
    ollama_embedding_model: str = "nomic-embed-text"
    llm_timeout_seconds: int = 300    # CPU inference is slow; one scenario call can take minutes
    llm_healthcheck_timeout_seconds: int = 3
    llm_max_retries: int = 2 # small models need an extra self-correction round more often
    llm_max_tool_steps: int = 3
    llm_check_on_startup: bool = True

    # Optional deterministic fallback. Off by default: this is an AI-first platform.
    rule_based_fallback: bool = False

    # Requirements and API protection
    max_requirement_chars: int = 5000
    qa_approval_token: str = ""
    analyze_requires_token: bool = False
    max_stored_runs: int = 500

    # Execution target (one HTTP origin serves API and UI)
    qa_api_base_url: str = ""
    qa_api_timeout_seconds: int = 10
    qa_allowed_target_hosts: str = "" # comma list; empty = host of qa_api_base_url only
    qa_allow_private_targets: bool = False # True for Docker Compose networks
    # Browser runner (Playwright)
    browser_enabled: bool = False
    browser_headless: bool = True
    browser_timeout_ms: int = 5000
    playwright_chromium_executable: str = ""    

    '''qa_browser_base_url: str = ""
    qa_browser_timeout_ms: int = 10_000
    qa_browser_headless: bool = True
    qa_approval_token: str = ""'''

    # Persistence and artifacts
    qa_database_path: str = "data/qa_runs.sqlite3"
    qa_artifacts_dir: str = Field(
        default="artifacts",
        validation_alias=AliasChoices("qa_artifacts_dir", "qa_artifact_directory"),
    )

    # RAG
    rag_enabled: bool = True
    knowledge_dir: str = "tools/rag_tools/corpus"
    rag_top_k: int = 3
    rag_use_embeddings: bool = True

    # Test design sizing (smaller batches for small CPU models)
    test_design_target_count: str = "10-12"
    test_design_min_scenarios: int = 8

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    def allowed_hosts_for(self, base_url: str) -> Set[str]:
        configured = {h.strip().lower() for h in self.qa_allowed_target_hosts.split(",") if h.strip()}
        if configured:
            return configured
        host = urlsplit(base_url).hostname
        return {host.lower()} if host else set()


settings = Settings()
