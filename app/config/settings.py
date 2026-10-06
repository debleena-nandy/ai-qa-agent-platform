from pydantic_settings import BaseSettings, SettingsConfigDict


# What: Central configuration for the API and the optional LLM provider.
# Why: The same code must run offline in CI (rule-based) and with a local LLM (Ollama) in development.
# How: Values come from environment variables or .env; LLM usage is opt-in via LLM_ENABLED=true.
class Settings(BaseSettings):
    app_name: str = "ai-qa-agent-platform"
    app_env: str = "development"
    api_port: int = 8000

    llm_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    llm_timeout_seconds: int = 60
    llm_healthcheck_timeout_seconds: int = 3

    max_requirement_chars: int = 5000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
