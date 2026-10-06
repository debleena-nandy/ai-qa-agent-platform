from __future__ import annotations

from types import TracebackType
from typing import Any, Dict, Optional, Tuple

import requests

from app.config.settings import settings
from app.llm.base import LLMResponseError, LLMUnavailableError, T

__all__ = ["OllamaClient", "LLMResponseError", "LLMUnavailableError", "build_llm_client", "CONNECT_TIMEOUT_SECONDS"]

CONNECT_TIMEOUT_SECONDS = 3.05


def _with_default_tag(model: str) -> str:
    return model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = requests.Session()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    @property
    def _request_timeout(self) -> Tuple[float, float]:
        return (CONNECT_TIMEOUT_SECONDS, float(self.timeout))

    def generate_structured(self, prompt: str, schema: type[T]) -> T:
        try:
            response = self._post_chat(prompt, schema)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise LLMUnavailableError(f"Ollama is not reachable at {self.base_url}: {exc}") from exc
        if not response.ok:
            raise LLMResponseError(f"Ollama returned HTTP {response.status_code}: {self._error_text(response)}")
        try:
            content = response.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMResponseError(f"Unexpected Ollama response: {response.text[:200]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("Ollama returned an empty message.")
        return schema.model_validate_json(content)

    def _post_chat(self, prompt: str, schema: type[T]) -> requests.Response:
        return self._session.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "format": schema.model_json_schema(),
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=self._request_timeout,
        )

    def health(self, timeout: float = 3) -> Dict[str, Any]:
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=timeout)
            response.raise_for_status()
            models = response.json().get("models") or []
            names = {m.get("model") or m.get("name") or "" for m in models}
        except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
            return {"reachable": False, "model_available": False, "model": self.model, "error": str(exc)[:200]}
        installed = {_with_default_tag(name) for name in names if name}
        return {"reachable": True, "model_available": _with_default_tag(self.model) in installed, "model": self.model}

    @staticmethod
    def _error_text(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text[:200]
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])[:200]
        return response.text[:200]


def build_llm_client() -> Optional[OllamaClient]:
    if not settings.llm_enabled:
        return None
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.llm_timeout_seconds,
    )
