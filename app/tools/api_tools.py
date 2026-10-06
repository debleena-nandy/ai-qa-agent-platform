from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class ApiTool:
    """Utility wrapper around HTTP requests for QA workflows."""

    def request(self, method: str, url: str, timeout: int = 10, **kwargs: Any) -> Dict[str, Any]:
        response = requests.request(method.upper(), url, timeout=timeout, **kwargs)
        body: Optional[Any]
        try:
            body = response.json()
        except ValueError:
            body = None
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "json": body,
            "text": response.text,
            "elapsed_ms": int(response.elapsed.total_seconds() * 1000),
        }

    def get(self, url: str, timeout: int = 10, **kwargs: Any) -> Dict[str, Any]:
        return self.request("GET", url, timeout=timeout, **kwargs)

    def post(self, url: str, timeout: int = 10, **kwargs: Any) -> Dict[str, Any]:
        return self.request("POST", url, timeout=timeout, **kwargs)

    # Security/account templates use PUT/PATCH/DELETE ("PUT the profile", "cancel the order").
    def put(self, url: str, timeout: int = 10, **kwargs: Any) -> Dict[str, Any]:
        return self.request("PUT", url, timeout=timeout, **kwargs)

    def patch(self, url: str, timeout: int = 10, **kwargs: Any) -> Dict[str, Any]:
        return self.request("PATCH", url, timeout=timeout, **kwargs)

    def delete(self, url: str, timeout: int = 10, **kwargs: Any) -> Dict[str, Any]:
        return self.request("DELETE", url, timeout=timeout, **kwargs)
