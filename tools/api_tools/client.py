# File: tools/api_tools/client.py
# Description: Provides constrained HTTP requests for API scenario execution.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30

from __future__ import annotations

from typing import Any, Dict, Optional, Set

import requests
from requests import RequestException

from tools.api_tools.target_policy import ResolvedTarget, TargetPolicy


class ApiTool:
    """HTTP client bound to one validated target. Requests go to the pinned IP with the original Host header."""

    def __init__(self, allowed_hosts: Set[str] | None = None, allow_private: bool = False):
        self.policy = TargetPolicy(allowed_hosts or set(), allow_private=allow_private)
        self.session = requests.Session()
        self.session.trust_env = False  # never route through environment proxies

    def validate_target_url(self, url: str) -> ResolvedTarget:
        return self.policy.resolve(url)

    def request(
        self,
        method: str,
        target: ResolvedTarget,
        path: str,
        timeout: float = 10,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Request path must be relative to the target origin.")
        send_headers = dict(headers or {})
        send_headers["Host"] = target.host_header
        try:
            response = self.session.request(
                method.upper(), f"{target.pinned_origin}{path}", timeout=timeout,
                headers=send_headers, allow_redirects=False, **kwargs,
            )
        except RequestException as exc:
            return {
                "status_code": None, "headers": {}, "json": None, "text": "", "elapsed_ms": 0,
                "error": str(exc)[:300], "error_type": type(exc).__name__,
            }
        try:
            body: Any = response.json()
        except ValueError:
            body = None
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "json": body,
            "text": response.text,
            "elapsed_ms": int(response.elapsed.total_seconds() * 1000),
        }

    def close(self) -> None:
        self.session.close()

    @staticmethod
    def json_path_value(body: Any, json_path: str) -> Any:
        current = body
        for part in json_path.split("."):
            if not part:
                raise ValueError("JSON path must not contain empty segments.")
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                raise KeyError(json_path)
        return current