"""Read and query the configured loopback target's OpenAPI document."""
# File: tools/api_tools/openapi.py
# Description: Exposes read-only OpenAPI operation listing and lookup for a QA target.
# Author Name: Debleena Nandy
# Date: 07-10-2026
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tools.api_tools.client import ApiTool
from tools.api_tools.target_policy import ResolvedTarget

HTTP_METHODS = ("get", "post", "put", "patch", "delete")
EndpointLines = List[str]


@dataclass(frozen=True)
class Endpoint:
    method: str
    path: str
    summary: str

    def matches(self, method: str, concrete_path: str) -> bool:
        if method.upper() != self.method:
            return False
        pattern = "^" + re.sub(r"\\\{[^/]+?\\\}", r"[^/]+", re.escape(self.path)) + "$"
        path_only = re.sub(r"\{\{[a-z0-9_]+\}\}", "x", concrete_path.split("?")[0])
        return re.match(pattern, path_only) is not None


class OpenApiSpec:
    """Read-only view of a target's OpenAPI document: the grounding for action design (tool calling)."""

    def __init__(self, document: Dict[str, Any]):
        self.document = document or {}
        self.endpoints: List[Endpoint] = []
        for path, operations in (self.document.get("paths") or {}).items():
            for method, operation in (operations or {}).items():
                if method in HTTP_METHODS:
                    self.endpoints.append(Endpoint(method.upper(), path, (operation or {}).get("summary", "")))

    def has(self, method: str, path: str) -> bool:
        return any(endpoint.matches(method, path) for endpoint in self.endpoints)

    def list_endpoints(self) -> EndpointLines:
        return [f"{e.method} {e.path} - {e.summary}" for e in self.endpoints]

    def endpoint_schema(self, method: str, path: str) -> Dict[str, Any]:
        operation = ((self.document.get("paths") or {}).get(path) or {}).get(method.lower())
        if operation is None:
            return {"error": f"No operation {method.upper()} {path}"}
        body = ((operation.get("requestBody") or {}).get("content") or {}).get("application/json", {}).get("schema")
        return {
            "parameters": [
                {"name": p.get("name"), "in": p.get("in"), "required": p.get("required", False)}
                for p in operation.get("parameters", [])
            ],
            "request_body": self._resolve(body),
            "responses": sorted((operation.get("responses") or {}).keys()),
        }

    def _resolve(self, schema: Optional[Dict[str, Any]], depth: int = 0) -> Any:
        if not schema or depth > 3:
            return schema
        ref = schema.get("$ref")
        if ref and ref.startswith("#/components/schemas/"):
            name = ref.rsplit("/", 1)[-1]
            return self._resolve((self.document.get("components") or {}).get("schemas", {}).get(name), depth + 1)
        return schema


def fetch_openapi(api_tool: ApiTool, target: ResolvedTarget, timeout: float = 10) -> OpenApiSpec:
    response = api_tool.request("GET", target, "/openapi.json", timeout=timeout)
    if response["status_code"] != 200 or not isinstance(response["json"], dict):
        raise ValueError(f"Target did not serve an OpenAPI document (status {response['status_code']}).")
    return OpenApiSpec(response["json"])