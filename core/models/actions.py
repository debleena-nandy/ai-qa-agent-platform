# File: core/models/actions.py
# Description: Defines validated API actions and response assertions for test plans.
# Author Name: Debleena Nandy
# Date: 07-10-2026
# Time: 11:41:01 +05:30
from __future__ import annotations
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

HttpMethod = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
ApiAssertionType = Literal["status_code", "status_in", "json_equals", "json_exists", "json_not_exists"]
BrowserActionType = Literal["navigate", "goto", "fill", "click", "select", "expect_text", "expect_visible", "expect_url_contains", "expect_url", "screenshot"]
StatusList = List[int]

# Headers an action may never set: they change routing, target identity or transport framing.
FORBIDDEN_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "cookie", "proxy-authorization"}
VARIABLE_NAME = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

def validate_relative_path(path: str) -> str:
    """Shared rule for API and browser paths: one leading slash, no scheme, no traversal, no control chars."""
    if not path.startswith("/") or path.startswith("//"):
        raise ValueError("Path must be a relative path beginning with one slash.")
    if "://" in path or "\\" in path or any(part == ".." for part in path.split("?")[0].split("/")):
        raise ValueError("Path must not contain an absolute URL, backslash, or parent segment.")
    if any(ord(character) < 32 for character in path):
        raise ValueError("Path must not contain control characters.")
    return path

class ApiAssertion(BaseModel):
    type: ApiAssertionType
    expected: Any = None
    json_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_assertion_shape(self) -> ApiAssertion:
        if self.type == "status_code":
            if not isinstance(self.expected, int) or isinstance(self.expected, bool):
                raise ValueError("status_code assertion requires an integer expected value.")
            if not 100 <= self.expected <= 599:
                raise ValueError("Expected HTTP status code must be between 100 and 599.")
        elif self.type == "status_in":
            if not isinstance(self.expected, list) or not self.expected:
                raise ValueError("status_in assertion requires a non-empty list of status codes.")
            for code in self.expected:
                if not isinstance(code, int) or isinstance(code, bool) or not 100 <= code <= 599:
                    raise ValueError("status_in values must be HTTP status codes between 100 and 599.")
        elif not self.json_path:
            raise ValueError(f"{self.type} assertion requires json_path.")
        return self

    @property
    def expected_statuses(self) -> StatusList:
        if self.type == "status_code":
            return [self.expected]
        if self.type == "status_in":
            return list(self.expected)
        return []
    

class ApiRequestAction(BaseModel):
    method: HttpMethod
    path: str = Field(min_length=1, max_length=2048)
    headers: Dict[str, str] = Field(default_factory=dict)
    json_body: Optional[dict[str, Any]] = None
    assertions: list[ApiAssertion] = Field(min_length=1)
    # Values captured from this response for later actions, e.g. {"order_id": "id"}; used as {{order_id}}.
    capture: Dict[str, str] = Field(default_factory=dict)

    @field_validator("path")
    @classmethod
    def check_path(cls, path: str) -> str:
        return validate_relative_path(path)
    
    @field_validator("headers")
    @classmethod
    def check_headers(cls, headers: Dict[str, str]) -> Dict[str, str]|None:
        for name, value in headers.items():
            if name.lower() in FORBIDDEN_HEADERS:
                raise ValueError(f"Header '{name}' may not be set by a scenario action.")
            if any(ord(character) < 32 for character in name + value):
                raise ValueError("Header names and values must not contain control characters.")
        return headers
    
    @field_validator("capture")
    @classmethod
    def check_capture(cls, capture: Dict[str, str]) -> Dict[str, str]|None:
        for name, json_path in capture.items():
            if not VARIABLE_NAME.match(name) or not json_path:
                raise ValueError("Capture names must be lower_snake_case and map to a JSON path.")
        return capture

class BrowserAction(BaseModel):
    type: BrowserActionType
    selector: Optional[str] = Field(default=None, max_length=300)
    value: Optional[str] = Field(default=None, max_length=1000)
    path: Optional[str] = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def validate_shape(self) -> BrowserAction:
        if self.type == "goto":
            if not self.path:
                raise ValueError("goto requires a relative path.")
            validate_relative_path(self.path)
        elif self.type == "expect_url_contains":
            if not self.value:
                raise ValueError("expect_url_contains requires value.")
        else:
            if not self.selector:
                raise ValueError(f"{self.type} requires a selector.")
            if self.type in ("fill", "select", "expect_text") and self.value is None:
                raise ValueError(f"{self.type} requires value.")
        return self

    ###------- To Be Removed After Testing--------###
    @model_validator(mode="after")
    def validate_action_shape(self) -> BrowserAction:
        if self.type in {"click", "fill", "expect_text"} and not self.selector:
            raise ValueError(f"{self.type} browser action requires a selector.")
        if self.type in {"navigate", "expect_url", "fill", "expect_text"} and self.value is None:
            raise ValueError(f"{self.type} browser action requires a value.")
        if self.selector and any(ord(character) < 32 for character in self.selector):
            raise ValueError("Browser action selectors must not contain control characters.")
        if self.value and any(ord(character) < 32 for character in self.value):
            raise ValueError("Browser action values must not contain control characters.")
        if self.type in {"navigate", "expect_url"} and self.value:
            if self.value.startswith("//") or "://" in self.value or "\\" in self.value:
                raise ValueError("Browser navigation targets must be same-origin relative paths.")
        return self
