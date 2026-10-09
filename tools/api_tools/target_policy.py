"""Constrain test targets to explicit loopback-only HTTP origins."""
# File: tools/api_tools/target_policy.py
# Description: Validates API target origins and rejects non-loopback destinations.
# Author Name: Debleena Nandy
# Date: 07-10-2026

from __future__ import annotations
import ipaddress
import socket
from dataclasses import dataclass
from typing import Set
from urllib.parse import urlsplit

@dataclass(frozen=True)
class ResolvedTarget:
    """A validated origin plus the exact IP the runners must connect to (prevents DNS rebinding)."""
    scheme: str
    host: str
    port: int
    pinned_ip: str
    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"
    @property
    def pinned_origin(self) -> str:
        ip = f"[{self.pinned_ip}]" if ":" in self.pinned_ip else self.pinned_ip
        return f"{self.scheme}://{ip}:{self.port}"
    
    @property
    def host_header(self) -> str:
        return f"{self.host}:{self.port}"

class TargetPolicy:
    """Loopback is always allowed; private networks only when explicitly enabled; public never."""

    def __init__(self, allowed_hosts: Set[str], allow_private: bool = False):
        self.allowed_hosts = {host.lower() for host in allowed_hosts}
        self.allow_private = allow_private

    def resolve(self, url: str) -> ResolvedTarget:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("API target must be an HTTP URL without embedded credentials.")
        host = parsed.hostname.lower()
        if not self.allowed_hosts or host not in self.allowed_hosts:
            raise ValueError(f"API target host '{host}' is not in the configured allow-list.")
        port = parsed.port or 80
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            except OSError as exc:
                raise ValueError(f"API target host '{host}' could not be resolved.") from exc
            addresses = sorted({ipaddress.ip_address(info[4][0]) for info in infos}, key=str)
        if not addresses:
            raise ValueError(f"API target host '{host}' could not be resolved.")
        for address in addresses:
            if address.is_loopback:
                continue
            if self.allow_private and address.is_private and not address.is_link_local and not address.is_multicast:
                continue
            raise ValueError(
                "API targets must resolve exclusively to loopback addresses"
                + (" or private network addresses." if self.allow_private else ".")
            )
        return ResolvedTarget(scheme="http", host=host, port=port, pinned_ip=str(addresses[0]))

'''class TargetPolicy:
    """Loopback is always allowed; private networks only when explicitly enabled; public never."""
    def __init__(self, base_url: str, api_tool: ApiTool | None = None):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("API target must be a plain HTTP origin without credentials, path, query, or fragment.")
        self.host = parsed.hostname.lower()
        self.base_url = f"http://{parsed.netloc}"
        self.api_tool = api_tool or ApiTool({self.host})
        self.owns_api_tool = api_tool is None
        try:
            self.api_tool.validate_target_url(self.base_url, {self.host})
        except ValueError:
            if self.owns_api_tool:
                self.api_tool.close()
            raise

    def close(self) -> None:
        self.api_tool.close()'''
