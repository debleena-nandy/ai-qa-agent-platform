# File: tools/filesystem_tools/store.py
# Description: Provides safe, workspace-confined artifact file storage.
# Author Name: Debleena Nandy
# Date: 2026-10-07
# Time: 11:41:01 +05:30

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class WorkspaceFileStore:
    """Writes generated artifacts beneath one configured root directory."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    def write_text(self, relative_path: str, content: str) -> str:
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                             prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
                temporary_path = handle.name
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
        except OSError:
            if temporary_path is not None:
                Path(temporary_path).unlink(missing_ok=True)
            raise
        return str(target)

    def resolve(self, relative_path: str) -> Path:
        """Public so runners (screenshots, traces, DOM) obtain a path that cannot escape the root."""
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("Artifact path must be relative to the configured root.")
        target = (self.root / candidate).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Artifact path must remain inside the configured root.")
        return target

    def relative(self, path: Path | str) -> str:
        return str(Path(path).resolve().relative_to(self.root)).replace("\\", "/")