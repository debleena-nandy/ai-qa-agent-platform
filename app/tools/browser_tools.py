from __future__ import annotations


# What: Interface for browser execution tools.
# Why: Returning "Screenshot saved" without saving anything is fake evidence, which the executor explicitly avoids.
# How: Fails loudly until the Playwright implementation exists.
class BrowserTool:
    """Interface for browser execution tools (Playwright implementation is on the roadmap)."""

    def open_page(self, url: str) -> str:
        raise NotImplementedError("Browser runner not implemented yet (roadmap: Playwright).")

    def capture_screenshot(self, path: str) -> str:
        raise NotImplementedError("Browser runner not implemented yet (roadmap: Playwright).")
