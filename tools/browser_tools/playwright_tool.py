"""Run reviewed browser actions and capture artifacts for QA evidence."""
# File: tools/browser_tools/playwright_tool.py
# Description: Executes constrained Playwright actions and captures failure evidence.
# Author Name: Debleena Nandy
# Date: 07-10-2026

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.models.actions import BrowserAction
from tools.api_tools.target_policy import ResolvedTarget


@dataclass
class BrowserRunOutcome:
    passed: bool
    evidence: List[str] = field(default_factory=list)
    failure: Optional[str] = None
    error_type: Optional[str] = None
    console_errors: List[str] = field(default_factory=list)
    network_failures: List[str] = field(default_factory=list)
    http_statuses: List[int] = field(default_factory=list)
    artifacts: List[Path] = field(default_factory=list)
    duration_ms: int = 0


def playwright_available() -> bool:
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


class PlaywrightBrowserTool:
    """Async Playwright driver confined to one origin.

    Security: Chromium resolves the target host only to the pinned IP and every off-origin request is aborted.
    Evidence: screenshot, DOM snapshot, Playwright trace, console errors and failed requests are always collected.
    """

    def __init__(self, headless: bool = True, timeout_ms: int = 5000, executable_path: str = ""):
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.executable_path = executable_path or None

    def run(self, target: ResolvedTarget, actions: List[BrowserAction], artifact_dir: Path) -> BrowserRunOutcome:
        """Sync facade: runs the async session on a private event loop (safe inside FastAPI worker threads)."""
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run_async(target, actions, artifact_dir)).result()

    async def run_async(self, target: ResolvedTarget, actions: List[BrowserAction], artifact_dir: Path) -> BrowserRunOutcome:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeout
        from playwright.async_api import async_playwright

        artifact_dir.mkdir(parents=True, exist_ok=True)
        outcome = BrowserRunOutcome(passed=False)
        started = time.monotonic()
        origin = target.origin
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                headless=self.headless,
                executable_path=self.executable_path,
                args=[f"--host-resolver-rules=MAP {target.host} {target.pinned_ip}"],
            )
            context = await browser.new_context(base_url=origin)
            await context.tracing.start(screenshots=True, snapshots=True)
            page = await context.new_page()
            page.set_default_timeout(self.timeout_ms)

            async def guard(route) -> None:  
                url = route.request.url
                if url == origin or url.startswith(origin + "/"):
                    await route.continue_()
                else:
                    outcome.network_failures.append(f"Blocked off-origin request to {url[:120]}")
                    await route.abort()

            await context.route("**/*", guard)
            page.on("console", lambda msg: outcome.console_errors.append(msg.text[:300]) if msg.type == "error" else None)
            page.on("requestfailed", lambda req: outcome.network_failures.append(f"{req.method} {req.url[:120]} failed"))
            page.on("response", lambda resp: outcome.http_statuses.append(resp.status) if resp.url.startswith(origin) else None)

            try:
                for index, action in enumerate(actions, start=1):
                    await self._perform(page, action)
                    outcome.evidence.append(f"Browser step {index}: {self._describe(action)} ok.")
                outcome.passed = True
            except AssertionError as exc:
                outcome.failure, outcome.error_type = str(exc), "AssertionError"
            except PlaywrightTimeout as exc:
                outcome.failure, outcome.error_type = f"Timed out: {str(exc).splitlines()[0][:200]}", "TimeoutError"
            except PlaywrightError as exc:
                message = str(exc).splitlines()[0][:200]
                outcome.error_type = "NavigationError" if "net::" in message else "PlaywrightError"
                outcome.failure = message
            finally:
                screenshot, dom, trace = artifact_dir / "final.png", artifact_dir / "dom.html", artifact_dir / "trace.zip"
                try:
                    await page.screenshot(path=str(screenshot), full_page=True)
                    outcome.artifacts.append(screenshot)
                    dom.write_text(await page.content(), encoding="utf-8")
                    outcome.artifacts.append(dom)
                except PlaywrightError:
                    pass
                await context.tracing.stop(path=str(trace))
                outcome.artifacts.append(trace)
                await context.close()
                await browser.close()
        outcome.duration_ms = int((time.monotonic() - started) * 1000)
        if not outcome.passed:
            outcome.evidence.append(f"Browser failure: {outcome.failure}")
        return outcome

    async def _perform(self, page, action: BrowserAction) -> None:
        if action.type == "goto":
            response = await page.goto(action.path or "/")
            if response is not None and response.status >= 400:
                raise AssertionError(f"Navigation to {action.path} returned HTTP {response.status}")
        elif action.type == "fill":
            await page.fill(action.selector, action.value or "")
        elif action.type == "click":
            await page.click(action.selector)
        elif action.type == "select":
            await page.select_option(action.selector, action.value)
        elif action.type == "expect_visible":
            await page.wait_for_selector(action.selector, state="visible")
        elif action.type == "expect_text":
            locator = page.locator(action.selector)
            await locator.wait_for(state="visible")
            deadline = time.monotonic() + self.timeout_ms / 1000
            while time.monotonic() < deadline:
                if (action.value or "") in await locator.inner_text():
                    return
                await asyncio.sleep(0.1)
            raise AssertionError(f"Text at '{action.selector}' did not contain the expected value")
        elif action.type == "expect_url_contains":
            if (action.value or "") not in page.url:
                raise AssertionError("Current URL did not contain the expected value")

    @staticmethod
    def _describe(action: BrowserAction) -> str:
        if action.type == "goto":
            return f"goto {action.path}"
        if action.type in ("fill", "select"):
            return f"{action.type} {action.selector}"
        return f"{action.type} {action.selector or action.value}"