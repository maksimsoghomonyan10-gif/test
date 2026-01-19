"""
Browser Controller - Playwright-based browser automation.
"""

import asyncio
import logging
import base64
from typing import Optional, Dict, Any
from pathlib import Path
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

logger = logging.getLogger(__name__)


class BrowserController:
    """
    High-level browser controller using Playwright.
    
    Handles browser lifecycle, navigation, and provides
    access to the page for actions and observations.
    """
    
    def __init__(
        self,
        headless: bool = False,
        browser_type: str = "chromium",
        viewport: Dict[str, int] = None,
        user_agent: str = None,
        slow_mo: int = 0
    ):
        self.headless = headless
        self.browser_type = browser_type
        self.viewport = viewport or {"width": 1280, "height": 800}
        self.user_agent = user_agent
        self.slow_mo = slow_mo
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        
        self._screenshot_dir = Path("screenshots")
        self._screenshot_dir.mkdir(exist_ok=True)
        
    @property
    def page(self) -> Optional[Page]:
        """Get the current page."""
        return self._page
    
    @property
    def is_running(self) -> bool:
        """Check if browser is running."""
        return self._browser is not None and self._browser.is_connected()
    
    async def start(self, url: str = "about:blank"):
        """Start the browser and navigate to initial URL."""
        logger.info(f"Starting {self.browser_type} browser (headless={self.headless})")
        
        self._playwright = await async_playwright().start()
        
        # Select browser type
        browser_launcher = getattr(self._playwright, self.browser_type)
        
        self._browser = await browser_launcher.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        
        # Create context with viewport and other settings
        context_options = {
            "viewport": self.viewport,
            "ignore_https_errors": True,
        }
        
        if self.user_agent:
            context_options["user_agent"] = self.user_agent
            
        self._context = await self._browser.new_context(**context_options)
        
        # Enable console logging from page
        self._page = await self._context.new_page()
        self._page.on("console", self._handle_console)
        self._page.on("pageerror", self._handle_page_error)
        
        if url != "about:blank":
            await self.navigate(url)
            
        logger.info("Browser started successfully")
        
    async def close(self):
        """Close the browser and cleanup."""
        logger.info("Closing browser")
        
        if self._page:
            await self._page.close()
            self._page = None
            
        if self._context:
            await self._context.close()
            self._context = None
            
        if self._browser:
            await self._browser.close()
            self._browser = None
            
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
            
    async def navigate(self, url: str, wait_until: str = "domcontentloaded"):
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation done
                       ('load', 'domcontentloaded', 'networkidle')
        """
        if not self._page:
            raise RuntimeError("Browser not started")
            
        logger.info(f"Navigating to: {url}")
        
        try:
            await self._page.goto(url, wait_until=wait_until, timeout=30000)
            await self._wait_for_stable()
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            raise
            
    async def screenshot(
        self,
        path: str = None,
        full_page: bool = False,
        return_base64: bool = True
    ) -> Optional[str]:
        """
        Take a screenshot of the current page.
        
        Args:
            path: Optional file path to save screenshot
            full_page: Whether to capture full scrollable page
            return_base64: Whether to return base64 encoded image
            
        Returns:
            Base64 encoded screenshot if return_base64=True
        """
        if not self._page:
            return None
            
        screenshot_bytes = await self._page.screenshot(full_page=full_page)
        
        if path:
            save_path = self._screenshot_dir / path
            save_path.write_bytes(screenshot_bytes)
            logger.debug(f"Screenshot saved to: {save_path}")
            
        if return_base64:
            return base64.b64encode(screenshot_bytes).decode('utf-8')
            
        return None
    
    async def get_page_content(self) -> str:
        """Get the current page HTML content."""
        if not self._page:
            return ""
        return await self._page.content()
    
    async def get_page_info(self) -> Dict[str, Any]:
        """Get basic page information."""
        if not self._page:
            return {}
            
        return {
            "url": self._page.url,
            "title": await self._page.title(),
            "viewport": self._page.viewport_size
        }
    
    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript on the page."""
        if not self._page:
            return None
        return await self._page.evaluate(script)
    
    async def wait_for_selector(
        self,
        selector: str,
        state: str = "visible",
        timeout: int = 10000
    ) -> bool:
        """
        Wait for an element to appear.
        
        Args:
            selector: CSS selector
            state: State to wait for ('attached', 'detached', 'visible', 'hidden')
            timeout: Maximum time to wait in milliseconds
            
        Returns:
            True if element found, False if timeout
        """
        if not self._page:
            return False
            
        try:
            await self._page.wait_for_selector(
                selector,
                state=state,
                timeout=timeout
            )
            return True
        except:
            return False
            
    async def wait_for_navigation(self, timeout: int = 30000):
        """Wait for navigation to complete."""
        if not self._page:
            return
            
        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=timeout)
        except:
            pass  # Timeout is acceptable in some cases
            
    async def _wait_for_stable(self, timeout: float = 2.0):
        """Wait for page to become stable (no major changes)."""
        await asyncio.sleep(0.5)  # Initial wait
        
        # Could implement more sophisticated stability detection here
        # (e.g., comparing DOM snapshots)
        
    def _handle_console(self, msg):
        """Handle console messages from the page."""
        if msg.type == "error":
            logger.debug(f"Page console error: {msg.text}")
        elif msg.type == "warning":
            logger.debug(f"Page console warning: {msg.text}")
            
    def _handle_page_error(self, error):
        """Handle page errors."""
        logger.warning(f"Page error: {error}")


class BrowserPool:
    """
    Pool of browser instances for parallel execution.
    """
    
    def __init__(self, pool_size: int = 3, **browser_kwargs):
        self.pool_size = pool_size
        self.browser_kwargs = browser_kwargs
        self._browsers: list[BrowserController] = []
        self._available: asyncio.Queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize the browser pool."""
        for _ in range(self.pool_size):
            browser = BrowserController(**self.browser_kwargs)
            await browser.start()
            self._browsers.append(browser)
            await self._available.put(browser)
            
    async def acquire(self) -> BrowserController:
        """Acquire a browser from the pool."""
        return await self._available.get()
    
    async def release(self, browser: BrowserController):
        """Release a browser back to the pool."""
        # Reset browser state
        await browser.navigate("about:blank")
        await self._available.put(browser)
        
    async def close_all(self):
        """Close all browsers in the pool."""
        for browser in self._browsers:
            await browser.close()
        self._browsers.clear()
