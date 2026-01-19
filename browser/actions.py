"""
Browser Actions - Click, type, scroll, and other browser interactions.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Union
from playwright.async_api import Page, ElementHandle, TimeoutError as PlaywrightTimeout

from agent.tools import ActionResult

logger = logging.getLogger(__name__)


class BrowserActions:
    """
    Browser action implementations for the agent.
    
    Each method returns an ActionResult indicating success/failure.
    """
    
    def __init__(self, browser_controller):
        self.browser = browser_controller
        
    @property
    def page(self) -> Page:
        return self.browser.page
    
    # ==================== Navigation Actions ====================
    
    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> ActionResult:
        """
        Navigate to a URL.
        
        Args:
            url: URL to navigate to
            wait_until: When to consider navigation complete
        """
        try:
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            await self.page.goto(url, wait_until=wait_until, timeout=30000)
            await asyncio.sleep(1)  # Extra wait for dynamic content
            
            return ActionResult(
                success=True,
                message=f"Navigated to {url}",
                data={"url": self.page.url}
            )
        except PlaywrightTimeout:
            return ActionResult(
                success=False,
                message=f"Navigation timeout for {url}",
                data=None
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Navigation failed: {str(e)}",
                data=None
            )
    
    async def go_back(self) -> ActionResult:
        """Navigate back in history."""
        try:
            await self.page.go_back(wait_until="domcontentloaded")
            return ActionResult(
                success=True,
                message="Navigated back",
                data={"url": self.page.url}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Go back failed: {str(e)}",
                data=None
            )
    
    async def go_forward(self) -> ActionResult:
        """Navigate forward in history."""
        try:
            await self.page.go_forward(wait_until="domcontentloaded")
            return ActionResult(
                success=True,
                message="Navigated forward",
                data={"url": self.page.url}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Go forward failed: {str(e)}",
                data=None
            )
    
    async def refresh(self) -> ActionResult:
        """Refresh the current page."""
        try:
            await self.page.reload(wait_until="domcontentloaded")
            return ActionResult(
                success=True,
                message="Page refreshed",
                data={"url": self.page.url}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Refresh failed: {str(e)}",
                data=None
            )
    
    # ==================== Click Actions ====================
    
    async def click(
        self,
        selector: str = None,
        text: str = None,
        index: int = None,
        position: Dict[str, int] = None,
        button: str = "left",
        click_count: int = 1,
        timeout: int = 10000
    ) -> ActionResult:
        """
        Click on an element.
        
        Args:
            selector: CSS selector
            text: Text content to find element by
            index: Element index from DOM snapshot
            position: {x, y} coordinates for click
            button: Mouse button ('left', 'right', 'middle')
            click_count: Number of clicks (1 for single, 2 for double)
            timeout: Maximum time to wait for element
        """
        try:
            element = await self._find_element(selector, text, index, timeout)
            
            if element:
                await element.scroll_into_view_if_needed()
                await element.click(button=button, click_count=click_count, timeout=timeout)
                await asyncio.sleep(0.5)  # Wait for potential navigation/changes
                
                return ActionResult(
                    success=True,
                    message=f"Clicked element",
                    data={"selector": selector, "text": text}
                )
            elif position:
                await self.page.mouse.click(position['x'], position['y'], button=button)
                return ActionResult(
                    success=True,
                    message=f"Clicked at position ({position['x']}, {position['y']})",
                    data=position
                )
            else:
                return ActionResult(
                    success=False,
                    message="Element not found",
                    data=None
                )
        except PlaywrightTimeout:
            return ActionResult(
                success=False,
                message=f"Click timeout - element not found or not clickable",
                data={"selector": selector, "text": text}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Click failed: {str(e)}",
                data=None
            )
    
    async def hover(
        self,
        selector: str = None,
        text: str = None,
        index: int = None,
        timeout: int = 10000
    ) -> ActionResult:
        """Hover over an element."""
        try:
            element = await self._find_element(selector, text, index, timeout)
            
            if element:
                await element.hover()
                return ActionResult(
                    success=True,
                    message="Hovered over element",
                    data={"selector": selector}
                )
            else:
                return ActionResult(
                    success=False,
                    message="Element not found for hover",
                    data=None
                )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Hover failed: {str(e)}",
                data=None
            )
    
    # ==================== Type Actions ====================
    
    async def type(
        self,
        text: str,
        selector: str = None,
        element_text: str = None,
        index: int = None,
        clear_first: bool = True,
        press_enter: bool = False,
        delay: int = 50,
        timeout: int = 10000
    ) -> ActionResult:
        """
        Type text into an input field.
        
        Args:
            text: Text to type
            selector: CSS selector for the input
            element_text: Find input by its placeholder/label text
            index: Element index from DOM snapshot
            clear_first: Whether to clear existing content first
            press_enter: Whether to press Enter after typing
            delay: Delay between keystrokes in ms
            timeout: Maximum time to wait for element
        """
        try:
            element = await self._find_element(selector, element_text, index, timeout)
            
            if element:
                await element.scroll_into_view_if_needed()
                
                if clear_first:
                    await element.fill("")
                    
                await element.type(text, delay=delay)
                
                if press_enter:
                    await element.press("Enter")
                    await asyncio.sleep(0.5)
                
                return ActionResult(
                    success=True,
                    message=f"Typed '{text[:50]}...' into field",
                    data={"text_length": len(text)}
                )
            else:
                return ActionResult(
                    success=False,
                    message="Input field not found",
                    data=None
                )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Type failed: {str(e)}",
                data=None
            )
    
    async def fill(
        self,
        text: str,
        selector: str = None,
        element_text: str = None,
        index: int = None,
        timeout: int = 10000
    ) -> ActionResult:
        """
        Fill an input field (faster than type, no keystroke simulation).
        """
        try:
            element = await self._find_element(selector, element_text, index, timeout)
            
            if element:
                await element.fill(text)
                return ActionResult(
                    success=True,
                    message=f"Filled field with '{text[:50]}'",
                    data={"text": text[:100]}
                )
            else:
                return ActionResult(
                    success=False,
                    message="Input field not found",
                    data=None
                )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Fill failed: {str(e)}",
                data=None
            )
    
    async def press_key(self, key: str, modifiers: list = None) -> ActionResult:
        """
        Press a keyboard key.
        
        Args:
            key: Key to press (e.g., 'Enter', 'Tab', 'Escape', 'a', 'ArrowDown')
            modifiers: Optional modifiers ['Control', 'Shift', 'Alt', 'Meta']
        """
        try:
            if modifiers:
                key_combo = "+".join(modifiers + [key])
            else:
                key_combo = key
                
            await self.page.keyboard.press(key_combo)
            
            return ActionResult(
                success=True,
                message=f"Pressed key: {key_combo}",
                data={"key": key_combo}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Key press failed: {str(e)}",
                data=None
            )
    
    # ==================== Scroll Actions ====================
    
    async def scroll(
        self,
        direction: str = "down",
        amount: int = 500,
        selector: str = None
    ) -> ActionResult:
        """
        Scroll the page or an element.
        
        Args:
            direction: 'up', 'down', 'left', 'right'
            amount: Pixels to scroll
            selector: Optional selector to scroll within element
        """
        try:
            if selector:
                element = await self.page.query_selector(selector)
                if element:
                    scroll_js = {
                        'down': f"el.scrollTop += {amount}",
                        'up': f"el.scrollTop -= {amount}",
                        'right': f"el.scrollLeft += {amount}",
                        'left': f"el.scrollLeft -= {amount}"
                    }
                    await self.page.evaluate(
                        f"(el) => {{ {scroll_js.get(direction, scroll_js['down'])} }}",
                        element
                    )
            else:
                scroll_js = {
                    'down': f"window.scrollBy(0, {amount})",
                    'up': f"window.scrollBy(0, -{amount})",
                    'right': f"window.scrollBy({amount}, 0)",
                    'left': f"window.scrollBy(-{amount}, 0)"
                }
                await self.page.evaluate(scroll_js.get(direction, scroll_js['down']))
            
            await asyncio.sleep(0.3)  # Wait for scroll animation
            
            return ActionResult(
                success=True,
                message=f"Scrolled {direction} by {amount}px",
                data={"direction": direction, "amount": amount}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Scroll failed: {str(e)}",
                data=None
            )
    
    async def scroll_to_element(
        self,
        selector: str = None,
        text: str = None,
        index: int = None
    ) -> ActionResult:
        """Scroll to bring an element into view."""
        try:
            element = await self._find_element(selector, text, index, 10000)
            
            if element:
                await element.scroll_into_view_if_needed()
                return ActionResult(
                    success=True,
                    message="Scrolled element into view",
                    data={"selector": selector}
                )
            else:
                return ActionResult(
                    success=False,
                    message="Element not found to scroll to",
                    data=None
                )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Scroll to element failed: {str(e)}",
                data=None
            )
    
    # ==================== Wait Actions ====================
    
    async def wait(
        self,
        duration: float = None,
        selector: str = None,
        text: str = None,
        state: str = "visible",
        timeout: int = 30000
    ) -> ActionResult:
        """
        Wait for time or element.
        
        Args:
            duration: Seconds to wait (if no selector)
            selector: CSS selector to wait for
            text: Text to wait to appear
            state: Element state to wait for ('visible', 'hidden', 'attached', 'detached')
            timeout: Maximum wait time in ms
        """
        try:
            if selector:
                await self.page.wait_for_selector(selector, state=state, timeout=timeout)
                return ActionResult(
                    success=True,
                    message=f"Element '{selector}' is now {state}",
                    data={"selector": selector, "state": state}
                )
            elif text:
                await self.page.wait_for_function(
                    f"document.body.innerText.includes('{text}')",
                    timeout=timeout
                )
                return ActionResult(
                    success=True,
                    message=f"Text '{text[:50]}' appeared",
                    data={"text": text[:100]}
                )
            elif duration:
                await asyncio.sleep(duration)
                return ActionResult(
                    success=True,
                    message=f"Waited {duration} seconds",
                    data={"duration": duration}
                )
            else:
                await asyncio.sleep(1)
                return ActionResult(
                    success=True,
                    message="Waited 1 second",
                    data={"duration": 1}
                )
        except PlaywrightTimeout:
            return ActionResult(
                success=False,
                message="Wait timeout exceeded",
                data=None
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Wait failed: {str(e)}",
                data=None
            )
    
    # ==================== Extract Actions ====================
    
    async def extract(
        self,
        selector: str = None,
        attribute: str = None,
        all_matches: bool = False
    ) -> ActionResult:
        """
        Extract text or attribute from element(s).
        
        Args:
            selector: CSS selector
            attribute: Attribute to extract (defaults to text content)
            all_matches: Whether to get all matching elements
        """
        try:
            if all_matches:
                elements = await self.page.query_selector_all(selector)
                if attribute:
                    values = [await el.get_attribute(attribute) for el in elements]
                else:
                    values = [await el.inner_text() for el in elements]
                
                return ActionResult(
                    success=True,
                    message=f"Extracted {len(values)} values",
                    data={"values": values}
                )
            else:
                element = await self.page.query_selector(selector)
                if element:
                    if attribute:
                        value = await element.get_attribute(attribute)
                    else:
                        value = await element.inner_text()
                    
                    return ActionResult(
                        success=True,
                        message=f"Extracted: {str(value)[:100]}",
                        data={"value": value}
                    )
                else:
                    return ActionResult(
                        success=False,
                        message="Element not found for extraction",
                        data=None
                    )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Extract failed: {str(e)}",
                data=None
            )
    
    # ==================== Select Actions ====================
    
    async def select(
        self,
        value: str = None,
        label: str = None,
        index: int = None,
        selector: str = None
    ) -> ActionResult:
        """
        Select an option from a dropdown.
        
        Args:
            value: Option value to select
            label: Option visible text to select
            index: Option index to select
            selector: CSS selector for the select element
        """
        try:
            element = await self.page.query_selector(selector)
            
            if not element:
                return ActionResult(
                    success=False,
                    message="Select element not found",
                    data=None
                )
            
            if value:
                await element.select_option(value=value)
            elif label:
                await element.select_option(label=label)
            elif index is not None:
                await element.select_option(index=index)
            
            return ActionResult(
                success=True,
                message=f"Selected option",
                data={"value": value, "label": label, "index": index}
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Select failed: {str(e)}",
                data=None
            )
    
    # ==================== Helper Methods ====================
    
    async def _find_element(
        self,
        selector: str = None,
        text: str = None,
        index: int = None,
        timeout: int = 10000
    ) -> Optional[ElementHandle]:
        """
        Find an element using various methods.
        
        Priority: selector > text > index
        """
        if selector:
            try:
                await self.page.wait_for_selector(selector, timeout=timeout)
                return await self.page.query_selector(selector)
            except:
                pass
        
        if text:
            # Try various text-based selectors
            text_selectors = [
                f"text={text}",
                f"button:has-text('{text}')",
                f"a:has-text('{text}')",
                f"[aria-label='{text}']",
                f"[placeholder='{text}']",
                f"label:has-text('{text}')"
            ]
            
            for sel in text_selectors:
                try:
                    element = await self.page.query_selector(sel)
                    if element:
                        return element
                except:
                    continue
        
        if index is not None:
            # Find by DOM snapshot index - requires interactive elements list
            interactive_selector = 'a, button, input, select, textarea, [role="button"], [role="link"], [tabindex]'
            elements = await self.page.query_selector_all(interactive_selector)
            if 0 <= index < len(elements):
                return elements[index]
        
        return None
    
    async def complete(self, reason: str = "Task completed") -> ActionResult:
        """
        Mark the task as complete.
        
        This is a special action that signals task completion.
        """
        return ActionResult(
            success=True,
            message=reason,
            data={"completed": True, "reason": reason}
        )
