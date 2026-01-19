"""
DOM Snapshot - Capture and process DOM state for LLM consumption.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from playwright.async_api import Page

logger = logging.getLogger(__name__)


@dataclass
class ElementInfo:
    """Information about a DOM element."""
    tag: str
    id: str = ""
    classes: List[str] = field(default_factory=list)
    text: str = ""
    href: str = ""
    src: str = ""
    placeholder: str = ""
    value: str = ""
    aria_label: str = ""
    role: str = ""
    type: str = ""
    name: str = ""
    is_visible: bool = True
    is_enabled: bool = True
    is_interactive: bool = False
    bounding_box: Optional[Dict[str, float]] = None
    index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tag": self.tag,
            "id": self.id,
            "class": " ".join(self.classes),
            "text": self.text[:100] if self.text else "",
            "href": self.href,
            "placeholder": self.placeholder,
            "aria_label": self.aria_label,
            "role": self.role,
            "type": self.type,
            "is_visible": self.is_visible,
            "is_interactive": self.is_interactive,
            "index": self.index
        }


class DOMSnapshot:
    """
    Captures and processes DOM snapshots for the agent.
    
    Converts complex DOM into simplified representation
    suitable for LLM consumption.
    """
    
    # Interactive element tags
    INTERACTIVE_TAGS = {
        'a', 'button', 'input', 'select', 'textarea',
        'details', 'summary', 'dialog', 'menu', 'menuitem'
    }
    
    # Roles that indicate interactivity
    INTERACTIVE_ROLES = {
        'button', 'link', 'menuitem', 'option', 'radio',
        'checkbox', 'textbox', 'combobox', 'listbox', 'tab',
        'switch', 'slider', 'spinbutton', 'searchbox'
    }
    
    def __init__(self, elements: List[ElementInfo], page_info: Dict[str, Any]):
        self.elements = elements
        self.page_info = page_info
        self._interactive_elements: List[ElementInfo] = []
        self._text_content: str = ""
        
    @classmethod
    async def capture(cls, page: Page) -> 'DOMSnapshot':
        """
        Capture a DOM snapshot from the current page.
        
        Args:
            page: Playwright page instance
            
        Returns:
            DOMSnapshot instance
        """
        # Get page info
        page_info = {
            "url": page.url,
            "title": await page.title(),
            "viewport": page.viewport_size
        }
        
        # Execute JavaScript to extract DOM information
        elements_data = await page.evaluate('''
            () => {
                const elements = [];
                const interactiveTags = new Set(['a', 'button', 'input', 'select', 'textarea', 'details', 'summary']);
                const interactiveRoles = new Set(['button', 'link', 'menuitem', 'option', 'radio', 'checkbox', 'textbox', 'combobox', 'tab', 'switch']);
                
                function isVisible(el) {
                    const style = window.getComputedStyle(el);
                    const rect = el.getBoundingClientRect();
                    return style.display !== 'none' && 
                           style.visibility !== 'hidden' && 
                           style.opacity !== '0' &&
                           rect.width > 0 && 
                           rect.height > 0;
                }
                
                function isInteractive(el) {
                    const tag = el.tagName.toLowerCase();
                    const role = el.getAttribute('role');
                    const tabIndex = el.getAttribute('tabindex');
                    const onclick = el.getAttribute('onclick') || el.onclick;
                    
                    return interactiveTags.has(tag) || 
                           interactiveRoles.has(role) ||
                           (tabIndex !== null && tabIndex !== '-1') ||
                           onclick !== null ||
                           el.hasAttribute('data-action') ||
                           el.hasAttribute('data-click');
                }
                
                function getElementText(el) {
                    // Get direct text, not text from children
                    let text = '';
                    for (const node of el.childNodes) {
                        if (node.nodeType === Node.TEXT_NODE) {
                            text += node.textContent.trim() + ' ';
                        }
                    }
                    // Fallback to innerText for some elements
                    if (!text.trim() && ['button', 'a', 'label'].includes(el.tagName.toLowerCase())) {
                        text = el.innerText;
                    }
                    return text.trim().substring(0, 200);
                }
                
                function extractElement(el, index) {
                    const tag = el.tagName.toLowerCase();
                    const rect = el.getBoundingClientRect();
                    
                    return {
                        tag: tag,
                        id: el.id || '',
                        classes: Array.from(el.classList),
                        text: getElementText(el),
                        href: el.href || '',
                        src: el.src || '',
                        placeholder: el.placeholder || '',
                        value: el.value || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                        role: el.getAttribute('role') || '',
                        type: el.type || '',
                        name: el.name || '',
                        isVisible: isVisible(el),
                        isEnabled: !el.disabled,
                        isInteractive: isInteractive(el),
                        boundingBox: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height
                        },
                        index: index
                    };
                }
                
                // Get all potentially interesting elements
                const selector = 'a, button, input, select, textarea, [role], [onclick], [tabindex], h1, h2, h3, h4, h5, h6, p, li, td, th, label, span, div';
                const allElements = document.querySelectorAll(selector);
                
                let index = 0;
                for (const el of allElements) {
                    // Skip hidden elements and very small elements
                    const rect = el.getBoundingClientRect();
                    if (rect.width < 5 || rect.height < 5) continue;
                    
                    // Skip elements outside viewport (with some margin)
                    if (rect.bottom < -100 || rect.top > window.innerHeight + 100) continue;
                    
                    const data = extractElement(el, index);
                    
                    // Only include if it has meaningful content or is interactive
                    if (data.isInteractive || data.text || data.placeholder || data.ariaLabel) {
                        elements.push(data);
                        index++;
                    }
                    
                    // Limit total elements
                    if (elements.length >= 200) break;
                }
                
                return elements;
            }
        ''')
        
        # Convert to ElementInfo objects
        elements = []
        for el_data in elements_data:
            element = ElementInfo(
                tag=el_data['tag'],
                id=el_data['id'],
                classes=el_data['classes'],
                text=el_data['text'],
                href=el_data['href'],
                src=el_data['src'],
                placeholder=el_data['placeholder'],
                value=el_data['value'],
                aria_label=el_data['ariaLabel'],
                role=el_data['role'],
                type=el_data['type'],
                name=el_data['name'],
                is_visible=el_data['isVisible'],
                is_enabled=el_data['isEnabled'],
                is_interactive=el_data['isInteractive'],
                bounding_box=el_data['boundingBox'],
                index=el_data['index']
            )
            elements.append(element)
        
        return cls(elements, page_info)
    
    def get_interactive_elements(self) -> List[Dict[str, Any]]:
        """Get only interactive elements for action selection."""
        if not self._interactive_elements:
            self._interactive_elements = [
                el for el in self.elements
                if el.is_interactive and el.is_visible
            ]
        
        return [el.to_dict() for el in self._interactive_elements]
    
    def get_visible_text(self, max_length: int = 5000) -> str:
        """Get visible text content from the page."""
        if not self._text_content:
            texts = []
            for el in self.elements:
                if el.is_visible and el.text:
                    texts.append(el.text)
            self._text_content = " ".join(texts)
        
        return self._text_content[:max_length]
    
    def to_simplified_json(self) -> Dict[str, Any]:
        """
        Convert to simplified JSON representation for LLM.
        """
        return {
            "page": self.page_info,
            "element_count": len(self.elements),
            "interactive_count": len(self.get_interactive_elements()),
            "elements_summary": self._create_elements_summary()
        }
    
    def _create_elements_summary(self) -> str:
        """Create a text summary of key elements."""
        lines = []
        
        # Group by type
        inputs = [el for el in self.elements if el.tag == 'input' and el.is_visible]
        buttons = [el for el in self.elements if el.tag == 'button' and el.is_visible]
        links = [el for el in self.elements if el.tag == 'a' and el.is_visible]
        
        if inputs:
            lines.append(f"Input fields ({len(inputs)}):")
            for inp in inputs[:10]:
                label = inp.placeholder or inp.aria_label or inp.name or inp.type
                lines.append(f"  - {label}")
        
        if buttons:
            lines.append(f"Buttons ({len(buttons)}):")
            for btn in buttons[:10]:
                label = btn.text or btn.aria_label or "unnamed"
                lines.append(f"  - {label}")
        
        if links:
            lines.append(f"Links ({len(links)}):")
            for link in links[:10]:
                label = link.text or link.aria_label or link.href
                lines.append(f"  - {label[:50]}")
        
        return "\n".join(lines)
    
    def find_element(
        self,
        selector: str = None,
        text: str = None,
        role: str = None,
        index: int = None
    ) -> Optional[ElementInfo]:
        """
        Find an element matching the criteria.
        
        Args:
            selector: CSS-like selector (id or class)
            text: Text content to match
            role: ARIA role to match
            index: Element index from the list
            
        Returns:
            Matching ElementInfo or None
        """
        for element in self.elements:
            if index is not None and element.index == index:
                return element
            
            if selector:
                if selector.startswith('#') and element.id == selector[1:]:
                    return element
                if selector.startswith('.') and selector[1:] in element.classes:
                    return element
                    
            if text and text.lower() in element.text.lower():
                return element
                
            if role and element.role == role:
                return element
        
        return None
    
    def find_elements_by_text(self, text: str, exact: bool = False) -> List[ElementInfo]:
        """Find all elements containing the given text."""
        results = []
        text_lower = text.lower()
        
        for element in self.elements:
            if not element.is_visible:
                continue
                
            element_text = element.text.lower()
            if exact:
                if element_text == text_lower:
                    results.append(element)
            else:
                if text_lower in element_text:
                    results.append(element)
        
        return results
    
    def get_form_fields(self) -> List[ElementInfo]:
        """Get all form input fields."""
        form_tags = {'input', 'select', 'textarea'}
        return [
            el for el in self.elements
            if el.tag in form_tags and el.is_visible
        ]
