"""
Tool Schema - Function calling definitions for the LLM.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str
    data: Optional[Any] = None


class ToolRegistry:
    """
    Registry of available tools/actions for the agent.
    
    Provides OpenAI function calling compatible schemas.
    """
    
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()
        
    def _register_default_tools(self):
        """Register all default browser actions."""
        
        # Navigation tools
        self.register(
            name="navigate",
            description="Navigate to a URL. Use this to go to a specific webpage.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to (e.g., 'https://google.com' or 'google.com')"
                    }
                },
                "required": ["url"]
            }
        )
        
        self.register(
            name="go_back",
            description="Navigate back to the previous page in browser history.",
            parameters={
                "type": "object",
                "properties": {}
            }
        )
        
        self.register(
            name="refresh",
            description="Refresh/reload the current page.",
            parameters={
                "type": "object",
                "properties": {}
            }
        )
        
        # Click tools
        self.register(
            name="click",
            description="Click on an element. Use selector, text content, or index to identify the element.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element (e.g., '#submit-btn', '.nav-link', 'button[type=submit]')"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content of the element to click (e.g., 'Sign In', 'Submit')"
                    },
                    "index": {
                        "type": "integer",
                        "description": "Index of the element from the interactive elements list"
                    }
                }
            }
        )
        
        self.register(
            name="hover",
            description="Hover over an element to reveal dropdown menus or tooltips.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content of the element"
                    }
                }
            }
        )
        
        # Type tools
        self.register(
            name="type",
            description="Type text into an input field. Simulates real typing with keystrokes.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to type"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the input field"
                    },
                    "element_text": {
                        "type": "string",
                        "description": "Placeholder or label text of the input field"
                    },
                    "clear_first": {
                        "type": "boolean",
                        "description": "Whether to clear existing content before typing (default: true)"
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": "Whether to press Enter after typing (useful for search)"
                    }
                },
                "required": ["text"]
            }
        )
        
        self.register(
            name="fill",
            description="Fill an input field instantly (faster than type, no keystroke simulation).",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to fill"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the input field"
                    },
                    "element_text": {
                        "type": "string",
                        "description": "Placeholder or label text of the input field"
                    }
                },
                "required": ["text"]
            }
        )
        
        self.register(
            name="press_key",
            description="Press a keyboard key (e.g., Enter, Tab, Escape, arrow keys).",
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to press (e.g., 'Enter', 'Tab', 'Escape', 'ArrowDown', 'Backspace')"
                    },
                    "modifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Modifier keys to hold (e.g., ['Control'], ['Shift', 'Control'])"
                    }
                },
                "required": ["key"]
            }
        )
        
        # Scroll tools
        self.register(
            name="scroll",
            description="Scroll the page up or down to see more content.",
            parameters={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right"],
                        "description": "Direction to scroll"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Pixels to scroll (default: 500)"
                    }
                },
                "required": ["direction"]
            }
        )
        
        self.register(
            name="scroll_to_element",
            description="Scroll to bring a specific element into view.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the element"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text content of the element"
                    }
                }
            }
        )
        
        # Wait tools
        self.register(
            name="wait",
            description="Wait for an element to appear, disappear, or just wait for some time.",
            parameters={
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "number",
                        "description": "Seconds to wait (use this for simple delays)"
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector to wait for"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to wait to appear on page"
                    },
                    "state": {
                        "type": "string",
                        "enum": ["visible", "hidden", "attached", "detached"],
                        "description": "State to wait for (default: visible)"
                    }
                }
            }
        )
        
        # Extract tools
        self.register(
            name="extract",
            description="Extract text or data from elements on the page.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for element(s) to extract from"
                    },
                    "attribute": {
                        "type": "string",
                        "description": "HTML attribute to extract (e.g., 'href', 'src'). Defaults to text content."
                    },
                    "all_matches": {
                        "type": "boolean",
                        "description": "Whether to extract from all matching elements (default: false)"
                    }
                },
                "required": ["selector"]
            }
        )
        
        # Select tools
        self.register(
            name="select",
            description="Select an option from a dropdown menu.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "CSS selector for the select element"
                    },
                    "value": {
                        "type": "string",
                        "description": "Option value to select"
                    },
                    "label": {
                        "type": "string",
                        "description": "Option visible text to select"
                    },
                    "index": {
                        "type": "integer",
                        "description": "Option index to select (0-based)"
                    }
                },
                "required": ["selector"]
            }
        )
        
        # Completion tool
        self.register(
            name="complete",
            description="Mark the task as completed. Use this when the goal has been achieved.",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Explanation of what was accomplished"
                    }
                },
                "required": ["reason"]
            }
        )
        
    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any]
    ):
        """Register a new tool."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters
        }
        
    def get_schema(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return list(self._tools.values())
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific tool's schema."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI tools format for API calls."""
        return [
            {
                "type": "function",
                "function": tool
            }
            for tool in self._tools.values()
        ]


# Pre-defined tool sets for different use cases
TOOL_SETS = {
    "navigation": ["navigate", "go_back", "refresh", "scroll"],
    "interaction": ["click", "type", "fill", "press_key", "select", "hover"],
    "observation": ["wait", "extract"],
    "control": ["complete"],
    "full": None  # All tools
}


def get_tool_set(set_name: str) -> List[str]:
    """Get a predefined set of tools."""
    tool_set = TOOL_SETS.get(set_name)
    if tool_set is None:
        return ToolRegistry().list_tools()
    return tool_set
