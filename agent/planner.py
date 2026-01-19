"""
Planner - LLM-based reasoning and action planning.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

from .prompt import SystemPrompts

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    """Represents a planned action with reasoning."""
    thought: str  # Chain of thought reasoning
    action: str   # Action to execute
    params: Dict[str, Any]  # Action parameters
    confidence: float  # Confidence score 0-1
    alternatives: List[Dict[str, Any]]  # Alternative actions considered


class Planner:
    """
    LLM-based planner that decides the next action based on:
    - Current task/goal
    - Browser observation (DOM, screenshot)
    - Action history from memory
    - Available tools/actions
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = AsyncOpenAI()
        logger.info(f"Planner initialized with model: {self.model}")
        
    async def plan(
        self,
        task: str,
        observation: Dict[str, Any],
        memory: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> Plan:
        """
        Generate the next action plan based on current state.
        
        Args:
            task: The user's task description
            observation: Current browser state
            memory: Agent's memory context
            available_tools: List of available actions
            
        Returns:
            Plan object with thought process and action
        """
        
        # Build the planning prompt
        system_prompt = SystemPrompts.PLANNER_SYSTEM
        
        user_prompt = self._build_user_prompt(
            task=task,
            observation=observation,
            memory=memory,
            available_tools=available_tools
        )
        
        # Call LLM with function calling
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=self._get_planning_tools(available_tools),
            tool_choice="required",
            temperature=0.2
        )
        
        # Parse response
        message = response.choices[0].message
        
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            action = tool_call.function.name
            params = json.loads(tool_call.function.arguments)
            
            # Extract thought from the content or params
            thought = params.pop("thought", "") or message.content or "Proceeding with action"
            confidence = params.pop("confidence", 0.8)
            
            return Plan(
                thought=thought,
                action=action,
                params=params,
                confidence=confidence,
                alternatives=[]
            )
        
        # Fallback if no tool call (shouldn't happen with tool_choice="required")
        return Plan(
            thought=message.content or "Unable to determine action",
            action="wait",
            params={"duration": 1},
            confidence=0.5,
            alternatives=[]
        )
    
    def _build_user_prompt(
        self,
        task: str,
        observation: Dict[str, Any],
        memory: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> str:
        """Build the user prompt for the planner."""
        
        # Summarize interactive elements (limit to avoid token overflow)
        elements = observation.get("interactive_elements", [])[:30]
        elements_str = "\n".join([
            f"[{i}] {el.get('tag', 'unknown')} - {el.get('text', '')[:50]} "
            f"(id={el.get('id', '')}, class={el.get('class', '')[:30]})"
            for i, el in enumerate(elements)
        ])
        
        # Recent actions from memory
        recent_actions = memory.get("recent_actions", [])[-5:]
        actions_str = "\n".join([
            f"- {a['action']}: {a.get('result', 'done')}"
            for a in recent_actions
        ])
        
        prompt = f"""
## Current Task
{task}

## Current Page
URL: {observation.get('url', 'unknown')}
Title: {observation.get('title', 'unknown')}

## Interactive Elements on Page
{elements_str}

## Visible Text (truncated)
{observation.get('visible_text', '')[:1000]}

## Recent Actions Taken
{actions_str if actions_str else 'No actions yet'}

## Memory Notes
{memory.get('notes', 'None')}

Based on the above, decide the SINGLE BEST next action to progress toward completing the task.
Think step by step about what you see and what action would be most effective.
"""
        return prompt
    
    def _get_planning_tools(self, available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert available tools to OpenAI function calling format."""
        
        # Add thought and confidence to each tool
        tools = []
        for tool in available_tools:
            enhanced_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thought": {
                                "type": "string",
                                "description": "Your reasoning for choosing this action"
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score 0-1 for this action"
                            },
                            **tool.get("parameters", {}).get("properties", {})
                        },
                        "required": ["thought"] + tool.get("parameters", {}).get("required", [])
                    }
                }
            }
            tools.append(enhanced_tool)
        
        return tools


class ReActPlanner(Planner):
    """
    ReAct-style planner that explicitly separates:
    - Reasoning (Thought)
    - Action selection
    - Observation processing
    """
    
    async def plan(
        self,
        task: str,
        observation: Dict[str, Any],
        memory: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> Plan:
        """ReAct-style planning with explicit reasoning steps."""
        
        # Step 1: Generate thought/reasoning
        thought = await self._generate_thought(task, observation, memory)
        
        # Step 2: Select action based on thought
        action, params = await self._select_action(
            thought=thought,
            observation=observation,
            available_tools=available_tools
        )
        
        return Plan(
            thought=thought,
            action=action,
            params=params,
            confidence=0.8,
            alternatives=[]
        )
    
    async def _generate_thought(
        self,
        task: str,
        observation: Dict[str, Any],
        memory: Dict[str, Any]
    ) -> str:
        """Generate reasoning about current state."""
        
        prompt = f"""
Task: {task}
Current URL: {observation.get('url')}
Current Page Title: {observation.get('title')}

Previous actions: {memory.get('recent_actions', [])[-3:]}

Think step by step:
1. What is the current state?
2. What progress has been made toward the task?
3. What should be done next?
4. What specific element or action would help?

Provide your reasoning:
"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are reasoning about a browser automation task."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    
    async def _select_action(
        self,
        thought: str,
        observation: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> tuple[str, Dict[str, Any]]:
        """Select the best action based on reasoning."""
        
        elements = observation.get("interactive_elements", [])[:20]
        
        prompt = f"""
Based on this reasoning:
{thought}

And these available elements:
{json.dumps(elements, indent=2)}

Select the best action and provide parameters.
Respond with JSON: {{"action": "action_name", "params": {{...}}}}
"""
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Select an action. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("action", "wait"), result.get("params", {})
