"""
Browser Agent - Main agent loop with reasoning and action execution.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .planner import Planner
from .memory import Memory
from .tools import ToolRegistry, ActionResult
from browser.playwright import BrowserController
from browser.dom_parser import DOMSnapshot
from browser.actions import BrowserActions

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class StepResult:
    step_number: int
    thought: str
    action: str
    action_params: Dict[str, Any]
    result: ActionResult
    screenshot_path: Optional[str] = None


class BrowserAgent:
    """
    Autonomous browser agent that uses LLM for reasoning and planning.
    
    The agent follows a loop:
    1. Observe (get DOM snapshot + screenshot)
    2. Think (LLM reasoning about current state)
    3. Plan (decide next action)
    4. Act (execute browser action)
    5. Reflect (evaluate result, adjust if needed)
    """
    
    def __init__(
        self,
        browser: BrowserController,
        model: str = "gpt-4",
        max_steps: int = 50
    ):
        self.browser = browser
        self.model = model
        self.max_steps = max_steps
        
        self.memory = Memory()
        self.planner = Planner(model=model)
        self.actions = BrowserActions(browser)
        self.tools = ToolRegistry()
        
        self.state = AgentState.IDLE
        self.current_task: Optional[str] = None
        self.step_history: list[StepResult] = []
        
    async def run(self, task: str) -> Dict[str, Any]:
        """
        Execute the main agent loop for a given task.
        
        Args:
            task: Natural language description of the task to complete
            
        Returns:
            Dictionary with task result and execution history
        """
        self.current_task = task
        self.state = AgentState.THINKING
        self.memory.set_goal(task)
        
        logger.info(f"Starting task: {task}")
        
        step = 0
        while step < self.max_steps:
            step += 1
            logger.info(f"=== Step {step}/{self.max_steps} ===")
            
            try:
                # 1. Observe current state
                observation = await self._observe()
                self.memory.add_observation(observation)
                
                # 2. Think and Plan
                self.state = AgentState.THINKING
                plan = await self.planner.plan(
                    task=task,
                    observation=observation,
                    memory=self.memory.get_context(),
                    available_tools=self.tools.get_schema()
                )
                
                logger.info(f"Thought: {plan.thought}")
                logger.info(f"Action: {plan.action} with params: {plan.params}")
                
                # Check if task is completed
                if plan.action == "complete":
                    self.state = AgentState.COMPLETED
                    logger.info("Task marked as completed by agent")
                    return self._create_result(success=True, reason=plan.thought)
                
                # 3. Execute action
                self.state = AgentState.ACTING
                result = await self._execute_action(plan.action, plan.params)
                
                # Record step
                step_result = StepResult(
                    step_number=step,
                    thought=plan.thought,
                    action=plan.action,
                    action_params=plan.params,
                    result=result
                )
                self.step_history.append(step_result)
                self.memory.add_action(plan.action, plan.params, result)
                
                # Small delay between steps
                await asyncio.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Error in step {step}: {e}", exc_info=True)
                self.memory.add_error(str(e))
                
                if self._should_abort(e):
                    self.state = AgentState.FAILED
                    return self._create_result(success=False, reason=str(e))
        
        # Max steps reached
        self.state = AgentState.FAILED
        logger.warning(f"Max steps ({self.max_steps}) reached without completing task")
        return self._create_result(success=False, reason="Max steps reached")
    
    async def _observe(self) -> Dict[str, Any]:
        """Capture current browser state."""
        dom_snapshot = await DOMSnapshot.capture(self.browser.page)
        screenshot = await self.browser.screenshot()
        url = self.browser.page.url
        title = await self.browser.page.title()
        
        return {
            "url": url,
            "title": title,
            "dom": dom_snapshot.to_simplified_json(),
            "interactive_elements": dom_snapshot.get_interactive_elements(),
            "visible_text": dom_snapshot.get_visible_text(),
            "screenshot": screenshot
        }
    
    async def _execute_action(self, action: str, params: Dict[str, Any]) -> ActionResult:
        """Execute a browser action."""
        action_method = getattr(self.actions, action, None)
        
        if action_method is None:
            return ActionResult(
                success=False,
                message=f"Unknown action: {action}",
                data=None
            )
        
        try:
            result = await action_method(**params)
            return result
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Action failed: {str(e)}",
                data=None
            )
    
    def _should_abort(self, error: Exception) -> bool:
        """Determine if the agent should abort due to an error."""
        critical_errors = [
            "browser disconnected",
            "context destroyed",
            "target closed"
        ]
        return any(err in str(error).lower() for err in critical_errors)
    
    def _create_result(self, success: bool, reason: str) -> Dict[str, Any]:
        """Create the final result dictionary."""
        return {
            "success": success,
            "reason": reason,
            "task": self.current_task,
            "steps_taken": len(self.step_history),
            "final_url": self.browser.page.url if self.browser.page else None,
            "history": [
                {
                    "step": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "params": s.action_params,
                    "success": s.result.success
                }
                for s in self.step_history
            ]
        }
