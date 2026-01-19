"""
Memory - Short-term and long-term memory management for the agent.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Single memory item with metadata."""
    content: Any
    item_type: str  # observation, action, reflection, error
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5  # 0-1 scale
    tags: List[str] = field(default_factory=list)


@dataclass
class ActionMemory:
    """Memory of an executed action."""
    action: str
    params: Dict[str, Any]
    result: Any
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


class Memory:
    """
    Manages agent memory with both short-term (working) and long-term storage.
    
    Short-term memory: Recent observations, actions, and thoughts (limited capacity)
    Long-term memory: Important learnings, successful patterns, persistent notes
    """
    
    def __init__(
        self,
        short_term_capacity: int = 20,
        long_term_capacity: int = 100
    ):
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        
        # Short-term memory (FIFO queue)
        self.short_term: deque[MemoryItem] = deque(maxlen=short_term_capacity)
        
        # Long-term memory (important items)
        self.long_term: List[MemoryItem] = []
        
        # Structured storage
        self.goal: Optional[str] = None
        self.sub_goals: List[str] = []
        self.actions_history: List[ActionMemory] = []
        self.observations_cache: deque[Dict[str, Any]] = deque(maxlen=5)
        self.errors: List[str] = []
        self.notes: List[str] = []
        self.reflections: List[Dict[str, Any]] = []
        
        # Semantic index for retrieval (simplified)
        self.semantic_index: Dict[str, List[int]] = {}
        
    def set_goal(self, goal: str):
        """Set the main goal/task."""
        self.goal = goal
        self._add_to_short_term(
            content=goal,
            item_type="goal",
            importance=1.0,
            tags=["goal", "task"]
        )
        
    def add_sub_goal(self, sub_goal: str):
        """Add a sub-goal identified during planning."""
        self.sub_goals.append(sub_goal)
        self._add_to_short_term(
            content=sub_goal,
            item_type="sub_goal",
            importance=0.8,
            tags=["sub_goal"]
        )
        
    def add_observation(self, observation: Dict[str, Any]):
        """Add a browser observation to memory."""
        # Cache recent observations
        self.observations_cache.append(observation)
        
        # Add summarized version to short-term
        summary = {
            "url": observation.get("url"),
            "title": observation.get("title"),
            "element_count": len(observation.get("interactive_elements", []))
        }
        
        self._add_to_short_term(
            content=summary,
            item_type="observation",
            importance=0.5,
            tags=["observation", "page"]
        )
        
    def add_action(self, action: str, params: Dict[str, Any], result: Any):
        """Record an executed action."""
        action_memory = ActionMemory(
            action=action,
            params=params,
            result=result,
            success=getattr(result, 'success', True)
        )
        self.actions_history.append(action_memory)
        
        # Important failed actions should be remembered longer
        importance = 0.7 if action_memory.success else 0.9
        
        self._add_to_short_term(
            content={
                "action": action,
                "params": params,
                "success": action_memory.success
            },
            item_type="action",
            importance=importance,
            tags=["action", action]
        )
        
    def add_reflection(self, reflection: Any):
        """Add a reflection/self-correction note."""
        reflection_dict = {
            "content": reflection.content if hasattr(reflection, 'content') else str(reflection),
            "adjustment": getattr(reflection, 'adjustment', None),
            "timestamp": datetime.now().isoformat()
        }
        self.reflections.append(reflection_dict)
        
        self._add_to_short_term(
            content=reflection_dict,
            item_type="reflection",
            importance=0.85,
            tags=["reflection", "learning"]
        )
        
        # Reflections are usually important - consider for long-term
        if getattr(reflection, 'is_important', False):
            self._promote_to_long_term(
                content=reflection_dict,
                item_type="reflection",
                importance=0.9
            )
            
    def add_error(self, error: str):
        """Record an error for debugging."""
        self.errors.append(error)
        
        self._add_to_short_term(
            content=error,
            item_type="error",
            importance=0.8,
            tags=["error"]
        )
        
    def add_note(self, note: str, importance: float = 0.6):
        """Add a general note to memory."""
        self.notes.append(note)
        
        self._add_to_short_term(
            content=note,
            item_type="note",
            importance=importance,
            tags=["note"]
        )
        
    def get_context(self, max_items: int = 10) -> Dict[str, Any]:
        """
        Get memory context for the planner.
        Returns a summary suitable for LLM consumption.
        """
        # Get recent actions
        recent_actions = [
            {
                "action": a.action,
                "params": a.params,
                "success": a.success,
                "result": str(a.result)[:100] if a.result else None
            }
            for a in self.actions_history[-5:]
        ]
        
        # Get recent short-term items
        recent_items = list(self.short_term)[-max_items:]
        
        # Get important long-term items
        important_long_term = sorted(
            self.long_term,
            key=lambda x: x.importance,
            reverse=True
        )[:3]
        
        return {
            "goal": self.goal,
            "sub_goals": self.sub_goals,
            "recent_actions": recent_actions,
            "recent_observations": [
                {"url": o.get("url"), "title": o.get("title")}
                for o in list(self.observations_cache)[-3:]
            ],
            "notes": self.notes[-5:],
            "reflections": self.reflections[-3:],
            "errors": self.errors[-3:],
            "important_memories": [
                {"type": item.item_type, "content": str(item.content)[:200]}
                for item in important_long_term
            ]
        }
        
    def search(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """
        Search memory for relevant items.
        Simple keyword-based search (can be enhanced with embeddings).
        """
        query_lower = query.lower()
        results = []
        
        # Search short-term
        for item in self.short_term:
            content_str = str(item.content).lower()
            if query_lower in content_str or any(query_lower in tag for tag in item.tags):
                results.append(item)
                
        # Search long-term
        for item in self.long_term:
            content_str = str(item.content).lower()
            if query_lower in content_str or any(query_lower in tag for tag in item.tags):
                results.append(item)
        
        # Sort by importance and recency
        results.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        
        return results[:limit]
    
    def get_action_pattern(self, action_type: str) -> List[ActionMemory]:
        """Get history of a specific action type for pattern analysis."""
        return [a for a in self.actions_history if a.action == action_type]
    
    def get_success_rate(self, last_n: int = 10) -> float:
        """Calculate recent action success rate."""
        recent = self.actions_history[-last_n:]
        if not recent:
            return 1.0
        return sum(1 for a in recent if a.success) / len(recent)
    
    def clear_short_term(self):
        """Clear short-term memory (useful for new task)."""
        self.short_term.clear()
        self.observations_cache.clear()
        
    def _add_to_short_term(
        self,
        content: Any,
        item_type: str,
        importance: float = 0.5,
        tags: List[str] = None
    ):
        """Add item to short-term memory."""
        item = MemoryItem(
            content=content,
            item_type=item_type,
            importance=importance,
            tags=tags or []
        )
        self.short_term.append(item)
        
        # Index for retrieval
        for tag in item.tags:
            if tag not in self.semantic_index:
                self.semantic_index[tag] = []
            self.semantic_index[tag].append(len(self.short_term) - 1)
            
    def _promote_to_long_term(
        self,
        content: Any,
        item_type: str,
        importance: float = 0.7
    ):
        """Promote an important item to long-term memory."""
        if len(self.long_term) >= self.long_term_capacity:
            # Remove least important item
            self.long_term.sort(key=lambda x: x.importance)
            self.long_term.pop(0)
            
        item = MemoryItem(
            content=content,
            item_type=item_type,
            importance=importance
        )
        self.long_term.append(item)
        
    def export(self) -> Dict[str, Any]:
        """Export memory state for persistence."""
        return {
            "goal": self.goal,
            "sub_goals": self.sub_goals,
            "actions_history": [
                {
                    "action": a.action,
                    "params": a.params,
                    "success": a.success,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in self.actions_history
            ],
            "notes": self.notes,
            "reflections": self.reflections,
            "errors": self.errors
        }
        
    def import_state(self, state: Dict[str, Any]):
        """Import memory state from persistence."""
        self.goal = state.get("goal")
        self.sub_goals = state.get("sub_goals", [])
        self.notes = state.get("notes", [])
        self.reflections = state.get("reflections", [])
        self.errors = state.get("errors", [])
