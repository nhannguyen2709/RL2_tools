from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, Tuple
from uuid import uuid4


class BaseTool(ABC):
    """Base class for all tools following VERL interface."""

    def __init__(self, config: Dict[str, Any] = None, tool_schema: Any = None):
        """
        Initialize the tool.
        
        Args:
            config: Tool-specific configuration parameters
            tool_schema: OpenAI-compatible tool schema (can be dict or pydantic model)
        """
        self.config = config or {}
        self.tool_schema = tool_schema
        # Extract name from schema if available
        if hasattr(tool_schema, 'function') and hasattr(tool_schema.function, 'name'):
            self._name = tool_schema.function.name
        elif isinstance(tool_schema, dict) and 'function' in tool_schema:
            self._name = tool_schema['function']['name']
        else:
            self._name = None

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return self._name or self._get_default_name()

    @abstractmethod
    def _get_default_name(self) -> str:
        """Get default name for the tool (fallback when schema doesn't provide name)."""
        pass

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
        """
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    @abstractmethod
    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict[str, Any]]:
        """Execute the tool.

        Args:
            instance_id: The instance id of the tool.
            parameters: The parameters of the tool as a dictionary.

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        pass

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        pass

    def get_openai_tool_schema(self):
        """Get OpenAI tool schema for function call parsing."""
        if self.tool_schema:
            # Handle both pydantic models and dictionaries
            if hasattr(self.tool_schema, 'model_dump'):
                return self.tool_schema
            elif hasattr(self.tool_schema, 'dict'):
                return self.tool_schema
            else:
                return self.tool_schema
        else:
            # Fallback to class-defined schema if available
            return getattr(self, '_default_schema', {})