from abc import ABC, abstractmethod
from typing import Dict, Any, Union


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, config: Dict[str, Any] = None, tool_schema: Dict[str, Any] = None):
        """
        Initialize the tool.
        
        Args:
            config: Tool-specific configuration parameters
            tool_schema: OpenAI-compatible tool schema
        """
        self.config = config or {}
        self.tool_schema = tool_schema

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier."""
        pass

    @abstractmethod
    async def execute(self, function_args: Union[str, Dict[str, Any]], **kwargs) -> str:
        """
        Execute the tool with given arguments.
        
        Args:
            function_args: Function arguments as dict or JSON string
            **kwargs: Additional keyword arguments
            
        Returns:
            Tool execution result as string
        """
        pass

    def get_openai_tool_schema(self) -> Dict[str, Any]:
        """Get OpenAI tool schema for function call parsing."""
        if self.tool_schema:
            return self.tool_schema
        else:
            # Fallback to class-defined schema if available
            return getattr(self, '_default_schema', {})