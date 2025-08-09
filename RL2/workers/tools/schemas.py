from typing import Any, Dict, Literal
from pydantic import BaseModel
import json


class OpenAIFunctionParsedSchema(BaseModel):
    """Parsed function schema from SGLang parser."""
    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""
    name: str
    arguments: Dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(parsed_schema: OpenAIFunctionParsedSchema) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    """The tool call in OpenAI format."""
    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema


class OpenAIFunctionToolSchema(BaseModel):
    """OpenAI function tool schema for configuration."""
    type: Literal["function"] = "function"
    function: Dict[str, Any]

    def get_openai_tool_schema(self) -> Dict[str, Any]:
        """Get the OpenAI tool schema format."""
        return self.model_dump()