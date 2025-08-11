import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Union, Tuple

from RL2.workers.tools.base import BaseTool
from RL2.workers.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)


class WebVisitTool(BaseTool):
    """
    WebVisit tool: fetches a URL and returns full text content/context.
    """

    def __init__(
        self,
        config: dict,
        tool_schema: OpenAIFunctionToolSchema,
    ):
        super().__init__(config, tool_schema)
        cfg = config or {}
        self.visit_url = cfg.get("visit_url", "http://localhost:10000/visit")
        self.timeout = cfg.get("timeout", 30)

    def _get_default_name(self) -> str:
        return "web_visit"

    def _format_response(self, success: bool = True, data: str = "", error: str = "") -> str:
        """
        Standardize web tool response format for consistent handling.
        
        Args:
            success: Whether the tool execution was successful
            data: The successful result data (used when success=True)
            error: The error message (used when success=False)
            
        Returns:
            JSON string with standardized format:
            {
                "success": bool,
                "tool_name": str,
                "data": str,        # Only when success=True
                "error": str        # Only when success=False
            }
        """
        response = {
            "success": success,
            "tool_name": self.name
        }
        
        if success:
            response["data"] = data
        else:
            response["error"] = error
            
        return json.dumps(response)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[str, float, Dict[str, Any]]:
        """Execute the web visit tool following VERL interface."""
        try:
            url = (parameters.get("url") or "").strip()
            if not url:
                response = self._format_response(success=False, error="Empty URL")
                return response, 0.0, {"error": "empty_url"}

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.visit_url, json={"url": url}) as http_response:
                    http_response.raise_for_status()
                    raw = await http_response.json()

            # Extract content from the raw response and format consistently
            response = self._format_response(success=True, data=raw["data"])
            reward = 1.0 if raw.get("data") else 0.0  # Basic reward based on content retrieval
            metrics = {
                "url": url,
                "content_length": len(raw.get("data", "")),
                "status": "success"
            }
            
            return response, reward, metrics

        except asyncio.TimeoutError:
            response = self._format_response(success=False, error=f"Visit timed out after {self.timeout} seconds")
            return response, 0.0, {"error": "timeout"}
        except aiohttp.ClientError as e:
            response = self._format_response(success=False, error=f"Visit API error: {str(e)}")
            return response, 0.0, {"error": "api_error", "details": str(e)}
        except Exception as e:
            logger.exception("web_visit failed")
            response = self._format_response(success=False, error=str(e))
            return response, 0.0, {"error": "unexpected", "details": str(e)}


if __name__ == "__main__":
    from omegaconf import OmegaConf

    tool_config = OmegaConf.load("RL2/workers/tools/config/config_example.yaml").tools.tool_instances.web_visit
    tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
    tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
    tool = WebVisitTool(tool_config, tool_schema)
    print(tool.get_openai_tool_schema())
    
    async def test():
        result = await tool.execute("test_instance", {"url": "https://en.wikipedia.org/wiki/Big%20Little%20Lies%20%28TV%20series%29"})
        print(f"Response: {result[0]}")
        print(f"Reward: {result[1]}")
        print(f"Metrics: {result[2]}")
    
    asyncio.run(test())