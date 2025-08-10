import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Union

from RL2.workers.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WebVisitTool(BaseTool):
    """
    WebVisit tool: fetches a URL and returns full text content/context.
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        tool_schema: Dict[str, Any] | None = None,
    ):
        super().__init__(config, tool_schema)
        cfg = config or {}
        self.visit_url = cfg.get("visit_url", "http://localhost:10000/visit")
        self.timeout = cfg.get("timeout", 30)

    @property
    def name(self) -> str:
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

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_visit",
                "description": "Visit a URL and return full page content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"},
                    },
                    "required": ["url"],
                },
            },
        }

    async def execute(self, function_args: Union[str, Dict[str, Any]], **kwargs) -> str:
        try:
            args = json.loads(function_args) if isinstance(function_args, str) else function_args
            url = (args.get("url") or "").strip()
            if not url:
                return self._format_response(success=False, error="Empty URL")

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.visit_url, json={"url": url}) as response:
                    response.raise_for_status()
                    raw = await response.json()

            # Extract content from the raw response and format consistently
            content = raw.get("content", "") if isinstance(raw, dict) else str(raw)
            return self._format_response(success=True, data=content)

        except asyncio.TimeoutError:
            return self._format_response(success=False, error=f"Visit timed out after {self.timeout} seconds")
        except aiohttp.ClientError as e:
            return self._format_response(success=False, error=f"Visit API error: {str(e)}")
        except Exception as e:
            logger.exception("web_visit failed")
            return self._format_response(success=False, error=str(e))


if __name__ == "__main__":
    tool = WebVisitTool()
    result = asyncio.run(tool.execute({"url": "https://en.wikipedia.org/wiki/Big%20Little%20Lies%20%28TV%20series%29"}))
    import pdb; pdb.set_trace()
    print(result)