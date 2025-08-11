import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Union, Tuple

from RL2.workers.tools.base import BaseTool
from RL2.workers.tools.schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    WebSearch tool: returns short previews only for candidate results.
    """

    def __init__(
        self,
        config: dict,
        tool_schema: OpenAIFunctionToolSchema,
    ):
        super().__init__(config, tool_schema)
        cfg = config or {}
        self.search_url = cfg.get("search_url", "http://localhost:10000/search")
        self.timeout = cfg.get("timeout", 30)
        self.top_k = cfg.get("top_k", 3)
        self.use_reranker = cfg.get("use_reranker", False)
        self.preview_chars = cfg.get("preview_chars", 256)

    def _get_default_name(self) -> str:
        return "web_search"

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
        """Execute the web search tool following VERL interface."""
        try:
            query = (parameters.get("query") or "").strip()
            if not query:
                response = self._format_response(success=False, error="Empty search query")
                return response, 0.0, {"error": "empty_query"}

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.search_url,
                    json={
                        "query": query,
                        "top_k": self.top_k,
                        "preview_char": self.preview_chars,
                    },
                ) as http_response:
                    http_response.raise_for_status()
                    raw = await http_response.json()

            # Normalize to short preview entries only
            normalized_results = []
            for idx, item in enumerate(raw.get("results", [])):
                url = item.get("url")
                metadata = item.get("metadata", {})
                title = metadata.get("title") or metadata.get("paper_title") or ""
                preview = (item.get("preview") or "")[: self.preview_chars]
                normalized_results.append(f"Index: {idx}\nURL: {url}\nTitle: {title}\nPreview: {preview}")
            normalized_results = "\n\n".join(normalized_results)

            response = self._format_response(success=True, data=normalized_results)
            reward = 1.0 if normalized_results else 0.0  # Basic reward based on results
            metrics = {
                "query": query,
                "num_results": len(raw.get("results", [])),
            }
            
            return response, reward, metrics

        except asyncio.TimeoutError:
            response = self._format_response(success=False, error=f"Search timed out after {self.timeout} seconds")
            return response, 0.0, {"error": "timeout"}
        except aiohttp.ClientError as e:
            response = self._format_response(success=False, error=f"Search API error: {str(e)}")
            return response, 0.0, {"error": "api_error", "details": str(e)}
        except Exception as e:
            logger.exception("web_search failed")
            response = self._format_response(success=False, error=str(e))
            return response, 0.0, {"error": "unexpected", "details": str(e)}


if __name__ == "__main__":
    from omegaconf import OmegaConf

    tool_config = OmegaConf.load("RL2/workers/tools/config/config_example.yaml").tools.tool_instances.web_search
    tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
    tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
    tool = WebSearchTool(tool_config, tool_schema)
    print(tool.get_openai_tool_schema())
    
    async def test():
        result = await tool.execute("test_instance", {"query": "Big Little Lies Season 2 number of episodes"})
        print(f"Response: {result[0]}")
        print(f"Reward: {result[1]}")
        print(f"Metrics: {result[2]}")
    
    asyncio.run(test())