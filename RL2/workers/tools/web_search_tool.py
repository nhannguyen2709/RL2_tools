import json
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Union

from RL2.workers.tools.base import BaseTool

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    WebSearch tool: returns short previews only for candidate results.
    """

    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        tool_schema: Dict[str, Any] | None = None,
    ):
        super().__init__(config, tool_schema)
        cfg = config or {}
        self.search_url = cfg.get("search_url", "http://localhost:10000/search")
        self.timeout = cfg.get("timeout", 30)
        self.top_k = cfg.get("top_k", 3)
        self.use_reranker = cfg.get("use_reranker", False)
        self.preview_chars = cfg.get("preview_chars", 256)

    @property
    def name(self) -> str:
        return "web_search"

    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web and return short previews of top results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "use_reranker": {
                            "type": "boolean",
                            "description": "Whether to use reranking",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, function_args: Union[str, Dict[str, Any]], **kwargs) -> str:
        try:
            args = json.loads(function_args) if isinstance(function_args, str) else function_args
            query = (args.get("query") or "").strip()
            if not query:
                return json.dumps({"error": "Empty search query"})

            top_k = int(args.get("top_k", self.top_k))
            use_reranker = bool(args.get("use_reranker", self.use_reranker))

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.search_url,
                    json={
                        "query": query,
                        "top_k": top_k,
                        "use_reranker": use_reranker,
                        "preview_char": self.preview_chars,
                    },
                ) as response:
                    response.raise_for_status()
                    raw = await response.json()

            # Normalize to short preview entries only
            normalized_results = []
            for idx, item in enumerate(raw.get("results", [])):
                url = item.get("url")
                metadata = item.get("metadata", {})
                title = metadata.get("title") or metadata.get("paper_title") or ""
                preview = (item.get("preview") or "")[: self.preview_chars]
                normalized_results.append(f"Index: {idx}\nURL: {url}\nTitle: {title}\nPreview: {preview}")
            normalized_results = "\n\n".join(normalized_results)

            return json.dumps({"data": normalized_results})

        except asyncio.TimeoutError:
            return json.dumps({"error": f"Search timed out after {self.timeout} seconds"})
        except aiohttp.ClientError as e:
            return json.dumps({"error": f"Search API error: {str(e)}"})
        except Exception as e:
            logger.exception("web_search failed")
            return json.dumps({"error": str(e)})


if __name__ == "__main__":
    tool = WebSearchTool()
    result = asyncio.run(tool.execute({"query": "Big Little Lies Season 2 number of episodes"}))
    print(result)