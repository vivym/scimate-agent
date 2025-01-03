from typing import Any, Literal

from scimate_agent.plugins import Plugin, register_plugin

try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError(
        "Please install tavily to use this plugin: `pip install tavily-python`"
    )


@register_plugin
class TavilySearch(Plugin):
    client: TavilyClient | None = None

    def __call__(
        self,
        query: str,
        include_url: bool = False,
        search_depth: Literal["basic", "advanced"] = "advanced",
        topic: Literal["general", "news"] = "general",
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        if self.client is None:
            self.client = TavilyClient(api_key=self.config.get("tavily_api_key", None))

        resp = self.client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
        )

        results = [
            {
                "title": result["title"],
                "content": result["content"],
                "score": result["score"],
            }
            for result in resp["results"]
        ]

        if include_url:
            for result, original_result in zip(results, resp["results"]):
                result["url"] = original_result["url"]

        return results
