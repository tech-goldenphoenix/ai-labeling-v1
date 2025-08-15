"""
Search agent for Enhanced RnD Assistant
"""
from typing import Dict, Any, List

from langchain_core.messages import AIMessage

from agents.base_agent import BaseAgent
from tools.search_tools import search_by_description_tool, search_products_with_filters_tool


class EnhancedSearchAgent(BaseAgent):
    """Agent to perform enhanced search operations"""

    def __init__(self):
        super().__init__(temperature=0)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced search with single collection approach"""
        query_type = state["query_type"]
        original_query = state["query"]

        # Generate optimized search queries based on query type
        search_queries = self._generate_search_queries(original_query, query_type)

        all_results = []
        for search_query in search_queries:
            # Use appropriate filters based on query type
            filters = self._get_filters_for_query_type(query_type)

            if filters:
                results = search_products_with_filters_tool.invoke({
                    "query": search_query,
                    "filters": filters
                })
            else:
                results = search_by_description_tool.invoke({"description": search_query})

            all_results.extend(results)

        # Remove duplicates and limit results
        unique_results = self._deduplicate_results(all_results)[:20]

        state["search_results"] = unique_results
        state["messages"].append(AIMessage(
            content=f"Tìm được {len(unique_results)} sản phẩm liên quan (bao gồm image URLs)"
        ))
        return state

    def _generate_search_queries(self, query: str, query_type: str) -> List[str]:
        """Generate multiple search queries based on query type"""
        base_queries = [query]

        if query_type == "benchmark":
            base_queries.extend([
                f"{query} popular",
                f"{query} trending",
                f"{query} best selling"
            ])
        elif query_type == "market_gap":
            base_queries.extend([
                f"{query} niche",
                f"{query} unique",
                f"{query} rare"
            ])
        elif query_type == "verify_idea":
            base_queries.extend([
                f"{query} similar",
                f"{query} concept",
                f"{query} design"
            ])
        elif query_type == "audience_volume":
            base_queries.extend([
                f"{query} popular",
                f"{query} engagement",
                f"{query} viral"
            ])

        return base_queries[:3]  # Limit to 3 queries

    def _get_filters_for_query_type(self, query_type: str) -> Dict[str, Any]:
        """Get appropriate filters for each query type"""
        filters = {}

        if query_type == "benchmark":
            # For benchmark, we might want high engagement products
            filters["min_engagement"] = 100

        elif query_type == "audience_volume":
            # For audience volume, we want diverse platforms
            filters["platform"] = ["facebook", "instagram", "tiktok", "youtube"]

        # Add more specific filters as needed
        return filters

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate products based on ID"""
        seen_ids = set()
        unique_results = []

        for result in results:
            product_id = result.get("id")
            if product_id and product_id not in seen_ids:
                seen_ids.add(product_id)
                unique_results.append(result)

        return unique_results

    # Alias for backward compatibility
    async def search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method"""
        return await self.process(state)