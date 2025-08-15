"""
Benchmark analysis agent for Enhanced RnD Assistant
"""
import json
from typing import Dict, Any, List

from agents.base_agent import BaseAgent


class BenchmarkAgent(BaseAgent):
    """Agent for competitor benchmark analysis"""

    def __init__(self):
        super().__init__(temperature=0.3)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark data"""
        products = state["search_results"]

        if not products:
            state["analysis_results"] = {"error": "No products found for benchmark analysis"}
            return state

        # Analyze engagement patterns
        engagement_analysis = self._analyze_engagement(products)

        # Classify winning vs losing products
        winning_products, losing_products = self._classify_products(products)

        # Extract insights from metadata
        metadata_insights = self._extract_metadata_insights(products)

        # Identify success factors
        success_factors = self._identify_success_factors(winning_products, metadata_insights)

        analysis = {
            "engagement_analysis": engagement_analysis,
            "winning_products": winning_products[:5],  # Top 5 winners
            "losing_products": losing_products[:5],  # Bottom 5
            "metadata_insights": metadata_insights,
            "key_success_factors": success_factors,
            "total_products_analyzed": len(products)
        }

        state["analysis_results"] = analysis
        return state

    def _analyze_engagement(self, products: List[Dict]) -> Dict[str, Any]:
        """Analyze engagement metrics across all products"""
        total_likes = 0
        total_comments = 0
        total_shares = 0

        for product in products:
            engagement = product.get("engagement", {})
            total_likes += self._safe_int_convert(engagement.get("like", 0))
            total_comments += self._safe_int_convert(engagement.get("comment", 0))
            total_shares += self._safe_int_convert(engagement.get("share", 0))

        avg_engagement = (total_likes + total_comments * 5 + total_shares * 10) / len(products) if products else 0

        return {
            "total_products": len(products),
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_shares": total_shares,
            "average_engagement": avg_engagement
        }

    def _classify_products(self, products: List[Dict]) -> tuple:
        """Classify products into winning and losing based on engagement"""
        if not products:
            return [], []

        # Calculate engagement scores
        product_scores = []
        for product in products:
            score = self._calculate_engagement_score(product)
            product_scores.append((product, score))

        # Sort by engagement score
        product_scores.sort(key=lambda x: x[1], reverse=True)

        # Split into top and bottom performers
        split_point = len(product_scores) // 2
        winning_products = [p[0] for p in product_scores[:split_point]]
        losing_products = [p[0] for p in product_scores[split_point:]]

        return winning_products, losing_products

    def _extract_metadata_insights(self, products: List[Dict]) -> Dict[str, Any]:
        """Extract insights from product metadata"""
        insights = {
            "themes": {},
            "target_audiences": {},
            "occasions": {},
            "platforms": {},
            "design_styles": {}
        }

        for product in products:
            try:
                metadata = product.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                # Count themes
                for theme in metadata.get("niche_theme", []):
                    insights["themes"][theme] = insights["themes"].get(theme, 0) + 1

                # Count target audiences
                for audience in metadata.get("target_audience", []):
                    insights["target_audiences"][audience] = insights["target_audiences"].get(audience, 0) + 1

                # Count occasions
                for occasion in metadata.get("occasion", []):
                    insights["occasions"][occasion] = insights["occasions"].get(occasion, 0) + 1

                # Count platforms
                platform = product.get("platform", "unknown")
                insights["platforms"][platform] = insights["platforms"].get(platform, 0) + 1

            except Exception as e:
                continue

        return insights

    def _identify_success_factors(self, winning_products: List[Dict], insights: Dict) -> List[str]:
        """Identify key success factors from winning products"""
        factors = []

        # Top themes in winning products
        if insights.get("themes"):
            top_theme = max(insights["themes"].items(), key=lambda x: x[1])
            factors.append(f"Theme hiệu quả: {top_theme[0]} ({top_theme[1]} products)")

        # Top target audiences
        if insights.get("target_audiences"):
            top_audience = max(insights["target_audiences"].items(), key=lambda x: x[1])
            factors.append(f"Target audience chính: {top_audience[0]} ({top_audience[1]} products)")

        # Engagement patterns
        if winning_products:
            avg_winning_engagement = sum(self._calculate_engagement_score(p) for p in winning_products) / len(
                winning_products)
            factors.append(f"Engagement trung bình của winners: {avg_winning_engagement:,.0f}")

        return factors

    # Alias for backward compatibility
    async def analyze_benchmark(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method"""
        return await self.process(state)