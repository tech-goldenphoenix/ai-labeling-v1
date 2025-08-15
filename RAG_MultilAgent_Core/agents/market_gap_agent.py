"""
Market gap analysis agent for Enhanced RnD Assistant
"""
import json
from typing import Dict, Any, List

from agents.base_agent import BaseAgent


class MarketGapAgent(BaseAgent):
    """Agent for market gap analysis"""

    def __init__(self):
        super().__init__(temperature=0.3)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Find market gaps"""
        products = state["search_results"]

        if not products:
            state["analysis_results"] = {"error": "No products found for market gap analysis"}
            return state

        # Analyze what's currently being done
        current_market_analysis = self._analyze_current_market(products)

        # Identify potential gaps
        identified_gaps = self._identify_gaps(current_market_analysis)

        # Suggest opportunities
        opportunities = self._suggest_opportunities(identified_gaps, state["query"])

        analysis = {
            "current_market_analysis": current_market_analysis,
            "identified_gaps": identified_gaps,
            "market_opportunities": opportunities,
            "underserved_segments": self._find_underserved_segments(current_market_analysis),
            "competitor_weaknesses": self._analyze_competitor_weaknesses(products)
        }

        state["analysis_results"] = analysis
        return state

    def _analyze_current_market(self, products: List[Dict]) -> Dict[str, Any]:
        """Analyze what competitors are currently doing"""
        analysis = {
            "popular_themes": {},
            "target_audiences": {},
            "occasions": {},
            "design_styles": {},
            "platforms": {},
            "price_ranges": {},
            "engagement_levels": []
        }

        for product in products:
            try:
                # Extract metadata
                metadata = product.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                # Analyze themes
                for theme in metadata.get("niche_theme", []):
                    analysis["popular_themes"][theme] = analysis["popular_themes"].get(theme, 0) + 1

                # Analyze audiences
                for audience in metadata.get("target_audience", []):
                    analysis["target_audiences"][audience] = analysis["target_audiences"].get(audience, 0) + 1

                # Analyze occasions
                for occasion in metadata.get("occasion", []):
                    analysis["occasions"][occasion] = analysis["occasions"].get(occasion, 0) + 1

                # Analyze platforms
                platform = product.get("platform", "unknown")
                analysis["platforms"][platform] = analysis["platforms"].get(platform, 0) + 1

                # Track engagement levels
                engagement_score = self._calculate_engagement_score(product)
                analysis["engagement_levels"].append(engagement_score)

            except Exception as e:
                continue

        return analysis

    def _identify_gaps(self, market_analysis: Dict) -> Dict[str, List[str]]:
        """Identify potential market gaps"""
        gaps = {
            "audience_gaps": [],
            "occasion_gaps": [],
            "theme_gaps": [],
            "platform_gaps": []
        }

        # Common audiences that might be underserved
        all_possible_audiences = ["Dad", "Mom", "Kids", "Teens", "Grandparents", "Couples", "Singles", "Professionals"]
        covered_audiences = set(market_analysis["target_audiences"].keys())
        gaps["audience_gaps"] = list(set(all_possible_audiences) - covered_audiences)

        # Common occasions that might be missed
        all_possible_occasions = ["Birthday", "Christmas", "Valentine", "Anniversary", "Graduation", "Wedding",
                                  "New Year", "Easter"]
        covered_occasions = set(market_analysis["occasions"].keys())
        gaps["occasion_gaps"] = list(set(all_possible_occasions) - covered_occasions)

        # Underrepresented themes (less than 2 products)
        for theme, count in market_analysis["popular_themes"].items():
            if count < 2:
                gaps["theme_gaps"].append(f"{theme} (chỉ {count} products)")

        return gaps

    def _suggest_opportunities(self, gaps: Dict, original_query: str) -> List[str]:
        """Suggest market opportunities based on identified gaps"""
        opportunities = []

        if gaps["audience_gaps"]:
            top_gap_audiences = gaps["audience_gaps"][:3]
            opportunities.append(f"🎯 Cơ hội target {', '.join(top_gap_audiences)} cho '{original_query}'")

        if gaps["occasion_gaps"]:
            top_gap_occasions = gaps["occasion_gaps"][:3]
            opportunities.append(f"🎉 Cơ hội khai thác occasions: {', '.join(top_gap_occasions)}")

        if gaps["theme_gaps"]:
            opportunities.append(f"💡 Cơ hội phát triển themes ít cạnh tranh: {', '.join(gaps['theme_gaps'][:2])}")

        if not any([gaps["audience_gaps"], gaps["occasion_gaps"], gaps["theme_gaps"]]):
            opportunities.append("📊 Thị trường đã khá bão hòa, cần tìm cách differentiate bằng quality hoặc innovation")

        return opportunities

    def _find_underserved_segments(self, market_analysis: Dict) -> List[str]:
        """Find segments that are underserved"""
        underserved = []

        # Audiences with very low representation
        for audience, count in market_analysis["target_audiences"].items():
            if count < 3:
                underserved.append(f"Audience '{audience}' chỉ có {count} products")

        # Occasions with low representation
        for occasion, count in market_analysis["occasions"].items():
            if count < 2:
                underserved.append(f"Occasion '{occasion}' chỉ có {count} products")

        return underserved

    def _analyze_competitor_weaknesses(self, products: List[Dict]) -> List[str]:
        """Analyze competitor weaknesses"""
        weaknesses = []

        # Low engagement products indicate weak execution
        low_engagement_count = 0
        for product in products:
            if self._calculate_engagement_score(product) < 100:
                low_engagement_count += 1

        if low_engagement_count > len(products) * 0.3:
            weaknesses.append(f"⚠️ {low_engagement_count} products có engagement thấp - cơ hội cải tiến execution")

        # Platform concentration risk
        platform_counts = {}
        for product in products:
            platform = product.get("platform", "unknown")
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        total_products = len(products)
        for platform, count in platform_counts.items():
            if count > total_products * 0.7:
                weaknesses.append(
                    f"🎯 Quá tập trung vào {platform} ({count}/{total_products}) - cơ hội diversify platform")

        return weaknesses

    # Alias for backward compatibility
    async def find_market_gaps(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method"""
        return await self.process(state)