"""
Verify idea agent for Enhanced RnD Assistant
"""
import json
from typing import Dict, Any, List

from agents.base_agent import BaseAgent


class VerifyIdeaAgent(BaseAgent):
    """Agent for idea verification analysis"""

    def __init__(self):
        super().__init__(temperature=0.2)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Verify idea based on market data"""
        products = state["search_results"]
        query = state["query"]

        if not products:
            state["analysis_results"] = {"error": "No similar products found for idea verification"}
            return state

        # Find most similar products
        similar_products = self._find_most_similar(products, query)

        # Analyze market test results
        test_results = self._analyze_market_tests(similar_products)

        # Assess idea viability
        viability_assessment = self._assess_viability(similar_products, test_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(viability_assessment, test_results)

        analysis = {
            "similar_products_found": len(similar_products),
            "market_test_results": test_results,
            "viability_assessment": viability_assessment,
            "similar_concepts": similar_products[:10],  # Top 5 most similar
            "recommendations": recommendations,
            "concept_analysis": self._analyze_concept_patterns(similar_products)
        }

        state["analysis_results"] = analysis
        return state

    def _find_most_similar(self, products: List[Dict], query: str) -> List[Dict]:
        """Find products most similar to the idea"""
        # Sort by similarity score (from vector search)
        return sorted(products, key=lambda x: x.get("similarity_score", 0), reverse=True)

    def _analyze_market_tests(self, products: List[Dict]) -> Dict[str, Any]:
        """Analyze how similar concepts performed in the market"""
        if not products:
            return {"status": "no_similar_products"}

        total_products = len(products)
        engagement_scores = [self._calculate_engagement_score(p) for p in products]

        # Define performance thresholds
        high_threshold = 500
        low_threshold = 100

        high_performing = len([score for score in engagement_scores if score > high_threshold])
        low_performing = len([score for score in engagement_scores if score < low_threshold])
        medium_performing = total_products - high_performing - low_performing

        success_rate = high_performing / total_products if total_products > 0 else 0
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0

        return {
            "total_tests": total_products,
            "high_performing": high_performing,
            "medium_performing": medium_performing,
            "low_performing": low_performing,
            "success_rate": success_rate,
            "average_engagement": avg_engagement,
            "market_validation": "validated" if success_rate > 0.3 else "needs_validation"
        }

    def _assess_viability(self, products: List[Dict], test_results: Dict) -> str:
        """Assess the viability of the idea based on market data"""
        if not products:
            return "untested_concept"

        success_rate = test_results.get("success_rate", 0)
        avg_engagement = test_results.get("average_engagement", 0)

        if success_rate > 0.5 and avg_engagement > 300:
            return "high_viability"
        elif success_rate > 0.3 and avg_engagement > 150:
            return "moderate_viability"
        elif success_rate > 0.1 or avg_engagement > 50:
            return "low_viability"
        else:
            return "high_risk"

    def _generate_recommendations(self, viability: str, test_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        viability_actions = {
            "high_viability": [
                "✅ Ý tưởng có tiềm năng cao, nên triển khai ngay",
                "📊 Tham khảo các successful implementations để optimize",
                "🚀 Scale nhanh để chiếm market share"
            ],
            "moderate_viability": [
                "⚠️ Ý tưởng có tiềm năng trung bình, cần optimize trước khi scale",
                "🔍 Nghiên cứu thêm yếu tố success từ high-performing products",
                "🧪 Test nhỏ trước, thu thập feedback để improve"
            ],
            "low_viability": [
                "🚨 Ý tưởng có rủi ro cao, cần pivot hoặc major improvements",
                "💡 Tìm cách differentiate mạnh để tránh competition",
                "📈 Focus vào unique value proposition"
            ],
            "high_risk": [
                "⛔ Ý tưởng có risk rất cao, nên reconsider",
                "🔄 Pivot sang direction khác hoặc target audience khác",
                "🧠 Brainstorm lại concept từ đầu"
            ],
            "untested_concept": [
                "🆕 Concept hoàn toàn mới, cơ hội first-mover advantage",
                "🧪 Cần validation kỹ lưỡng trước khi invest lớn",
                "📊 Collect data từ small test để đánh giá"
            ]
        }

        recommendations.extend(viability_actions.get(viability, []))

        # Add specific recommendations based on test results
        success_rate = test_results.get("success_rate", 0)
        if success_rate < 0.2:
            recommendations.append("📊 Low success rate - focus on understanding failure reasons")

        return recommendations

    def _analyze_concept_patterns(self, products: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in similar concepts"""
        patterns = {
            "successful_patterns": [],
            "failure_patterns": [],
            "common_themes": {},
            "platform_performance": {}
        }

        high_performers = [p for p in products if self._calculate_engagement_score(p) > 300]
        low_performers = [p for p in products if self._calculate_engagement_score(p) < 100]

        # Analyze successful patterns
        for product in high_performers:
            try:
                metadata = product.get("metadata", "{}")
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                for theme in metadata.get("niche_theme", []):
                    patterns["common_themes"][theme] = patterns["common_themes"].get(theme, 0) + 1

            except:
                continue

        # Platform performance
        for product in products:
            platform = product.get("platform", "unknown")
            engagement = self._calculate_engagement_score(product)

            if platform not in patterns["platform_performance"]:
                patterns["platform_performance"][platform] = {"total": 0, "engagement": 0, "count": 0}

            patterns["platform_performance"][platform]["engagement"] += engagement
            patterns["platform_performance"][platform]["count"] += 1

        # Calculate average engagement per platform
        for platform in patterns["platform_performance"]:
            data = patterns["platform_performance"][platform]
            data["avg_engagement"] = data["engagement"] / data["count"] if data["count"] > 0 else 0

        return patterns

    # Alias for backward compatibility
    async def verify_idea(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method"""
        return await self.process(state)