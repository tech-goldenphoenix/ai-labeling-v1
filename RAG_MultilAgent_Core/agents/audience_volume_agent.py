"""
Audience volume estimation agent for Enhanced RnD Assistant
"""
import numpy as np
from typing import Dict, Any, List

from agents.base_agent import BaseAgent
from config.settings import Config


class AudienceVolumeAgent(BaseAgent):
    """Agent for audience volume estimation"""

    def __init__(self):
        super().__init__(temperature=0.1)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate audience volume"""
        products = state["search_results"]

        if not products:
            state["analysis_results"] = {"error": "No products found for audience volume estimation"}
            return state

        # Analyze engagement data
        engagement_analysis = self._analyze_engagement_data(products)

        # Estimate audience volume
        volume_estimation = self._estimate_volume(products, engagement_analysis)

        # Analyze by platform
        platform_analysis = self._analyze_by_platform(products)

        # Trend analysis
        trend_analysis = self._analyze_trends(products)

        # Calculate confidence level
        confidence_level = self._calculate_confidence(products)

        analysis = {
            "estimated_audience_volume": volume_estimation,
            "platform_breakdown": platform_analysis,
            "trend_analysis": trend_analysis,
            "confidence_level": confidence_level,
            "volume_insights": self._generate_volume_insights(volume_estimation, platform_analysis),
            "methodology": self._explain_methodology()
        }

        state["analysis_results"] = analysis
        return state

    def _analyze_engagement_data(self, products: List[Dict]) -> Dict[str, Any]:
        """Comprehensive engagement analysis"""
        total_likes = 0
        total_comments = 0
        total_shares = 0
        engagement_scores = []

        for product in products:
            engagement = product.get("engagement", {})
            likes = self._safe_int_convert(engagement.get("like", 0))
            comments = self._safe_int_convert(engagement.get("comment", 0))
            shares = self._safe_int_convert(engagement.get("share", 0))

            total_likes += likes
            total_comments += comments
            total_shares += shares

            score = likes + comments * 5 + shares * 10
            engagement_scores.append(score)

        return {
            "total_likes": total_likes,
            "total_comments": total_comments,
            "total_shares": total_shares,
            "total_engagement": sum(engagement_scores),
            "average_engagement": sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0,
            "median_engagement": np.median(engagement_scores) if engagement_scores else 0,
            "max_engagement": max(engagement_scores) if engagement_scores else 0,
            "min_engagement": min(engagement_scores) if engagement_scores else 0
        }

    def _estimate_volume(self, products: List[Dict], engagement_data: Dict) -> Dict[str, Any]:
        """Estimate audience volume based on engagement metrics"""
        if not products:
            return {"error": "Insufficient data for estimation"}

        total_engagement = engagement_data.get("total_engagement", 0)
        total_likes = engagement_data.get("total_likes", 0)

        # Industry benchmarks for engagement rates
        conservative_rate = Config.ENGAGEMENT_RATES["conservative"]
        moderate_rate = Config.ENGAGEMENT_RATES["moderate"]
        optimistic_rate = Config.ENGAGEMENT_RATES["optimistic"]

        # Base estimation on likes (most common engagement)
        conservative_estimate = int(total_likes / conservative_rate)
        moderate_estimate = int(total_likes / moderate_rate)
        optimistic_estimate = int(total_likes / optimistic_rate)

        # Adjust based on platform mix and content type
        platform_multiplier = self._get_platform_multiplier(products)

        return {
            "conservative_estimate": int(conservative_estimate * platform_multiplier),
            "moderate_estimate": int(moderate_estimate * platform_multiplier),
            "optimistic_estimate": int(optimistic_estimate * platform_multiplier),
            "recommended_estimate": int(moderate_estimate * platform_multiplier),
            "total_engagement_analyzed": total_engagement,
            "sample_size": len(products),
            "methodology_notes": f"Based on {len(products)} products with {total_engagement:,} total engagement"
        }

    def _get_platform_multiplier(self, products: List[Dict]) -> float:
        """Get platform-specific multiplier for audience estimation"""
        platform_weights = Config.PLATFORM_WEIGHTS

        platform_counts = {}
        for product in products:
            platform = product.get("platform", "unknown").lower()
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        if not platform_counts:
            return 1.0

        # Calculate weighted average
        total_products = len(products)
        weighted_sum = 0

        for platform, count in platform_counts.items():
            weight = platform_weights.get(platform, 1.0)
            proportion = count / total_products
            weighted_sum += weight * proportion

        return weighted_sum

    def _analyze_by_platform(self, products: List[Dict]) -> Dict[str, Any]:
        """Detailed platform-wise analysis"""
        platform_data = {}

        for product in products:
            platform = product.get("platform", "unknown")
            if platform not in platform_data:
                platform_data[platform] = {
                    "product_count": 0,
                    "total_engagement": 0,
                    "total_likes": 0,
                    "total_comments": 0,
                    "total_shares": 0,
                    "products": []
                }

            engagement = product.get("engagement", {})
            likes = self._safe_int_convert(engagement.get("like", 0))
            comments = self._safe_int_convert(engagement.get("comment", 0))
            shares = self._safe_int_convert(engagement.get("share", 0))
            engagement_score = likes + comments * 5 + shares * 10

            platform_data[platform]["product_count"] += 1
            platform_data[platform]["total_engagement"] += engagement_score
            platform_data[platform]["total_likes"] += likes
            platform_data[platform]["total_comments"] += comments
            platform_data[platform]["total_shares"] += shares
            platform_data[platform]["products"].append(product)

        # Calculate averages and estimates per platform
        for platform in platform_data:
            data = platform_data[platform]
            count = data["product_count"]

            if count > 0:
                data["avg_engagement"] = data["total_engagement"] / count
                data["avg_likes"] = data["total_likes"] / count
                data["estimated_audience"] = int(data["total_likes"] / 0.04)  # 4% engagement rate

        return platform_data

    def _analyze_trends(self, products: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal trends in the data"""
        # Group products by time period
        monthly_data = {}

        for product in products:
            date_str = product.get("date", "")
            # Extract month-year (simplified)
            if len(date_str) >= 7:
                month_key = date_str[:7]  # YYYY-MM format
            else:
                month_key = "unknown"

            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    "count": 0,
                    "total_engagement": 0,
                    "products": []
                }

            engagement_score = self._calculate_engagement_score(product)
            monthly_data[month_key]["count"] += 1
            monthly_data[month_key]["total_engagement"] += engagement_score
            monthly_data[month_key]["products"].append(product)

        # Calculate trend direction
        trend_direction = self._calculate_trend_direction(monthly_data)

        # Calculate monthly averages
        for month in monthly_data:
            data = monthly_data[month]
            if data["count"] > 0:
                data["avg_engagement"] = data["total_engagement"] / data["count"]

        return {
            "monthly_breakdown": monthly_data,
            "trend_direction": trend_direction,
            "data_span_months": len([k for k in monthly_data.keys() if k != "unknown"])
        }

    def _calculate_trend_direction(self, monthly_data: Dict) -> str:
        """Calculate overall trend direction"""
        valid_months = [(k, v) for k, v in monthly_data.items() if k != "unknown"]

        if len(valid_months) < 2:
            return "insufficient_data"

        # Sort by month
        sorted_months = sorted(valid_months, key=lambda x: x[0])

        if len(sorted_months) < 3:
            # Simple comparison between first and last
            first_engagement = sorted_months[0][1]["total_engagement"]
            last_engagement = sorted_months[-1][1]["total_engagement"]

            if last_engagement > first_engagement * 1.2:
                return "increasing"
            elif last_engagement < first_engagement * 0.8:
                return "decreasing"
            else:
                return "stable"

        # Linear trend for more data points
        engagements = [month_data["total_engagement"] for _, month_data in sorted_months]

        # Simple linear regression slope
        n = len(engagements)
        x_mean = (n - 1) / 2
        y_mean = sum(engagements) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(engagements))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope > y_mean * 0.1:  # 10% positive slope
            return "increasing"
        elif slope < -y_mean * 0.1:  # 10% negative slope
            return "decreasing"
        else:
            return "stable"

    def _calculate_confidence(self, products: List[Dict]) -> str:
        """Calculate confidence level of the estimation"""
        sample_size = len(products)

        # Check engagement variety
        engagement_scores = [self._calculate_engagement_score(p) for p in products]
        engagement_variance = np.var(engagement_scores) if engagement_scores else 0

        # Check platform diversity
        platforms = set(p.get("platform", "unknown") for p in products)
        platform_diversity = len(platforms)

        # Check time span
        dates = [p.get("date", "") for p in products if p.get("date")]
        time_span = len(set(d[:7] for d in dates if len(d) >= 7))  # Unique months

        # Scoring system
        score = 0

        # Sample size score (0-40 points)
        if sample_size >= 50:
            score += 40
        elif sample_size >= 20:
            score += 30
        elif sample_size >= 10:
            score += 20
        elif sample_size >= 5:
            score += 10

        # Platform diversity score (0-30 points)
        if platform_diversity >= 4:
            score += 30
        elif platform_diversity >= 3:
            score += 20
        elif platform_diversity >= 2:
            score += 15
        elif platform_diversity >= 1:
            score += 10

        # Time span score (0-30 points)
        if time_span >= 6:
            score += 30
        elif time_span >= 3:
            score += 20
        elif time_span >= 2:
            score += 15
        elif time_span >= 1:
            score += 10

        # Convert to confidence level
        if score >= 80:
            return "very_high"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        elif score >= 20:
            return "low"
        else:
            return "very_low"

    def _generate_volume_insights(self, volume_data: Dict, platform_data: Dict) -> List[str]:
        """Generate actionable insights about audience volume"""
        insights = []

        recommended_estimate = volume_data.get("recommended_estimate", 0)

        # Volume size insights
        if recommended_estimate > 1000000:
            insights.append("🎯 Large audience volume (1M+) - Mass market potential")
            insights.append("📈 Consider broad marketing strategies and mass production")
        elif recommended_estimate > 100000:
            insights.append("📊 Substantial audience volume (100K+) - Good market opportunity")
            insights.append("🎯 Focus on targeted marketing with potential for scale")
        elif recommended_estimate > 10000:
            insights.append("🔍 Niche audience volume (10K+) - Focused targeting needed")
            insights.append("💎 Premium positioning may be more effective")
        else:
            insights.append("⚠️ Small audience volume (<10K) - Very niche market")
            insights.append("🎨 Consider expanding concept or finding broader appeal")

        # Platform insights
        if platform_data:
            top_platform = max(platform_data.keys(),
                               key=lambda x: platform_data[x].get("avg_engagement", 0))
            top_engagement = platform_data[top_platform].get("avg_engagement", 0)

            insights.append(f"🏆 {top_platform.title()} shows highest engagement potential ({top_engagement:,.0f} avg)")

            # Multi-platform insights
            if len(platform_data) > 1:
                insights.append(f"🌐 Multi-platform presence across {len(platform_data)} platforms increases reach")
            else:
                insights.append("📱 Single platform focus - consider expanding to other channels")

        return insights

    def _explain_methodology(self) -> Dict[str, str]:
        """Explain the estimation methodology"""
        return {
            "engagement_rate_assumption": "Assumed 2-7% engagement rate based on industry benchmarks",
            "platform_adjustments": "Applied platform-specific multipliers based on typical audience behavior",
            "calculation_method": "Audience = Total Likes / Estimated Engagement Rate",
            "confidence_factors": "Sample size, platform diversity, time span, engagement variance",
            "limitations": "Estimates based on visible engagement, actual reach may vary significantly"
        }

    # Alias for backward compatibility
    async def estimate_audience_volume(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method"""
        return await self.process(state)