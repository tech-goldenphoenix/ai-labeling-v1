"""
Response generator agent for Enhanced RnD Assistant
"""
from typing import Dict, Any

from langchain_core.messages import AIMessage

from agents.base_agent import BaseAgent


class EnhancedResponseGeneratorAgent(BaseAgent):
    """Agent to generate final responses based on analysis results"""

    def __init__(self):
        super().__init__(temperature=0.3)

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response"""
        query_type = state["query_type"]

        if query_type == "smart_search":
            response = self._generate_smart_search_response(state)
        elif query_type == "benchmark":
            response = self._generate_benchmark_response(state)
        elif query_type == "market_gap":
            response = self._generate_market_gap_response(state)
        elif query_type == "verify_idea":
            response = self._generate_verify_idea_response(state)
        elif query_type == "audience_volume":
            response = self._generate_audience_volume_response(state)
        else:
            response = "❌ Không thể xác định loại câu hỏi. Vui lòng thử lại."

        state["final_answer"] = response
        state["messages"].append(AIMessage(content=response))
        return state

    def _generate_smart_search_response(self, state: Dict[str, Any]) -> str:
        """Generate response for smart search queries"""
        search_type = state.get("search_type", "text_to_text")
        results = state.get("search_results", [])
        query = state["query"]

        response = f"## 🔍 Smart Search Results: {query}\n\n"
        response += f"**Search Type:** {search_type.replace('_', ' → ').title()}\n\n"

        if not results or (len(results) == 1 and "error" in results[0]):
            response += "❌ Không tìm thấy kết quả phù hợp.\n"
            if results and "error" in results[0]:
                response += f"⚠️ Lỗi: {results[0]['error']}\n"
            return response

        response += f"### 📊 Tổng Quan\n"
        response += f"- **Số sản phẩm tìm thấy:** {len(results)}\n"
        response += f"- **Bao gồm image URLs:** ✅\n\n"

        response += f"### 🎯 Top Results\n"
        for i, product in enumerate(results[:12], 1):
            similarity = product.get("similarity_score", 0)

            response += f"**{i}. {product.get('store', 'Unknown Store')}**\n"
            response += f"   - 🔗 **Image URL:** {product.get('image_url', 'N/A')}\n"
            response += f"   - 📝 **Description:** {product.get('description', 'N/A')}...\n"
            response += f"   - 📊 **Engagement:** {product.get('engagement', {})}\n"
            response += f"   - 🎯 **Similarity:** {similarity:.2%}\n"
            response += f"   - 📱 **Platform:** {product.get('platform', 'N/A')}\n"
            response += f"   - 📅 **Date:** {product.get('date', 'N/A')}\n\n"

        # Add image-specific insights
        if search_type in ["text_to_image", "image_to_image"]:
            response += f"### 🖼️ Image Analysis Insights\n"
            response += "- Tất cả kết quả đều có image URLs để tham khảo visual\n"
            response += "- Có thể download images từ URLs để phân tích thiết kế\n"
            response += "- Sử dụng cho inspiration và competitive analysis\n\n"

        # Add description if available (for image_to_text)
        if state.get("search_description"):
            response += f"### 📝 Image Description\n"
            response += f"{state['search_description']}\n\n"

        response += f"### 💡 Next Steps\n"
        response += "- Click vào image URLs để xem chi tiết visual\n"
        response += "- Phân tích engagement patterns của top performers\n"
        response += "- Sử dụng insights cho creative development\n"

        return response

    def _generate_benchmark_response(self, state: Dict[str, Any]) -> str:
        """Generate benchmark analysis response"""
        analysis = state.get("analysis_results", {})
        query = state["query"]

        if "error" in analysis:
            return f"❌ {analysis['error']}"

        response = f"## 📊 Benchmark Analysis: {query}\n\n"

        # Overview
        engagement = analysis.get("engagement_analysis", {})
        response += f"### 📈 Performance Overview\n"
        response += f"- **Total Products Analyzed:** {engagement.get('total_products', 0)}\n"
        response += f"- **Total Engagement:** {engagement.get('total_likes', 0):,} likes, {engagement.get('total_comments', 0):,} comments, {engagement.get('total_shares', 0):,} shares\n"
        response += f"- **Average Engagement:** {engagement.get('average_engagement', 0):,.0f}\n\n"

        # Winning vs Losing Analysis
        winning_products = analysis.get("winning_products", [])
        losing_products = analysis.get("losing_products", [])

        response += f"### 🏆 Winners vs Losers Analysis\n"
        response += f"- **Top Performers:** {len(winning_products)} products\n"
        response += f"- **Underperformers:** {len(losing_products)} products\n\n"

        # Top Winners with Image URLs
        if winning_products:
            response += f"### ✅ Top Winning Products\n"
            for i, product in enumerate(winning_products[:3], 1):
                engagement_score = self._calculate_engagement_score(product)
                response += f"**{i}. {product.get('store', 'Unknown')} - {engagement_score:,} engagement**\n"
                response += f"   - 🔗 **Image:** {product.get('image_url', 'N/A')}\n"
                response += f"   - 📝 **Description:** {product.get('description', 'N/A')[:80]}...\n"
                response += f"   - 📱 **Platform:** {product.get('platform', 'N/A')}\n\n"

        # Success Factors
        success_factors = analysis.get("key_success_factors", [])
        if success_factors:
            response += f"### 🎯 Key Success Factors\n"
            for factor in success_factors:
                response += f"- {factor}\n"
            response += "\n"

        # Metadata Insights
        metadata_insights = analysis.get("metadata_insights", {})
        if metadata_insights:
            response += f"### 🔍 Market Insights\n"

            top_themes = sorted(metadata_insights.get("themes", {}).items(), key=lambda x: x[1], reverse=True)[:3]
            if top_themes:
                response += f"**Popular Themes:** {', '.join([f'{theme}({count})' for theme, count in top_themes])}\n"

            top_audiences = sorted(metadata_insights.get("target_audiences", {}).items(), key=lambda x: x[1],
                                   reverse=True)[:3]
            if top_audiences:
                response += f"**Target Audiences:** {', '.join([f'{aud}({count})' for aud, count in top_audiences])}\n"

            top_platforms = sorted(metadata_insights.get("platforms", {}).items(), key=lambda x: x[1], reverse=True)[:3]
            if top_platforms:
                response += f"**Platform Distribution:** {', '.join([f'{plat}({count})' for plat, count in top_platforms])}\n\n"

        # Actionable Recommendations
        response += f"### 💡 Actionable Recommendations\n"
        if len(winning_products) > len(losing_products):
            response += "- ✅ **Positive Market Signal** - Many successful implementations exist\n"
            response += "- 📚 **Learning Opportunity** - Study winning products' image designs and messaging\n"
            response += "- 🚀 **Strategy** - Adapt successful elements while adding unique differentiation\n"
        else:
            response += "- ⚠️ **Challenging Market** - High failure rate observed\n"
            response += "- 🔍 **Deep Analysis Needed** - Investigate failure reasons in losing products\n"
            response += "- 💡 **Innovation Required** - Consider new approaches or pivot strategy\n"

        response += "- 🖼️ **Visual Analysis** - All products include image URLs for design inspiration\n"

        return response

    def _generate_market_gap_response(self, state: Dict[str, Any]) -> str:
        """Generate market gap analysis response"""
        analysis = state.get("analysis_results", {})
        query = state["query"]

        if "error" in analysis:
            return f"❌ {analysis['error']}"

        response = f"## 🕳️ Market Gap Analysis: {query}\n\n"

        # Current Market Analysis
        current_market = analysis.get("current_market_analysis", {})
        response += f"### 📊 Current Market Landscape\n"

        popular_themes = current_market.get("popular_themes", {})
        if popular_themes:
            top_themes = sorted(popular_themes.items(), key=lambda x: x[1], reverse=True)[:5]
            response += f"**Dominant Themes:** {', '.join([f'{theme}({count})' for theme, count in top_themes])}\n"

        target_audiences = current_market.get("target_audiences", {})
        if target_audiences:
            top_audiences = sorted(target_audiences.items(), key=lambda x: x[1], reverse=True)[:5]
            response += f"**Main Audiences:** {', '.join([f'{aud}({count})' for aud, count in top_audiences])}\n"

        occasions = current_market.get("occasions", {})
        if occasions:
            top_occasions = sorted(occasions.items(), key=lambda x: x[1], reverse=True)[:5]
            response += f"**Popular Occasions:** {', '.join([f'{occ}({count})' for occ, count in top_occasions])}\n\n"

        # Identified Gaps
        gaps = analysis.get("identified_gaps", {})
        response += f"### 🎯 Identified Market Gaps\n"

        audience_gaps = gaps.get("audience_gaps", [])
        if audience_gaps:
            response += f"**Underserved Audiences:** {', '.join(audience_gaps[:5])}\n"

        occasion_gaps = gaps.get("occasion_gaps", [])
        if occasion_gaps:
            response += f"**Missed Occasions:** {', '.join(occasion_gaps[:5])}\n"

        theme_gaps = gaps.get("theme_gaps", [])
        if theme_gaps:
            response += f"**Underdeveloped Themes:** {', '.join(theme_gaps[:3])}\n\n"

        # Market Opportunities
        opportunities = analysis.get("market_opportunities", [])
        if opportunities:
            response += f"### 🚀 Market Opportunities\n"
            for i, opp in enumerate(opportunities, 1):
                response += f"{i}. {opp}\n"
            response += "\n"

        # Underserved Segments
        underserved = analysis.get("underserved_segments", [])
        if underserved:
            response += f"### 📈 Underserved Segments\n"
            for segment in underserved[:5]:
                response += f"- {segment}\n"
            response += "\n"

        # Competitor Weaknesses
        weaknesses = analysis.get("competitor_weaknesses", [])
        if weaknesses:
            response += f"### 🎯 Competitor Weaknesses to Exploit\n"
            for weakness in weaknesses:
                response += f"- {weakness}\n"
            response += "\n"

        # Action Plan
        response += f"### 💡 Recommended Action Plan\n"
        response += "1. 🎯 **Target Gap Segments** - Focus on identified underserved audiences/occasions\n"
        response += "2. 📊 **Validate Opportunities** - Research deeper into gap segments using image analysis\n"
        response += "3. 🚀 **Develop Unique Positioning** - Create products that fill identified gaps\n"
        response += "4. 📱 **Multi-Platform Strategy** - Leverage underutilized platforms\n"
        response += "5. 🖼️ **Visual Differentiation** - Use image URLs to analyze visual gaps in design styles\n"

        return response

    def _generate_verify_idea_response(self, state: Dict[str, Any]) -> str:
        """Generate idea verification response"""
        analysis = state.get("analysis_results", {})
        query = state["query"]

        if "error" in analysis:
            return f"❌ {analysis['error']}"

        response = f"## ✅ Idea Verification: {query}\n\n"

        # Verification Overview
        similar_count = analysis.get("similar_products_found", 0)
        response += f"### 🔍 Verification Results\n"
        response += f"- **Similar Products Found:** {similar_count}\n"

        # Market Test Results
        test_results = analysis.get("market_test_results", {})
        response += f"- **Market Validation:** {test_results.get('market_validation', 'unknown').title()}\n"
        response += f"- **Success Rate:** {test_results.get('success_rate', 0):.1%}\n"
        response += f"- **Average Engagement:** {test_results.get('average_engagement', 0):,.0f}\n\n"

        # Viability Assessment
        viability = analysis.get("viability_assessment", "unknown")
        response += f"### 🎯 Viability Assessment\n"

        viability_messages = {
            "high_viability": "🟢 **HIGH VIABILITY** - Strong market validation",
            "moderate_viability": "🟡 **MODERATE VIABILITY** - Good potential with optimization",
            "low_viability": "🟠 **LOW VIABILITY** - Requires significant improvements",
            "high_risk": "🔴 **HIGH RISK** - Market shows poor performance",
            "untested_concept": "🆕 **UNTESTED CONCEPT** - No similar products found"
        }

        response += f"**Assessment:** {viability_messages.get(viability, viability)}\n\n"

        # Performance Breakdown
        if test_results.get("total_tests", 0) > 0:
            response += f"### 📊 Performance Breakdown\n"
            response += f"- **High Performers:** {test_results.get('high_performing', 0)} products\n"
            response += f"- **Medium Performers:** {test_results.get('medium_performing', 0)} products\n"
            response += f"- **Low Performers:** {test_results.get('low_performing', 0)} products\n\n"

        # Similar Concepts with Images
        similar_concepts = analysis.get("similar_concepts", [])
        if similar_concepts:
            response += f"### 🎯 Top Similar Concepts\n"
            for i, concept in enumerate(similar_concepts[:3], 1):
                engagement_score = self._calculate_engagement_score(concept)
                similarity = concept.get("similarity_score", 0)
                response += f"**{i}. {concept.get('store', 'Unknown')} - Similarity: {similarity:.1%}**\n"
                response += f"   - 🔗 **Image:** {concept.get('image_url', 'N/A')}\n"
                response += f"   - 📊 **Engagement:** {engagement_score:,}\n"
                response += f"   - 📝 **Description:** {concept.get('description', 'N/A')[:80]}...\n"
                response += f"   - 📱 **Platform:** {concept.get('platform', 'N/A')}\n\n"

        # Concept Patterns Analysis
        concept_analysis = analysis.get("concept_analysis", {})
        if concept_analysis:
            response += f"### 🔍 Pattern Analysis\n"

            platform_performance = concept_analysis.get("platform_performance", {})
            if platform_performance:
                best_platform = max(platform_performance.keys(),
                                    key=lambda x: platform_performance[x].get("avg_engagement", 0))
                best_engagement = platform_performance[best_platform].get("avg_engagement", 0)
                response += f"**Best Performing Platform:** {best_platform.title()} (avg: {best_engagement:,.0f})\n"

            common_themes = concept_analysis.get("common_themes", {})
            if common_themes:
                top_themes = sorted(common_themes.items(), key=lambda x: x[1], reverse=True)[:3]
                response += f"**Common Success Themes:** {', '.join([f'{theme}({count})' for theme, count in top_themes])}\n\n"

        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            response += f"### 💡 Strategic Recommendations\n"
            for rec in recommendations:
                response += f"- {rec}\n"
            response += "\n"

        # Next Steps
        response += f"### 🚀 Next Steps\n"
        response += "1. 🖼️ **Visual Analysis** - Study images of similar successful concepts\n"
        response += "2. 📊 **Engagement Deep Dive** - Analyze what drives high engagement\n"
        response += "3. 🎯 **Differentiation Strategy** - Find ways to improve upon existing concepts\n"
        response += "4. 🧪 **A/B Testing** - Test variations based on successful patterns\n"

        return response

    def _generate_audience_volume_response(self, state: Dict[str, Any]) -> str:
        """Generate audience volume estimation response"""
        analysis = state.get("analysis_results", {})
        query = state["query"]

        if "error" in analysis:
            return f"❌ {analysis['error']}"

        response = f"## 📊 Audience Volume Estimation: {query}\n\n"

        # Volume Estimation
        volume_est = analysis.get("estimated_audience_volume", {})
        if "error" not in volume_est:
            response += f"### 🎯 Estimated Audience Volume\n"
            response += f"- **Conservative Estimate:** {volume_est.get('conservative_estimate', 0):,}\n"
            response += f"- **Recommended Estimate:** {volume_est.get('recommended_estimate', 0):,}\n"
            response += f"- **Optimistic Estimate:** {volume_est.get('optimistic_estimate', 0):,}\n"
            response += f"- **Sample Size:** {volume_est.get('sample_size', 0)} products\n"
            response += f"- **Total Engagement Analyzed:** {volume_est.get('total_engagement_analyzed', 0):,}\n\n"
        else:
            response += f"- ⚠️ {volume_est['error']}\n\n"

        # Platform Breakdown
        platform_data = analysis.get("platform_breakdown", {})
        if platform_data:
            response += f"### 📱 Platform-wise Analysis\n"
            sorted_platforms = sorted(platform_data.items(),
                                      key=lambda x: x[1].get('avg_engagement', 0),
                                      reverse=True)

            for platform, data in sorted_platforms:
                response += f"**{platform.title()}:**\n"
                response += f"  - Products: {data['product_count']}\n"
                response += f"  - Avg Engagement: {data.get('avg_engagement', 0):,.0f}\n"
                response += f"  - Estimated Audience: {data.get('estimated_audience', 0):,}\n\n"

        # Trend Analysis
        trend_analysis = analysis.get("trend_analysis", {})
        trend_direction = trend_analysis.get("trend_direction", "unknown")

        response += f"### 📈 Trend Analysis\n"
        trend_messages = {
            "increasing": "📈 **GROWING** - Audience interest is increasing over time",
            "decreasing": "📉 **DECLINING** - Audience interest is decreasing",
            "stable": "➡️ **STABLE** - Consistent audience interest",
            "insufficient_data": "❓ **INSUFFICIENT DATA** - Need more temporal data for trend analysis"
        }
        response += f"**Trend Direction:** {trend_messages.get(trend_direction, trend_direction)}\n"

        data_span = trend_analysis.get("data_span_months", 0)
        if data_span > 0:
            response += f"**Data Span:** {data_span} months of data analyzed\n\n"
        else:
            response += "\n"

        # Confidence Level
        confidence = analysis.get("confidence_level", "unknown")
        response += f"### 🎯 Confidence Level\n"
        confidence_messages = {
            "very_high": "🟢 **VERY HIGH** - Highly reliable estimation",
            "high": "🟢 **HIGH** - Reliable estimation",
            "medium": "🟡 **MEDIUM** - Moderately reliable estimation",
            "low": "🟠 **LOW** - Limited reliability, need more data",
            "very_low": "🔴 **VERY LOW** - Unreliable, need significantly more data"
        }
        response += f"**Confidence:** {confidence_messages.get(confidence, confidence)}\n\n"

        # Volume Insights
        insights = analysis.get("volume_insights", [])
        if insights:
            response += f"### 💡 Key Insights\n"
            for insight in insights:
                response += f"- {insight}\n"
            response += "\n"

        # Methodology
        methodology = analysis.get("methodology", {})
        if methodology:
            response += f"### 📋 Estimation Methodology\n"
            response += f"- **Calculation:** {methodology.get('calculation_method', 'N/A')}\n"
            response += f"- **Assumptions:** {methodology.get('engagement_rate_assumption', 'N/A')}\n"
            response += f"- **Adjustments:** {methodology.get('platform_adjustments', 'N/A')}\n"
            response += f"- **Limitations:** {methodology.get('limitations', 'N/A')}\n\n"

        # Actionable Recommendations
        recommended_estimate = volume_est.get("recommended_estimate", 0)
        response += f"### 🚀 Strategic Recommendations\n"

        if recommended_estimate > 500000:
            response += "- 📈 **Mass Market Strategy** - Large audience justifies broad marketing\n"
            response += "- 🏭 **Scale Production** - High volume potential\n"
            response += "- 💰 **Competitive Pricing** - Volume economics allow competitive pricing\n"
        elif recommended_estimate > 50000:
            response += "- 🎯 **Targeted Marketing** - Substantial audience for focused campaigns\n"
            response += "- 📊 **Market Segmentation** - Consider sub-segments within audience\n"
            response += "- 💎 **Premium Positioning** - Medium audience allows premium approach\n"
        else:
            response += "- 🔍 **Niche Strategy** - Small audience requires highly targeted approach\n"
            response += "- 💎 **Premium/Luxury** - Small audience justifies higher pricing\n"
            response += "- 🎨 **Personalization** - Focus on customization and personal connection\n"

        response += "- 🖼️ **Visual Content Strategy** - Use image analysis to optimize creative content\n"
        response += "- 📊 **Continuous Monitoring** - Track engagement trends to refine estimates\n"

        return response

    # Alias for backward compatibility
    async def generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method"""
        return await self.process(state)