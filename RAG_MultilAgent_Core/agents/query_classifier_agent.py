"""
Query classifier agent for Enhanced RnD Assistant
"""
from typing import Dict, Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from agents.base_agent import BaseAgent


class EnhancedQueryClassifierAgent(BaseAgent):
    """Agent to classify user queries into different types"""

    def __init__(self):
        super().__init__(temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một classifier agent. Phân loại câu hỏi của user vào 1 trong 5 loại:

            1. "benchmark" - Câu hỏi về đối thủ cạnh tranh, phân tích winning/losing ideas
               Keywords: benchmark, đối thủ, cạnh tranh, winning, losing, so sánh

            2. "market_gap" - Câu hỏi về khoảng trống thị trường, cơ hội chưa được khai thác
               Keywords: gap, khoảng trống, cơ hội, chưa làm, thiếu, bỏ qua

            3. "verify_idea" - Câu hỏi xác minh ý tưởng, kiểm tra concept đã test chưa
               Keywords: verify, kiểm tra, xác minh, test, concept, ý tưởng

            4. "audience_volume" - Câu hỏi về ước tính audience volume của insight
               Keywords: audience, volume, ước tính, lượng người, khách hàng tiềm năng

            5. "smart_search" - Câu hỏi về tìm kiếm thông minh
               Keywords: tìm hình, show image, hình ảnh, tương tự, giống như, mô tả hình và các trường hợp khác còn lại ...

            Trả về chỉ 1 từ: benchmark, market_gap, verify_idea, audience_volume, hoặc smart_search"""),
            ("human", "{query}")
        ])

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the query type"""
        response = await self.llm.ainvoke(self.prompt.format_messages(query=state["query"]))
        query_type = response.content.strip().lower()

        state["query_type"] = query_type
        state["messages"].append(AIMessage(content=f"Đã phân loại câu hỏi: {query_type}"))
        return state

    # Alias for backward compatibility
    async def classify(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Alias for process method"""
        return await self.process(state)