"""
Base agent class for Enhanced RnD Assistant
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_openai import ChatOpenAI

from config.settings import Config


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, temperature: float = 0.3):
        self.llm = ChatOpenAI(model=Config.OPENAI_MODEL, temperature=temperature, api_key="")

    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state and return updated state"""
        pass

    def _safe_int_convert(self, value) -> int:
        """Safely convert value to integer"""
        try:
            if isinstance(value, str):
                return int(value.replace(",", ""))
            return int(value) if value else 0
        except:
            return 0

    def _calculate_engagement_score(self, product: Dict) -> int:
        """Calculate engagement score for a product"""
        try:
            engagement = product.get("engagement", {})
            likes = self._safe_int_convert(engagement.get("like", 0))
            comments = self._safe_int_convert(engagement.get("comment", 0))
            shares = self._safe_int_convert(engagement.get("share", 0))
            return likes + comments * 5 + shares * 10
        except:
            return 0