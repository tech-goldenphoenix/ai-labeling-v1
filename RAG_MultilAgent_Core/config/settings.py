"""
Configuration settings for Enhanced RnD Assistant
Updated for Jina v4 integration
"""
import os
from typing import Dict, Any


class Config:
    """Main configuration class"""

    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "10.10.4.25")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "filtered_collection_from_chinh")

    # OpenAI Configuration (kept for compatibility with other parts of system)
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4-vision-preview")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")  # Legacy, not used with Jina

    # Jina v4 Configuration
    JINA_MODEL = os.getenv("JINA_MODEL", "jinaai/jina-clip-v2")
    JINA_DEVICE = os.getenv("JINA_DEVICE", None)  # None for auto-detect, "cuda" or "cpu"
    # Search Configuration - Updated for Jina v4
    TOP_K = int(os.getenv("TOP_K", "12"))

    # Jina CLIP v2 produces 1024-dimensional embeddings
    # Update these values to match your actual Jina model dimensions
    VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))  # Updated for Jina v4
    IMAGE_VECTOR_DIM = int(os.getenv("IMAGE_VECTOR_DIM", "1024"))  # Same as text for multimodal models

    # Batch processing configuration
    TEXT_BATCH_SIZE = int(os.getenv("TEXT_BATCH_SIZE", "32"))
    IMAGE_BATCH_SIZE = int(os.getenv("IMAGE_BATCH_SIZE", "16"))

    # Analysis Configuration
    HIGH_ENGAGEMENT_THRESHOLD = int(os.getenv("HIGH_ENGAGEMENT_THRESHOLD", "500"))
    LOW_ENGAGEMENT_THRESHOLD = int(os.getenv("LOW_ENGAGEMENT_THRESHOLD", "100"))

    # Platform Configuration
    PLATFORM_WEIGHTS = {
        "facebook": 1.2,
        "instagram": 1.0,
        "tiktok": 0.8,
        "youtube": 1.3,
        "twitter": 0.9,
    }

    # Engagement Rate Assumptions
    ENGAGEMENT_RATES = {
        "conservative": 0.02,
        "moderate": 0.04,
        "optimistic": 0.07
    }

    # Similarity thresholds for various use cases
    SIMILARITY_THRESHOLDS = {
        "high": 0.8,      # Very similar products
        "medium": 0.6,    # Moderately similar
        "low": 0.4,       # Loosely related
        "clustering": 0.7  # For trend clustering
    }

    # Multimodal search weights
    DEFAULT_TEXT_WEIGHT = float(os.getenv("DEFAULT_TEXT_WEIGHT", "0.6"))
    DEFAULT_IMAGE_WEIGHT = float(os.getenv("DEFAULT_IMAGE_WEIGHT", "0.4"))

    @classmethod
    def get_milvus_config(cls) -> Dict[str, str]:
        """Get Milvus connection configuration"""
        return {
            "host": cls.MILVUS_HOST,
            "port": cls.MILVUS_PORT,
            "collection_name": cls.COLLECTION_NAME
        }

    @classmethod
    def get_openai_config(cls) -> Dict[str, str]:
        """Get OpenAI configuration (for LLM, not embeddings)"""
        return {
            "model": cls.OPENAI_MODEL,
            "vision_model": cls.VISION_MODEL,
            "embedding_model": cls.EMBEDDING_MODEL  # Legacy
        }

    @classmethod
    def get_jina_config(cls) -> Dict[str, Any]:
        """Get Jina v4 configuration"""
        return {
            "model": cls.JINA_MODEL,
            "device": cls.JINA_DEVICE,
            "vector_dim": cls.VECTOR_DIM,
            "text_batch_size": cls.TEXT_BATCH_SIZE,
            "image_batch_size": cls.IMAGE_BATCH_SIZE
        }

    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "top_k": cls.TOP_K,
            "vector_dim": cls.VECTOR_DIM,
            "image_vector_dim": cls.IMAGE_VECTOR_DIM,
            "similarity_thresholds": cls.SIMILARITY_THRESHOLDS,
            "default_weights": {
                "text": cls.DEFAULT_TEXT_WEIGHT,
                "image": cls.DEFAULT_IMAGE_WEIGHT
            }
        }

    @classmethod
    def validate_dimensions(cls) -> bool:
        """
        Validate that vector dimensions are consistent
        This should be called after initializing the embedding service
        """
        return cls.VECTOR_DIM == cls.IMAGE_VECTOR_DIM  # For multimodal models, these should be equal

    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            "text_batch_size": cls.TEXT_BATCH_SIZE,
            "image_batch_size": cls.IMAGE_BATCH_SIZE,
            "device": cls.JINA_DEVICE,
            "enable_gpu": cls.JINA_DEVICE != "cpu"
        }

    @classmethod
    def update_vector_dimensions(cls, embedding_service):
        """
        Update vector dimensions based on actual embedding service
        Call this after initializing the embedding service to ensure consistency
        """
        if hasattr(embedding_service, 'embedding_dim'):
            actual_dim = embedding_service.embedding_dim
            if actual_dim != cls.VECTOR_DIM:
                print(f"⚠️ Updating vector dimensions from {cls.VECTOR_DIM} to {actual_dim}")
                cls.VECTOR_DIM = actual_dim
                cls.IMAGE_VECTOR_DIM = actual_dim
                return True
        return False