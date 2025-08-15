"""
Milvus database manager for Enhanced RnD Assistant
Updated to use Jina v4 for consistent embeddings
"""
from typing import List, Dict, Any, Union
import json

from pymilvus import connections, Collection, utility
from PIL import Image
import numpy as np

from config.settings import Config
from database.embedding_service import EmbeddingService  # Import your Jina v4 service

class SingleCollectionMilvusManager:
    """Manages single collection Milvus operations with Jina v4 embeddings"""

    def __init__(self):
        self.collection = None
        # Replace OpenAI embeddings with Jina v4
        self.embedding_service = EmbeddingService()
        print(f"🔧 Initialized MilvusManager with Jina v4")
        print(f"📊 Embedding dimensions: {self.embedding_service.embedding_dim}")

    def connect(self):
        """Connect to Milvus and load the single collection"""
        connections.connect(
            alias="default",
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT
        )

        if utility.has_collection(Config.COLLECTION_NAME):
            self.collection = Collection(Config.COLLECTION_NAME)
            self.collection.load()
            print(f"✅ Collection {Config.COLLECTION_NAME} loaded successfully!")
        else:
            raise Exception(f"Collection {Config.COLLECTION_NAME} not found!")

    def search_products(self, query_vector: List[float], top_k: int = Config.TOP_K,
                        filter_expr: str = None) -> List[Dict]:
        """
        Universal search function for the single collection
        Returns products with image URLs included
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 12}
        }

        # Define output fields including image URL
        output_fields = [
            "id_sanpham", "description", "metadata", "date", "image",
            "like", "comment", "share", "platform", "name_store"
        ]

        results = self.collection.search(
            data=[query_vector],
            anns_field="description_vector",  # Main vector field for search
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
            expr=filter_expr  # Optional filtering
        )

        return self._format_search_results(results)

    def search_by_text_description(self, description: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """Search products by text description using Jina v4 - returns with image URLs"""
        query_vector = self.get_query_vector(description)
        return self.search_products(query_vector, top_k)

    def search_by_image_vector(self, image_vector: List[float], top_k: int = Config.TOP_K) -> List[Dict]:
        """
        Search by image vector - use image_vector field if available,
        otherwise use description_vector with image-based query
        """
        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 12}
            }

            output_fields = [
                "id_sanpham", "description", "metadata", "date", "image",
                "like", "comment", "share", "platform", "name_store"
            ]

            # Try image_vector field first
            results = self.collection.search(
                data=[image_vector],
                anns_field="image_vector",  # Use image vector field
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )

            return self._format_search_results(results)

        except Exception as e:
            print(f"Image vector search failed, falling back to description vector: {e}")
            # Fallback to description vector search
            return self.search_products(image_vector, top_k)

    def search_by_image_url(self, image_url: str, top_k: int = Config.TOP_K) -> List[Dict]:
        """
        Search by image URL using Jina v4 image embedding
        """
        try:
            # Generate image embedding using Jina v4
            image_vector = self.embedding_service.embed_image(image_url, normalize_output=True)
            return self.search_by_image_vector(image_vector.tolist(), top_k)
        except Exception as e:
            print(f"Error in image URL search: {e}")
            return []

    def search_multimodal(self, text: str = "", image_url: str = "",
                          top_k: int = Config.TOP_K) -> List[Dict]:
        """
        Sequential multimodal search: image first, then filter by text similarity

        Args:
            text: Text description for filtering
            image_url: Image URL for initial search
            top_k: Number of final results to return
        """
        try:
            # Step 1: Search by image (if provided)
            if image_url:
                # Get more candidates from image search (2x top_k)
                image_candidates = self.search_by_image_url(image_url, top_k * 2)

                # Step 2: Filter by text similarity (if provided)
                if text and image_candidates:
                    text_vector = self.embedding_service.embed_text(text, normalize_output=True)

                    # Calculate text similarity for each candidate
                    for candidate in image_candidates:
                        desc_vector = self.get_query_vector(candidate['description'])
                        # Cosine similarity
                        text_sim = np.dot(text_vector, desc_vector) / (
                                np.linalg.norm(text_vector) * np.linalg.norm(desc_vector)
                        )
                        candidate['text_similarity'] = text_sim

                    # Sort by text similarity and return top_k
                    image_candidates.sort(key=lambda x: x['text_similarity'], reverse=True)
                    return image_candidates[:top_k]

                return image_candidates[:top_k]

            # Step 3: Fallback to text-only search
            elif text:
                return self.search_by_text_description(text, top_k)

            return []

        except Exception as e:
            print(f"Error in multimodal search: {e}")
            return []

    def search_with_filters(self, query_vector: List[float], filters: Dict[str, Any],
                            top_k: int = Config.TOP_K) -> List[Dict]:
        """Search with additional filters for specific analysis needs"""

        # Build filter expression
        filter_conditions = []

        if "platform" in filters:
            platforms = filters["platform"] if isinstance(filters["platform"], list) else [filters["platform"]]
            platform_conditions = [f'platform == "{p}"' for p in platforms]
            if platform_conditions:
                filter_conditions.append(f"({' or '.join(platform_conditions)})")

        if "date_range" in filters:
            start_date, end_date = filters["date_range"]
            filter_conditions.append(f'date >= "{start_date}" and date <= "{end_date}"')

        if "min_engagement" in filters:
            # This would require a computed field or pre-processing
            pass  # Skip for now, can be post-processed

        filter_expr = " and ".join(filter_conditions) if filter_conditions else None

        return self.search_products(query_vector, top_k, filter_expr)

    def _format_search_results(self, results) -> List[Dict]:
        """Format search results to include image URLs and all necessary data"""
        products = []
        for hits in results:
            for hit in hits:
                product_data = {
                    "id": hit.entity.get("id_sanpham"),
                    "description": hit.entity.get("description"),
                    "image_url": hit.entity.get("image"),  # Image URL from the collection
                    "metadata": hit.entity.get("metadata"),
                    "engagement": {
                        "like": hit.entity.get("like"),
                        "comment": hit.entity.get("comment"),
                        "share": hit.entity.get("share")
                    },
                    "platform": hit.entity.get("platform"),
                    "store": hit.entity.get("name_store"),
                    "date": hit.entity.get("date"),
                    "similarity_score": hit.score
                }
                products.append(product_data)
        return products

    def get_query_vector(self, text: str) -> List[float]:
        """Convert text to vector embedding using Jina v4"""
        try:
            text_vector = self.embedding_service.embed_text(text, normalize_output=True)
            return text_vector.tolist()
        except Exception as e:
            print(f"Error generating text vector: {e}")
            return [0.0] * self.embedding_service.embedding_dim

    def get_image_vector(self, image_data: Union[str, bytes, Image.Image]) -> List[float]:
        """Convert image to vector embedding using Jina v4"""
        try:
            if isinstance(image_data, str):
                image_vector = self.embedding_service.embed_image(image_data, normalize_output=True)
            elif isinstance(image_data, bytes):
                from io import BytesIO
                pil_image = Image.open(BytesIO(image_data))
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file_path = tmp_file.name
                pil_image.save(tmp_file_path)
                image_vector = self.embedding_service.embed_image(tmp_file_path, normalize_output=True)
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass  # Ignore if can't delete
            elif isinstance(image_data, Image.Image):
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file_path = tmp_file.name
                image_data.save(tmp_file_path)
                image_vector = self.embedding_service.embed_image(tmp_file_path, normalize_output=True)
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass  # Ignore if can't delete
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")

            return image_vector.tolist()

        except Exception as e:
            print(f"Error generating image vector: {e}")
            return [0.0] * self.embedding_service.embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information"""
        return {
            "embedding_service": self.embedding_service.get_model_info(),
            "milvus_collection": Config.COLLECTION_NAME,
            "vector_dimensions": self.embedding_service.embedding_dim
        }

    def batch_search_texts(self, texts: List[str], top_k: int = Config.TOP_K) -> List[List[Dict]]:
        """
        Batch search for multiple text queries (more efficient)

        Returns:
            List of search results for each text query
        """
        try:
            # Generate batch embeddings
            text_vectors = self.embedding_service.embed_texts_batch(texts, normalize=True)

            results = []
            for vector in text_vectors:
                search_result = self.search_products(vector.tolist(), top_k)
                results.append(search_result)

            return results

        except Exception as e:
            print(f"Error in batch text search: {e}")
            return [[] for _ in texts]

    def batch_search_images(self, image_urls: List[str], top_k: int = Config.TOP_K) -> List[List[Dict]]:
        """
        Batch search for multiple image URLs (more efficient)

        Returns:
            List of search results for each image
        """
        try:
            # Generate batch embeddings
            image_vectors = self.embedding_service.embed_images_batch(image_urls, normalize=True)

            results = []
            for vector in image_vectors:
                search_result = self.search_by_image_vector(vector.tolist(), top_k)
                results.append(search_result)

            return results

        except Exception as e:
            print(f"Error in batch image search: {e}")
            return [[] for _ in image_urls]


# Global instance
milvus_manager = SingleCollectionMilvusManager()