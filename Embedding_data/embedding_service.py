import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel, AutoProcessor
from sklearn.preprocessing import normalize
import warnings
from typing import List, Union, Optional
import base64

warnings.filterwarnings("ignore")


class JinaV4EmbeddingService:
    """
    Service sử dụng Jina v4 để tạo embedding cho text và image
    Hỗ trợ cả single và batch processing
    """

    def __init__(self, device=None,max_length=8192):
        """
        Khởi tạo Jina v4 embedding service

        Args:
            device: Device để chạy model ('cuda', 'cpu', hoặc None để auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Khởi tạo Jina v4 trên device: {self.device}")
        self.max_length = max_length
        # Load Jina v4 model và processor
        self.model_name = "jinaai/jina-clip-v2"

        # Xác định dtype phù hợp với device và hardware
        if self.device == 'cuda' and torch.cuda.is_available():
            # Kiểm tra khả năng hỗ trợ của GPU
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                model_dtype = torch.bfloat16
                print("🔧 Sử dụng bfloat16 cho GPU")
            else:
                model_dtype = torch.float16
                print("🔧 Sử dụng float16 cho GPU")
        else:
            model_dtype = torch.float32
            print("🔧 Sử dụng float32 cho CPU")

        try:
            # Load model với dtype phù hợp
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=model_dtype
            ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Đặt model ở chế độ eval
            self.model.eval()

            # Lấy embedding dimension
            self.embedding_dim = self._get_embedding_dimension()

            print(f"✅ Load model {self.model_name} thành công!")
            print(f"📊 Embedding dimension: {self.embedding_dim}")
            print(f"🔧 Model dtype: {model_dtype}")

        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            # Thử lại với float32 nếu có lỗi dtype
            print("🔄 Thử lại với float32...")
            try:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                ).to(self.device)

                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )

                self.model.eval()
                self.embedding_dim = self._get_embedding_dimension()
                print(f"✅ Load model thành công với float32!")
                print(f"📊 Embedding dimension: {self.embedding_dim}")
            except Exception as e2:
                raise Exception(f"Không thể load model: {e2}")

    def _get_embedding_dimension(self):
        """Lấy dimension của embedding vector với error handling tốt hơn"""
        try:
            # Test với một text ngắn để lấy dimension
            test_inputs = self.processor(text=["test"], return_tensors="pt", padding=True, truncation=True,max_length=self.max_length)

            # Chuyển inputs sang device
            test_inputs = {k: v.to(self.device) for k, v in test_inputs.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                test_output = self.model.get_text_features(**test_inputs)
                return test_output.shape[-1]
        except Exception as e:
            print(f"⚠️ Không thể tự động detect embedding dimension: {e}")
            return 1024  # Default dimension cho Jina CLIP v2

    def _load_image(self, image_url: str) -> Image.Image:
        """
        Load image từ URL hoặc đường dẫn local

        Args:
            image_url: URL hoặc đường dẫn đến image

        Returns:
            PIL Image object
        """
        try:
            if image_url.startswith(('http://', 'https://')):
                response = requests.get(image_url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_url)

            # Convert sang RGB nếu cần
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Resize image nếu quá lớn (tối ưu performance)
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            return image

        except Exception as e:
            raise ValueError(f"Không thể load image từ {image_url}: {e}")

    def _safe_model_inference(self, model_fn, **kwargs):
        """
        Safe wrapper cho model inference với dtype error handling

        Args:
            model_fn: Model function (get_text_features hoặc get_image_features)
            **kwargs: Arguments cho model function

        Returns:
            Model output tensor
        """
        try:
            return model_fn(**kwargs)
        except RuntimeError as e:
            if "BFloat16" in str(e) or "unsupported ScalarType" in str(e):
                print("⚠️ Dtype error detected, chuyển model sang float32...")
                # Chuyển model sang float32
                self.model = self.model.float()
                # Thử lại
                return model_fn(**kwargs)
            else:
                raise e

    def embed_text(self, text: str, normalize_output: bool = True) -> np.ndarray:
        """
        Tạo embedding cho text với error handling cải thiện

        Args:
            text: Text cần embedding
            normalize_output: Có normalize vector hay không

        Returns:
            numpy array chứa text embedding
        """
        if not text or not text.strip():
            # Trả về zero vector nếu text rỗng
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            # Preprocess input
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)

            # Chuyển inputs sang device
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                # Safe inference với dtype error handling
                outputs = self._safe_model_inference(self.model.get_text_features, **inputs)
                embedding = outputs.cpu().float().numpy()[0]  # Luôn chuyển về float32

            if normalize_output:
                embedding = normalize([embedding])[0]

            return embedding.astype(np.float32)

        except Exception as e:
            print(f"❌ Lỗi embedding text: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_image(self, image_url: str, normalize_output: bool = True) -> np.ndarray:
        """
        Tạo embedding cho image với error handling cải thiện

        Args:
            image_url: URL hoặc đường dẫn đến image
            normalize_output: Có normalize vector hay không

        Returns:
            numpy array chứa image embedding
        """
        if not image_url or not image_url.strip():
            # Trả về zero vector nếu image_url rỗng
            return np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            image = self._load_image(image_url)

            # Preprocess input
            inputs = self.processor(images=[image], return_tensors="pt")

            # Chuyển inputs sang device
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                # Safe inference với dtype error handling
                outputs = self._safe_model_inference(self.model.get_image_features, **inputs)
                embedding = outputs.cpu().float().numpy()[0]  # Luôn chuyển về float32

            if normalize_output:
                embedding = normalize([embedding])[0]

            return embedding.astype(np.float32)

        except Exception as e:
            print(f"❌ Lỗi embedding image {image_url}: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_multimodal(self, text: str, image_url: str = None, normalize: bool = True) -> tuple:
        """
        Tạo embedding cho cả text và image

        Args:
            text: Text cần embedding
            image_url: URL hoặc đường dẫn đến image (optional)
            normalize: Có normalize vectors hay không

        Returns:
            tuple: (image_vector, text_vector)
        """
        # Tạo text embedding
        text_vector = self.embed_text(text, normalize_output=normalize)

        # Tạo image embedding nếu có image_url
        if image_url and image_url.strip():
            image_vector = self.embed_image(image_url, normalize_output=normalize)
        else:
            # Trả về zero vector với cùng dimension nếu không có image
            image_vector = np.zeros(self.embedding_dim, dtype=np.float32)

        return image_vector, text_vector

    def embed_texts_batch(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> List[np.ndarray]:
        """
        Batch embedding cho nhiều text cùng lúc (hiệu quả hơn)

        Args:
            texts: List text cần embedding
            normalize: Có normalize vectors hay không
            batch_size: Kích thước batch

        Returns:
            List numpy arrays chứa text embeddings
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )

                # Chuyển inputs sang device
                inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    # Safe inference với dtype error handling
                    outputs = self._safe_model_inference(self.model.get_text_features, **inputs)
                    embeddings = outputs.cpu().float().numpy()  # Luôn chuyển về float32

                if normalize:
                    embeddings = normalize(embeddings)

                for emb in embeddings:
                    all_embeddings.append(emb.astype(np.float32))

            except Exception as e:
                print(f"❌ Lỗi batch embedding texts: {e}")
                # Thêm zero vectors cho batch bị lỗi
                for _ in batch_texts:
                    all_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        return all_embeddings

    def embed_images_batch(self, image_urls: List[str], normalize: bool = True, batch_size: int = 16) -> List[
        np.ndarray]:
        """
        Batch embedding cho nhiều image cùng lúc

        Args:
            image_urls: List URL images cần embedding
            normalize: Có normalize vectors hay không
            batch_size: Kích thước batch (nhỏ hơn text vì image tốn memory hơn)

        Returns:
            List numpy arrays chứa image embeddings
        """
        all_embeddings = []

        for i in range(0, len(image_urls), batch_size):
            batch_urls = image_urls[i:i + batch_size]
            batch_images = []

            # Load batch images
            for url in batch_urls:
                try:
                    if url and url.strip():
                        image = self._load_image(url)
                        batch_images.append(image)
                    else:
                        batch_images.append(None)
                except:
                    batch_images.append(None)

            # Process batch
            try:
                valid_images = [img for img in batch_images if img is not None]

                if not valid_images:
                    # Tất cả images trong batch đều invalid
                    for _ in batch_images:
                        all_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                    continue

                # Preprocess valid images
                inputs = self.processor(images=valid_images, return_tensors="pt")

                # Chuyển inputs sang device
                inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

                with torch.no_grad():
                    # Safe inference với dtype error handling
                    outputs = self._safe_model_inference(self.model.get_image_features, **inputs)
                    embeddings = outputs.cpu().float().numpy()  # Luôn chuyển về float32

                if normalize:
                    embeddings = normalize(embeddings)

                # Map embeddings back to original order
                valid_idx = 0
                for img in batch_images:
                    if img is not None:
                        all_embeddings.append(embeddings[valid_idx].astype(np.float32))
                        valid_idx += 1
                    else:
                        all_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

            except Exception as e:
                print(f"❌ Lỗi batch embedding images: {e}")
                # Thêm zero vectors cho batch bị lỗi
                for _ in batch_images:
                    all_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        return all_embeddings

    def get_model_info(self) -> dict:
        """
        Lấy thông tin về model

        Returns:
            Dictionary chứa thông tin model
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'torch_dtype': str(self.model.dtype) if hasattr(self, 'model') and hasattr(self.model,
                                                                                       'dtype') else 'unknown'
        }

    def similarity_search(self, query_vector: np.ndarray, candidate_vectors: List[np.ndarray], top_k: int = 5) -> List[
        tuple]:
        """
        Tìm kiếm similarity giữa query vector và danh sách candidate vectors

        Args:
            query_vector: Vector query
            candidate_vectors: List vectors để so sánh
            top_k: Số kết quả top trả về

        Returns:
            List tuple (index, similarity_score) được sắp xếp theo similarity giảm dần
        """
        similarities = []

        for i, candidate in enumerate(candidate_vectors):
            # Cosine similarity
            similarity = np.dot(query_vector, candidate) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(candidate)
            )
            similarities.append((i, float(similarity)))

        # Sắp xếp theo similarity giảm dần
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def __del__(self):
        """Cleanup khi object bị destroy"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass


# Tích hợp vào class chính
class EmbeddingService(JinaV4EmbeddingService):
    """
    Service wrapper để tương thích với code hiện tại
    Tích hợp với hàm _generate_vectors từ pipeline
    """

    def __init__(self, device=None):
        super().__init__(device)
        print(f"🤖 EmbeddingService khởi tạo với Jina v4")
        print(f"📊 Embedding dimensions: {self.embedding_dim}")

    def _generate_vectors(self, text: str, image_url: str = None) -> tuple:
        """
        Tạo embedding vectors cho text và image sử dụng Jina v4

        Hàm này tương thích với pipeline hiện tại

        Args:
            text: Text description để embedding
            image_url: URL của image để embedding

        Returns:
            tuple: (image_vector, text_vector)
        """
        image_vector, text_vector = self.embed_multimodal(
            text=text,
            image_url=image_url,
            normalize=True  # Normalize để tối ưu cho cosine similarity
        )

        print(f"✅ Tạo embedding thành công - Text: {len(text_vector)}D, Image: {len(image_vector)}D")
        return image_vector, text_vector

    def _generate_vectors_batch(self, descriptions: List[str], image_urls: List[str] = None) -> tuple:
        """
        Tạo embedding vectors cho nhiều text và image cùng lúc (hiệu quả hơn)

        Hàm này tương thích với pipeline hiện tại

        Args:
            descriptions: List text descriptions
            image_urls: List image URLs (optional)

        Returns:
            tuple: (image_vectors_list, text_vectors_list)
        """
        # Batch embedding cho text
        text_vectors = self.embed_texts_batch(
            descriptions,
            normalize=True,
            batch_size=32
        )

        # Batch embedding cho images (nếu có)
        if image_urls:
            image_vectors = self.embed_images_batch(
                image_urls,
                normalize=True,
                batch_size=16
            )
        else:
            image_vectors = [np.zeros(self.embedding_dim, dtype=np.float32) for _ in descriptions]

        print(f"✅ Tạo batch embedding thành công - {len(descriptions)} records")
        return image_vectors, text_vectors