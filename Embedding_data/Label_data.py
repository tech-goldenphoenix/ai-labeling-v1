import psycopg2
import os
import json
import requests
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import base64
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from datetime import datetime, timedelta
import uuid
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import time
from embedding_service import EmbeddingService
import ollama


# Configuration
class ModelProvider(Enum):
    OLLAMA = "ollama"
    GOOGLE = "google"
    BOTH = "both"


@dataclass
class ProductLabel:
    """Cấu trúc label sản phẩm theo file PDF"""
    image_url: str
    image_recipient: List[str]
    target_audience: List[str]
    usage_purpose: List[str]
    occasion: List[str]
    niche_theme: List[str]
    sentiment_tone: List[str]
    message_type: List[str]
    personalization_type: List[str]
    product_type: List[str]
    placement_display_context: List[str]
    design_style: List[str]
    color_aesthetic: List[str]
    trademark_level: str
    main_subject: List[str]
    text: List[str]


@dataclass
class ProductRecord:
    """Cấu trúc record để insert vào Milvus"""
    id_sanpham: str
    image: str
    date: str
    like: str
    comment: str
    share: str
    link_redirect: str
    platform: str
    name_store: str
    description: str
    metadata: dict
    image_vector: List[float]
    description_vector: List[float]


class IntegratedProductPipeline:
    """Module tích hợp: Crawl Data → Label → Insert Milvus với GPT OSS 20B"""

    def __init__(self,
                 db_config: Dict[str, str],
                 google_api_key: str,
                 ollama_model: str = "gpt-oss:20b",
                 milvus_host: str = "10.10.4.25",
                 milvus_port: str = "19530",
                 use_gpu: bool = True):
        """
        Khởi tạo pipeline tích hợp với GPT OSS 20B và GPU support

        Args:
            db_config: Cấu hình database PostgreSQL
            google_api_key: API key cho Google Gemini
            ollama_model: Model Ollama để sử dụng (mặc định: gpt-oss:20b)
            milvus_host: Milvus host
            milvus_port: Milvus port
            use_gpu: Sử dụng GPU hay không
        """
        # Database config
        self.db_config = db_config
        self.db_connection = None
        
        # GPU config
        self.use_gpu = use_gpu
        self._setup_gpu_environment()

        # AI Labeler - Updated for GPT OSS 20B
        self.ollama_model = ollama_model
        self.google_client = None

        # Kiểm tra và verify model availability
        self._verify_ollama_model()

        # Khởi tạo EmbeddingService với GPU support
        print("🔧 Khởi tạo Embedding Service với GPU support...")
        self.embedding_service = EmbeddingService()
        self.embedding_dim = self.embedding_service.embedding_dim

        # Khởi tạo Google Gemini
        if google_api_key:
            genai.configure(api_key=google_api_key)
            self.google_client = genai.GenerativeModel('gemini-1.5-pro-latest')

        # Milvus config
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "product_collection"
        self.collection = None

        # Log model info
        model_info = self.embedding_service.get_model_info()
        print(f"🤖 Embedding Model: {model_info['model_name']}")
        print(f"📊 Embedding Dimensions: {model_info['embedding_dimension']}")
        print(f"🔧 Device: {model_info['device']}")
        print(f"🦾 Ollama Model: {self.ollama_model}")

        # Initialize connections
        self._connect_db()
        self._connect_milvus()
        self._setup_collection()

    def _setup_gpu_environment(self):
        """Cấu hình môi trường GPU"""
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                    print(f"🚀 GPU Support: Enabled")
                    print(f"   🔥 Device: {gpu_name}")
                    print(f"   📊 GPU Count: {gpu_count}")
                    print(f"   💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                    
                    # Set GPU environment variables cho Ollama
                    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                    os.environ['OLLAMA_GPU'] = '1'
                else:
                    print("⚠️  CUDA không khả dụng, sử dụng CPU")
                    self.use_gpu = False
            except ImportError:
                print("⚠️  PyTorch không được cài đặt, sử dụng CPU")
                self.use_gpu = False
        else:
            print("🖥️  GPU Support: Disabled (sử dụng CPU)")

    def _verify_ollama_model(self):
        """Kiểm tra và verify model Ollama có sẵn - Fixed version"""
        try:
            print(f"🔍 Kiểm tra model {self.ollama_model}...")
            
            # List available models với error handling tốt hơn
            try:
                available_models = ollama.list()
                print(f"📋 Raw response structure: {type(available_models)}")
                
                # Handle different response structures
                if isinstance(available_models, dict):
                    if 'models' in available_models:
                        model_list = available_models['models']
                    else:
                        model_list = available_models
                else:
                    model_list = available_models
                
                # Extract model names safely
                model_names = []
                if isinstance(model_list, list):
                    for model in model_list:
                        if isinstance(model, dict):
                            # Try different possible keys
                            name = model.get('name') or model.get('model') or model.get('id', 'unknown')
                            model_names.append(name)
                        else:
                            model_names.append(str(model))
                
                print(f"📋 Available models: {model_names}")
                
            except Exception as e:
                print(f"⚠️  Lỗi list models: {str(e)}")
                print(f"💡 Thử pull model trực tiếp...")
                model_names = []
            
            # Check if our model is available
            if self.ollama_model not in model_names:
                print(f"⚠️  Model {self.ollama_model} chưa có sẵn!")
                print(f"🔄 Đang tự động pull model {self.ollama_model}...")
                
                try:
                    # Pull model với progress tracking
                    print("⏳ Pulling model... (có thể mất vài phút)")
                    ollama.pull(self.ollama_model)
                    print(f"✅ Đã pull model {self.ollama_model} thành công!")
                    
                    # Verify model is working
                    test_response = ollama.generate(
                        model=self.ollama_model,
                        prompt="Test message",
                        options={'num_predict': 10}
                    )
                    print(f"✅ Model {self.ollama_model} đã sẵn sàng và hoạt động!")
                    
                except Exception as pull_error:
                    print(f"❌ Lỗi pull model: {str(pull_error)}")
                    print(f"💡 Vui lòng chạy thủ công: ollama pull {self.ollama_model}")
                    raise Exception(f"Model {self.ollama_model} không khả dụng")
            else:
                print(f"✅ Model {self.ollama_model} đã sẵn sàng!")
                
                # Quick test
                try:
                    test_response = ollama.generate(
                        model=self.ollama_model,
                        prompt="Hello",
                        options={'num_predict': 5}
                    )
                    print(f"✅ Model test successful!")
                except Exception as test_error:
                    print(f"⚠️  Model test warning: {str(test_error)}")
                
        except Exception as e:
            print(f"❌ Lỗi kiểm tra model: {str(e)}")
            print(f"💡 Vui lòng đảm bảo:")
            print(f"   1. Ollama đang chạy: ollama serve")
            print(f"   2. Model đã được pull: ollama pull {self.ollama_model}")
            print(f"   3. Có đủ VRAM cho model (khoảng 12-16GB)")

    def _connect_db(self) -> bool:
        """Kết nối đến PostgreSQL database"""
        try:
            self.db_connection = psycopg2.connect(**self.db_config)
            print("✅ Kết nối PostgreSQL thành công")
            return True
        except Exception as e:
            print(f"❌ Lỗi kết nối database: {e}")
            return False

    def _connect_milvus(self):
        """Kết nối tới Milvus"""
        try:
            connections.connect(
                alias="default",
                host=self.milvus_host,
                port=self.milvus_port
            )
            print("✅ Kết nối Milvus thành công")
        except Exception as e:
            raise Exception(f"❌ Lỗi kết nối Milvus: {str(e)}")

    def _create_collection_schema(self):
        """Tạo schema cho collection với embedding dimensions động"""
        fields = [
            FieldSchema(name="id_sanpham", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="image", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="like", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="comment", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="share", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="link_redirect", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="platform", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="name_store", dtype=DataType.VARCHAR, max_length=500)
        ]

        schema = CollectionSchema(
            fields=fields,
            description=f"Collection chứa thông tin sản phẩm với embedding {self.embedding_dim}D"
        )
        return schema

    def _setup_collection(self):
        """Tạo hoặc load collection"""
        try:
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                print(f"✅ Load collection '{self.collection_name}' thành công")
            else:
                schema = self._create_collection_schema()
                self.collection = Collection(self.collection_name, schema)
                self._create_indexes()
                print(f"✅ Tạo collection '{self.collection_name}' thành công với {self.embedding_dim}D vectors")

            self.collection.load()

        except Exception as e:
            raise Exception(f"❌ Lỗi setup collection: {str(e)}")

    def _create_indexes(self):
        """Tạo index cho vector fields"""
        # Chọn nlist phù hợp với embedding dimension
        nlist = min(self.embedding_dim, 1024)

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": nlist}
        }

        self.collection.create_index(
            field_name="image_vector",
            index_params=index_params,
            index_name="image_vector_index"
        )

        self.collection.create_index(
            field_name="description_vector",
            index_params=index_params,
            index_name="description_vector_index"
        )
        print(f"✅ Tạo indexes thành công với nlist={nlist}")

    # === DUPLICATE CHECK METHODS ===
    def check_id_exists(self, id_sanpham: str) -> bool:
        """
        Kiểm tra xem ID sản phẩm đã tồn tại trong Milvus chưa
        """
        try:
            expr = f'id_sanpham == "{id_sanpham}"'
            results = self.collection.query(
                expr=expr,
                output_fields=["id_sanpham"],
                limit=1
            )
            return len(results) > 0
        except Exception as e:
            print(f"⚠️  Lỗi kiểm tra ID tồn tại {id_sanpham}: {str(e)}")
            return False

    def check_ids_exist_batch(self, id_list: List[str]) -> Dict[str, bool]:
        """Kiểm tra nhiều ID cùng lúc để tối ưu performance"""
        try:
            if not id_list:
                return {}

            id_conditions = [f'id_sanpham == "{id_val}"' for id_val in id_list]
            expr = " or ".join(id_conditions)

            results = self.collection.query(
                expr=expr,
                output_fields=["id_sanpham"],
                limit=len(id_list)
            )

            existing_ids = {result["id_sanpham"] for result in results}
            return {id_val: id_val in existing_ids for id_val in id_list}

        except Exception as e:
            print(f"⚠️  Lỗi kiểm tra batch IDs: {str(e)}")
            return {id_val: False for id_val in id_list}

    def filter_existing_records(self, raw_data_list: List[Dict[str, Any]]) -> tuple:
        """Lọc bỏ các record đã tồn tại trong Milvus"""
        try:
            if not raw_data_list:
                return [], [], 0

            print(f"🔍 Kiểm tra trùng lặp cho {len(raw_data_list)} records...")

            id_list = [record.get('id_sanpham', '') for record in raw_data_list if record.get('id_sanpham')]

            if not id_list:
                return raw_data_list, [], 0

            existence_map = self.check_ids_exist_batch(id_list)

            new_records = []
            existing_records = []

            for record in raw_data_list:
                id_sanpham = record.get('id_sanpham', '')
                if id_sanpham and existence_map.get(id_sanpham, False):
                    existing_records.append(record)
                else:
                    new_records.append(record)

            duplicate_count = len(existing_records)

            print(f"✅ Kết quả kiểm tra trùng lặp:")
            print(f"   📦 Records mới: {len(new_records)}")
            print(f"   🔄 Records trùng lặp: {duplicate_count}")

            if duplicate_count > 0:
                print(f"   📋 Một số ID trùng lặp:")
                for i, record in enumerate(existing_records[:5]):
                    print(f"      {i + 1}. {record.get('id_sanpham', 'unknown')}")
                if duplicate_count > 5:
                    print(f"      ... và {duplicate_count - 5} ID khác")

            return new_records, existing_records, duplicate_count

        except Exception as e:
            print(f"❌ Lỗi khi lọc records trùng lặp: {str(e)}")
            return raw_data_list, [], 0

    # === CRAWL DATA METHODS ===
    def crawl_data_by_date_range(self, start_date: str, end_date: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Crawl data từ database theo khoảng thời gian"""
        if not self.db_connection:
            if not self._connect_db():
                return []

        try:
            cursor = self.db_connection.cursor()

            query = """
            SELECT 
                COALESCE(_id, CONCAT('SP_', SUBSTRING(MD5(_id::text), 1, 8))) as id_sanpham,
                COALESCE(original_url, thumb_url, '') as image,
                COALESCE(CAST(created_at_std AS text), CAST(NOW() AS text)) as date,
                COALESCE(CAST("like" AS text), '0') as like,
                COALESCE(CAST(comment AS text), '0') as comment,
                COALESCE(CAST(share AS text), '0') as share,
                COALESCE(final_url, link, '') as link_redirect,
                COALESCE(platform, 'Website') as platform,
                COALESCE(domain, 'unknown') as name_store
            FROM ai_craw.toidispy_full
            WHERE created_at_std BETWEEN %s AND %s
            ORDER BY created_at_std DESC
            LIMIT %s
            """

            cursor.execute(query, (start_date, end_date, limit))
            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                results.append(row_dict)

            cursor.close()
            print(f"✅ Crawl được {len(results)} records từ {start_date} đến {end_date}")
            return results

        except Exception as e:
            print(f"❌ Lỗi khi crawl data: {e}")
            return []
        

    # === LABELING METHODS - UPDATED FOR GPT OSS 20B ===
    def _create_labeling_prompt(self) -> str:
        """Tạo prompt chi tiết được tối ưu cho GPT OSS 20B"""
        return """
Bạn là một chuyên gia phân tích sản phẩm với khả năng hiểu sâu về thị trường và xu hướng người tiêu dùng. Hãy phân tích hình ảnh sản phẩm này và đánh label theo các tiêu chí sau:

**QUAN TRỌNG: CHỈ CHỌN 1-3 LABELS CHÍNH VÀ PHỦ HỢP NHẤT CHO MỖI TIÊU CHÍ**

**HƯỚNG DẪN PHÂN TÍCH:**
- Quan sát kỹ thiết kế, màu sắc, văn bản, và bối cảnh sử dụng
- Xác định đối tượng người dùng chính từ visual cues
- Phân tích cảm xúc và thông điệp mà sản phẩm truyền tải
- Chú ý đến chất lượng và phong cách thiết kế

**LƯU Ý QUAN TRỌNG:**
- Khi thấy "Children", hãy chi tiết hóa thành "Son", "Daughter", "Kids" thay vì dùng "Children" chung chung
- "3D Rendered" chỉ áp dụng khi THIẾT KẾ BÊN TRONG sản phẩm có vẻ được tạo bằng 3D rendering, KHÔNG PHẢI vì ảnh mockup trông 3D
- Phân biệt rõ ràng giữa mockup presentation và design style thực tế của sản phẩm
- Ưu tiên độ chính xác và cụ thể trong từng label

**TIÊU CHÍ ĐÁNH LABEL:**

1. **Image Recipient** (Người nhận - MAX 4 labels chính):
   - Thay vì "Children" → sử dụng "Son", "Daughter", "Kids" cụ thể
   - Chọn đối tượng chính và rõ ràng nhất (ví dụ: Mom, Dad, Son, Daughter, Wife, Husband)

2. **Target Audience** (Người mua - MAX 3 labels):
   - Chọn nhóm mua hàng chính 
   - Phải CỤ THỂ, không được chung chung như "Family Members" hay "Friends"
   - Ví dụ cụ thể: "From Daughter", "From Son", "From Husband", "From Wife", "From Mother", "From Father", "From Spouse", "From dog owners", "From beer enthusiasts", "From police officers", "From colleagues", etc.

3. **Usage Purpose** (Mục đích - MAX 3 labels):
   - Mục đích sử dụng chính (Gift, Home Decor, Personal Use, Keepsake, Functional Use)

4. **Occasion** (Dịp - MAX 3 labels):
   - Chỉ những dịp chính và rõ ràng nhất
   - Phải CỤ THỂ: "Mother's Birthday", "Father's Birthday", "Dad's Birthday", "Son's Birthday", "Daughter's Birthday", "Christmas Gift", "Mother's Day", "Father's Day", "Valentine's Day", "Anniversaries", "Pet birthdays", etc.

5. **Niche/Theme** (Chủ đề - MAX 3 labels):
   - Chủ đề chính của sản phẩm (Mother, Father, Police, Beer, Cowgirl, Witch, Pet, Sports, etc.)

6. **Sentiment/Tone** (Cảm xúc - MAX 3 labels):
   - Cảm xúc chính (Sentimental, Humorous, Elegant, Sophisticated, Playful, Inspirational, etc.)

7. **Message Type** (Loại thông điệp - MAX 1 label):
   - Chọn 1 loại phù hợp nhất (No quote, Symbolic Message, From-to Signature, Personal Identity)

8. **Personalization Type** (Cá nhân hóa - MAX 1 label):
   - Chọn loại cá nhân hóa chính (Personalized Name, Non-personalized, Custom Text, etc.)

9. **Product Type** (Loại sản phẩm - MAX 2 labels):
   - Loại sản phẩm cụ thể (Desk Plaque, Mug, Hoodie, Earrings, Watch, Keychain, Hanging Suncatcher, T-Shirt, etc.)

10. **Placement/Display Context** (Bối cảnh - MAX 2 labels):
    - Nơi trưng bày chính (Shelf decor, Desk decor, Bedroom display, Window decor, Wearable, etc.)

11. **Design Style** (Phong cách - MAX 4 labels):
    - CHÚ Ý: "3D Rendered" chỉ khi THIẾT KẾ in lên sản phẩm thực sự là 3D rendered
    - Các phong cách khác: Elegant, Vintage, Stained Glass, Floral Motif, Gothic, Minimalist, Abstract, etc.

12. **Color Aesthetic** (Màu sắc - MAX 2 labels):
    - Màu sắc chủ đạo (Pink, Blue, Gold, Pastel, Black, Purple, Rainbow, Monochrome, etc.)

13. **Trademark Level** (Mức độ thương hiệu - 1 label):
    - Chọn 1 mức: No TM, Slight TM, TM, TM resemblance

14. **Main Subject** (Chủ thể chính - MAX 2 labels):
    - Đối tượng chính trong thiết kế (Rose, Butterfly, Truck, Police Badge, Animal, Text Design, etc.)

15. **Text** (Nội dung văn bản):
    - Ghi chính xác toàn bộ văn bản xuất hiện trên sản phẩm
    - Nếu không có văn bản, ghi "No text"

**OUTPUT FORMAT - BẮT BUỘC PHẢI ĐÚNG ĐỊNH DẠNG JSON:**
```json
{
  "image_recipient": ["value1", "value2"],
  "target_audience": ["value1", "value2"], 
  "usage_purpose": ["value1", "value2", "value3"],
  "occasion": ["value1", "value2"],
  "niche_theme": ["value1", "value2"],
  "sentiment_tone": ["value1", "value2"],
  "message_type": ["value1"],
  "personalization_type": ["value1"],
  "product_type": ["value1", "value2"],
  "placement_display_context": ["value1", "value2"],
  "design_style": ["value1", "value2"],
  "color_aesthetic": ["value1", "value2"],
  "trademark_level": "value",
  "main_subject": ["value1", "value2"],
  "text": ["value1", "value2"]
}
```

Hãy phân tích cẩn thận và trả về kết quả JSON chính xác.
"""

    def _download_image(self, url: str) -> bytes:
        """Download image từ URL với retry mechanism"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                response.raise_for_status()
                return response.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Lỗi download ảnh sau {max_retries} lần thử: {str(e)}")
                time.sleep(1)

    def _analyze_with_ollama(self, image_url: str) -> Dict:
        """Phân tích với GPT OSS 20B - Tối ưu hóa cho model mới"""
        try:
            print(f"🔍 Đang phân tích với {self.ollama_model}...")
            
            # Download và encode image
            image_bytes = self._download_image(image_url)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            prompt = self._create_labeling_prompt()

            # Cấu hình tối ưu cho GPT OSS 20B (model lớn hơn, cần parameters khác)
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                images=[image_base64],
                options={
                    'temperature': 0.05,  # Giảm temperature cho kết quả ổn định
                    'top_p': 0.85,       # Tối ưu cho model lớn
                    'num_ctx': 12288,    # Context length lớn hơn cho GPT OSS 20B
                    'repeat_penalty': 1.05,  # Giảm repeat penalty
                    'num_predict': 3072, # Tăng prediction tokens
                    'top_k': 30,        # Thêm top_k constraint
                    'seed': 42          # Fixed seed để reproducible
                }
            )

            content = response['response']
            
            # Parse JSON response với error handling tốt hơn
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                try:
                    result = json.loads(json_str)
                    print(f"✅ {self.ollama_model} phân tích thành công")
                    return result
                except json.JSONDecodeError as e:
                    print(f"⚠️  Lỗi parse JSON từ {self.ollama_model}: {str(e)}")
                    print(f"Raw response: {content[:500]}...")
                    raise Exception(f"Invalid JSON response from {self.ollama_model}")
            else:
                raise Exception(f"Không tìm thấy JSON trong response từ {self.ollama_model}")

        except Exception as e:
            raise Exception(f"Lỗi {self.ollama_model} analysis: {str(e)}")

    def _analyze_with_google(self, image_url: str) -> Dict:
        """Phân tích với Google Gemini"""
        if not self.google_client:
            raise Exception("Google client chưa được khởi tạo")

        try:
            print("🔍 Đang phân tích với Google Gemini...")
            
            image_bytes = self._download_image(image_url)
            image = Image.open(BytesIO(image_bytes))
            prompt = self._create_labeling_prompt()

            response = self.google_client.generate_content([prompt, image])
            content = response.text

            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                print("✅ Google Gemini phân tích thành công")
                return result
            else:
                raise Exception("Không tìm thấy JSON trong response từ Google")

        except Exception as e:
            raise Exception(f"Lỗi Google analysis: {str(e)}")

    def _merge_results(self, ollama_result: Dict, google_result: Dict) -> Dict:
        """Kết hợp kết quả từ 2 model với logic thông minh hơn"""
        merged = {}
        all_keys = set(ollama_result.keys()) | set(google_result.keys())

        for key in all_keys:
            ollama_values = ollama_result.get(key, [])
            google_values = google_result.get(key, [])

            if isinstance(ollama_values, list) and isinstance(google_values, list):
                # Kết hợp và loại bỏ duplicate, ưu tiên GPT OSS 20B
                combined = ollama_values + [v for v in google_values if v not in ollama_values]
                merged[key] = combined[:4]  # Limit to max 4 items
            elif isinstance(ollama_values, str) and isinstance(google_values, str):
                # Ưu tiên GPT OSS 20B cho string values
                merged[key] = ollama_values if ollama_values else google_values
            else:
                merged[key] = ollama_values if ollama_values else google_values

        return merged

    def label_image(self, image_url: str, provider: ModelProvider = ModelProvider.OLLAMA) -> ProductLabel:
        """Đánh label cho 1 ảnh sản phẩm - Mặc định dùng GPT OSS 20B"""
        try:
            if provider == ModelProvider.OLLAMA:
                result = self._analyze_with_ollama(image_url)
            elif provider == ModelProvider.GOOGLE:
                result = self._analyze_with_google(image_url)
            else:  # BOTH
                try:
                    ollama_result = self._analyze_with_ollama(image_url)
                except Exception as e:
                    print(f"⚠️  GPT OSS 20B failed, fallback to Google: {str(e)}")
                    result = self._analyze_with_google(image_url)
                else:
                    try:
                        google_result = self._analyze_with_google(image_url)
                        result = self._merge_results(ollama_result, google_result)
                    except Exception as e:
                        print(f"⚠️  Google failed, using GPT OSS 20B only: {str(e)}")
                        result = ollama_result

            return ProductLabel(
                image_url=image_url,
                image_recipient=result.get('image_recipient', []),
                target_audience=result.get('target_audience', []),
                usage_purpose=result.get('usage_purpose', []),
                occasion=result.get('occasion', []),
                niche_theme=result.get('niche_theme', []),
                sentiment_tone=result.get('sentiment_tone', []),
                message_type=result.get('message_type', []),
                personalization_type=result.get('personalization_type', []),
                product_type=result.get('product_type', []),
                placement_display_context=result.get('placement_display_context', []),
                design_style=result.get('design_style', []),
                color_aesthetic=result.get('color_aesthetic', []),
                trademark_level=result.get('trademark_level', 'No TM'),
                main_subject=result.get('main_subject', []),
                text=result.get('text', [])
            )

        except Exception as e:
            raise Exception(f"Lỗi labeling: {str(e)}")

    # === VECTOR GENERATION METHODS ===
    def _generate_vectors(self, text: str, image_url: str = None) -> tuple:
        """
        Tạo embedding vectors cho text và image sử dụng Jina v4

        Args:
            text: Text description để embedding
            image_url: URL của image để embedding

        Returns:
            tuple: (image_vector, text_vector)
        """
        # Sử dụng method từ EmbeddingService
        image_vector, text_vector = self.embedding_service._generate_vectors(
            text=text,
            image_url=image_url
        )

        print(f"✅ Tạo embedding thành công - Text: {len(text_vector)}D, Image: {len(image_vector)}D")
        return image_vector, text_vector

    def _generate_vectors_batch(self, descriptions: List[str], image_urls: List[str] = None) -> tuple:
        """
        Tạo embedding vectors cho nhiều text và image cùng lúc (hiệu quả hơn)

        Args:
            descriptions: List text descriptions
            image_urls: List image URLs (optional)

        Returns:
            tuple: (image_vectors_list, text_vectors_list)
        """
        # Batch embedding cho text
        text_vectors = self.embedding_service.embed_texts_batch(
            descriptions,
            normalize=True,
            batch_size=32
        )

        # Batch embedding cho images (nếu có)
        image_vectors = []
        if image_urls:
            for image_url in image_urls:
                if image_url and image_url.strip():
                    img_vec = self.embedding_service.embed_image(image_url, normalize=True)
                else:
                    img_vec = [0.0] * self.embedding_dim
                image_vectors.append(img_vec)
        else:
            image_vectors = [[0.0] * self.embedding_dim] * len(descriptions)

        print(f"✅ Tạo batch embedding thành công - {len(descriptions)} records")
        return image_vectors, text_vectors

    def _create_description(self, label: ProductLabel) -> str:
        """Tạo description dạng markdown từ ProductLabel"""

        def format_list(items: List[str]) -> str:
            if not items:
                return "Không xác định"
            return ", ".join(items)

        description = f"""# Mô Tả Sản Phẩm

## Thông Tin Cơ Bản
- **Chủ Thể Chính**: {format_list(label.main_subject)}
- **Loại Sản Phẩm**: {format_list(label.product_type)}
- **Mức Độ Thương Hiệu**: {label.trademark_level}

## Đối Tượng & Mục Đích
- **Người Nhận**: {format_list(label.image_recipient)}
- **Người Mua**: {format_list(label.target_audience)}
- **Mục Đích Sử Dụng**: {format_list(label.usage_purpose)}
- **Dịp Sử Dụng**: {format_list(label.occasion)}

## Phân Loại Sản Phẩm
- **Chủ Đề/Ngách**: {format_list(label.niche_theme)}
- **Cảm Xúc/Tông Điệu**: {format_list(label.sentiment_tone)}
- **Loại Thông Điệp**: {format_list(label.message_type)}
- **Cá Nhân Hóa**: {format_list(label.personalization_type)}
- **Nội Dung Chữ In**: {format_list(label.text)}

## Thiết Kế & Trưng Bày
- **Bối Cảnh Trưng Bày**: {format_list(label.placement_display_context)}
- **Phong Cách Thiết Kế**: {format_list(label.design_style)}
- **Thẩm Mỹ Màu Sắc**: {format_list(label.color_aesthetic)}

## Tóm Tắt
{format_list(label.product_type)} này là một {format_list(label.main_subject)} được thiết kế dành cho {format_list(label.image_recipient)}, phù hợp cho {format_list(label.occasion)} với phong cách {format_list(label.design_style)} và tông màu {format_list(label.color_aesthetic)}.
"""
        return description

    # === MAIN PROCESSING METHODS ===
    def process_single_record(self, raw_data: Dict[str, Any],
                              provider: ModelProvider = ModelProvider.OLLAMA) -> ProductRecord:
        """
        Xử lý 1 record: raw data → label → vectors → ProductRecord

        Args:
            raw_data: Data thô từ database
            provider: Model provider để labeling

        Returns:
            ProductRecord sẵn sàng để insert vào Milvus
        """
        try:
            image_url = raw_data.get('image', '')
            if not image_url:
                raise Exception("Không có URL ảnh")

            # 1. Label metadata
            label = self.label_image(image_url, provider)
            metadata = asdict(label)

            # 2. Tạo description markdown
            description = self._create_description(label)

            # 3. Tạo embedding vectors bằng Sentence Transformers
            image_vector, description_vector = self._generate_vectors(description, image_url)

            # 4. Tạo ProductRecord
            record = ProductRecord(
                id_sanpham=raw_data.get('id_sanpham', f"SP_{uuid.uuid4().hex[:8]}"),
                image_vector=image_vector,
                description_vector=description_vector,
                image=image_url,
                description=description,
                metadata=metadata,
                date=raw_data.get('date', ''),
                like=raw_data.get('like', '0'),
                comment=raw_data.get('comment', '0'),
                share=raw_data.get('share', '0'),
                link_redirect=raw_data.get('link_redirect', ''),
                platform=raw_data.get('platform', ''),
                name_store=raw_data.get('name_store', '')
            )

            return record

        except Exception as e:
            raise Exception(f"Lỗi xử lý record {raw_data.get('id_sanpham', 'unknown')}: {str(e)}")

    def process_batch_records(self, raw_data_list: List[Dict[str, Any]],
                              provider: ModelProvider = ModelProvider.OLLAMA) -> List[ProductRecord]:
        """
        Xử lý nhiều records cùng lúc để tối ưu batch embedding

        Args:
            raw_data_list: List data thô từ database
            provider: Model provider để labeling

        Returns:
            List ProductRecord sẵn sàng để insert vào Milvus
        """
        try:
            # Chuẩn bị data cho batch processing
            records = []
            descriptions = []
            image_urls = []

            # Tạo labels và descriptions cho tất cả records
            for raw_data in raw_data_list:
                try:
                    image_url = raw_data.get('image', '')
                    if not image_url:
                        continue

                    # Label metadata
                    label = self.label_image(image_url, provider)
                    metadata = asdict(label)

                    # Tạo description
                    description = self._create_description(label)

                    # Tạo record template (chưa có vectors)
                    record = ProductRecord(
                        id_sanpham=raw_data.get('id_sanpham', f"SP_{uuid.uuid4().hex[:8]}"),
                        image_vector=[],  # Sẽ được fill sau
                        description_vector=[],  # Sẽ được fill sau
                        image=image_url,
                        description=description,
                        metadata=metadata,
                        date=raw_data.get('date', ''),
                        like=raw_data.get('like', '0'),
                        comment=raw_data.get('comment', '0'),
                        share=raw_data.get('share', '0'),
                        link_redirect=raw_data.get('link_redirect', ''),
                        platform=raw_data.get('platform', ''),
                        name_store=raw_data.get('name_store', '')
                    )

                    records.append(record)
                    descriptions.append(description)
                    image_urls.append(image_url)

                except Exception as e:
                    print(f"Lỗi xử lý record {raw_data.get('id_sanpham', 'unknown')}: {str(e)}")
                    continue

            if not records:
                return []

            # Batch embedding cho tất cả descriptions và images
            print(f"🔄 Bắt đầu batch embedding cho {len(records)} records...")
            image_vectors, text_vectors = self._generate_vectors_batch(descriptions, image_urls)

            # Gán vectors vào records
            for i, record in enumerate(records):
                record.image_vector = image_vectors[i]
                record.description_vector = text_vectors[i]

            print(f"✅ Hoàn thành batch processing {len(records)} records")
            return records

        except Exception as e:
            print(f"❌ Lỗi batch processing: {str(e)}")
            return []

    def insert_record(self, record: ProductRecord) -> str:
        """
        Insert 1 ProductRecord vào Milvus

        Args:
            record: ProductRecord để insert

        Returns:
            ID của record đã insert
        """
        try:
            data = [
                [record.id_sanpham],
                [record.image_vector],
                [record.description_vector],
                [record.image],
                [record.description],
                [record.metadata],
                [record.date],
                [record.like],
                [record.comment],
                [record.share],
                [record.link_redirect],
                [record.platform],
                [record.name_store]
            ]

            mr = self.collection.insert(data)
            self.collection.flush()

            return record.id_sanpham

        except Exception as e:
            raise Exception(f"Lỗi insert record {record.id_sanpham}: {str(e)}")

    def insert_batch_records(self, records: List[ProductRecord]) -> List[str]:
        """
        Insert nhiều ProductRecord vào Milvus cùng lúc (hiệu quả hơn)

        Args:
            records: List ProductRecord để insert

        Returns:
            List ID của các record đã insert
        """
        try:
            if not records:
                return []

            # Chuẩn bị data cho batch insert
            ids = [record.id_sanpham for record in records]
            image_vectors = [record.image_vector for record in records]
            description_vectors = [record.description_vector for record in records]
            images = [record.image for record in records]
            descriptions = [record.description for record in records]
            metadatas = [record.metadata for record in records]
            dates = [record.date for record in records]
            likes = [record.like for record in records]
            comments = [record.comment for record in records]
            shares = [record.share for record in records]
            link_redirects = [record.link_redirect for record in records]
            platforms = [record.platform for record in records]
            name_stores = [record.name_store for record in records]

            data = [
                ids,
                image_vectors,
                description_vectors,
                images,
                descriptions,
                metadatas,
                dates,
                likes,
                comments,
                shares,
                link_redirects,
                platforms,
                name_stores
            ]

            mr = self.collection.insert(data)
            self.collection.flush()

            print(f"✅ Batch insert thành công {len(records)} records")
            return ids

        except Exception as e:
            raise Exception(f"Lỗi batch insert: {str(e)}")

    def run_pipeline(self, start_date: str, end_date: str,
                     limit: int = 1000,
                     provider: ModelProvider = ModelProvider.OLLAMA,
                     batch_size: int = 10) -> Dict[str, Any]:
        """
        Chạy pipeline hoàn chỉnh: Crawl → Check Duplicates → Label → Insert

        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            limit: Số lượng record tối đa
            provider: Model provider để labeling (mặc định GPT OSS 20B)
            batch_size: Số record xử lý mỗi batch

        Returns:
            Dictionary chứa thống kê kết quả
        """
        print("🚀 BẮT ĐẦU PIPELINE TÍCH HỢP VỚI GPT OSS 20B")
        print(f"📅 Thời gian: {start_date} → {end_date}")
        print(f"📊 Giới hạn: {limit} records")
        print(f"🤖 Provider: {provider.value}")
        print(f"🦾 Ollama Model: {self.ollama_model}")
        print("-" * 80)

        start_time = time.time()

        # Statistics
        stats = {
            'start_time': datetime.now().isoformat(),
            'ollama_model': self.ollama_model,
            'crawled_count': 0,
            'duplicate_count': 0,
            'processed_count': 0,
            'inserted_count': 0,
            'failed_count': 0,
            'skipped_duplicates': [],
            'inserted_ids': [],
            'failed_records': [],
            'total_time_seconds': 0
        }

        try:
            # STEP 1: Crawl data từ database
            print("📥 STEP 1: Crawl data từ database...")
            raw_data_list = self.crawl_data_by_date_range(start_date, end_date, limit)

            if not raw_data_list:
                print("⚠️  Không có data để xử lý")
                return stats

            stats['crawled_count'] = len(raw_data_list)
            print(f"✅ Crawl được {len(raw_data_list)} records")

            # STEP 2: Check duplicates và lọc bỏ
            print("🔍 STEP 2: Kiểm tra và lọc bỏ records trùng lặp...")
            new_records, existing_records, duplicate_count = self.filter_existing_records(raw_data_list)

            stats['duplicate_count'] = duplicate_count
            stats['skipped_duplicates'] = [record.get('id_sanpham', 'unknown') for record in existing_records]

            if not new_records:
                print("⚠️  Tất cả records đã tồn tại trong Milvus, không có gì để xử lý")
                return stats

            print(f"✅ Sẽ xử lý {len(new_records)} records mới với {self.ollama_model}")

            # STEP 3: Process từng record với batch
            print(f"🔄 STEP 3: Xử lý {len(new_records)} records mới với batch_size={batch_size}")

            for i in range(0, len(new_records), batch_size):
                batch = new_records[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(new_records) + batch_size - 1) // batch_size

                print(f"📦 Batch {batch_num}/{total_batches}: Xử lý {len(batch)} records")

                for j, raw_data in enumerate(batch):
                    record_idx = i + j + 1

                    try:
                        print(f"  🔍 [{record_idx}/{len(new_records)}] Xử lý: {raw_data.get('id_sanpham', 'unknown')}")

                        # Process single record: label + vector
                        record = self.process_single_record(raw_data, provider)
                        stats['processed_count'] += 1

                        # Insert vào Milvus
                        inserted_id = self.insert_record(record)
                        stats['inserted_count'] += 1
                        stats['inserted_ids'].append(inserted_id)

                        print(f"  ✅ [{record_idx}/{len(new_records)}] Thành công: {inserted_id}")

                        # Delay nhỏ để tránh rate limit
                        time.sleep(0.5)

                    except Exception as e:
                        stats['failed_count'] += 1
                        error_info = {
                            'id_sanpham': raw_data.get('id_sanpham', 'unknown'),
                            'image_url': raw_data.get('image', ''),
                            'error': str(e)
                        }
                        stats['failed_records'].append(error_info)
                        print(f"  ❌ [{record_idx}/{len(new_records)}] Lỗi: {str(e)}")
                        continue

                # Log batch progress
                print(f"📦 Batch {batch_num}/{total_batches} hoàn thành")
                print(f"   ✅ Thành công: {stats['processed_count']}/{len(new_records)}")
                print(f"   ❌ Thất bại: {stats['failed_count']}/{len(new_records)}")

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng trong pipeline: {str(e)}")

        finally:
            # Tính toán thời gian
            end_time = time.time()
            stats['total_time_seconds'] = round(end_time - start_time, 2)
            stats['end_time'] = datetime.now().isoformat()

            # Log kết quả cuối cùng
            print("=" * 80)
            print("🎊 PIPELINE HOÀN THÀNH!")
            print(f"📊 THỐNG KÊ TỔNG KẾT:")
            print(f"   🦾 Model sử dụng: {self.ollama_model}")
            print(f"   📥 Crawl: {stats['crawled_count']} records")
            print(f"   🔄 Trùng lặp (bỏ qua): {stats['duplicate_count']} records")
            print(f"   🆕 Records mới: {len(new_records) if 'new_records' in locals() else 0} records")
            print(f"   🔄 Xử lý: {stats['processed_count']} records")
            print(f"   ✅ Insert thành công: {stats['inserted_count']} records")
            print(f"   ❌ Thất bại: {stats['failed_count']} records")
            print(f"   ⏱️  Tổng thời gian: {stats['total_time_seconds']}s")

            # Tính tỉ lệ thành công trên records mới (không tính trùng lặp)
            new_records_count = len(new_records) if 'new_records' in locals() else max(
                stats['crawled_count'] - stats['duplicate_count'], 1)
            success_rate = stats['inserted_count'] / max(new_records_count, 1) * 100
            print(f"   📈 Tỉ lệ thành công: {success_rate:.1f}%")

            # Hiển thị collection stats
            try:
                total_entities = self.collection.num_entities
                print(f"   💾 Tổng entities trong Milvus: {total_entities}")
            except:
                pass

            print("=" * 80)

            return stats

    def search_similar_products(self, query_vector: List[float],
                                field_name: str = "image_vector",
                                top_k: int = 5):
        """
        Tìm kiếm sản phẩm tương tự

        Args:
            query_vector: Vector để search
            field_name: Field vector để search ("image_vector" hoặc "description_vector")
            top_k: Số kết quả trả về
        """
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field=field_name,
            param=search_params,
            limit=top_k,
            output_fields=["id_sanpham", "image", "platform", "metadata"]
        )

        return results

    def save_stats_to_json(self, stats: Dict[str, Any], filename: str = None):
        """Lưu thống kê vào file JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_stats_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"💾 Đã lưu thống kê vào: {filename}")
        except Exception as e:
            print(f"❌ Lỗi lưu file: {e}")

    def close_connections(self):
        """Đóng tất cả kết nối"""
        try:
            if self.db_connection:
                self.db_connection.close()
                print("✅ Đã đóng kết nối PostgreSQL")
        except:
            pass


def main():
    """
    Hàm main được cập nhật cho GPT OSS 20B
    """
    print("🚀 KHỞI ĐỘNG INTEGRATED PRODUCT PIPELINE - GPT OSS 20B + GOOGLE")
    print("=" * 60)

    # ========== CONFIGURATION ==========
    # Database config
    db_config = {
        'host': '45.79.189.110',
        'database': 'ai_db',
        'user': 'ai_engineer',
        'password': 'StrongPassword123',
        'port': 5432
    }

    # API Keys
    google_api_key = "AIzaSyC0-LkawNpB_krGzPmR6fe7zpFyk476LGY"

    # Ollama config - Updated to GPT OSS 20B
    ollama_model = "gpt-oss:20b"  # Model mới được cập nhật

    # Milvus config
    milvus_host = "10.10.4.25"
    milvus_port = "19530"

    # ========== THỜI GIAN CRAWL ==========
    start_date = "2025-07-01"  # YYYY-MM-DD
    end_date = "2025-08-14"   # YYYY-MM-DD

    # ========== CÀI ĐẶT PIPELINE ==========
    limit = 5000  # Giới hạn số record để test
    provider = ModelProvider.OLLAMA  # Mặc định dùng GPT OSS 20B
    batch_size = 10  # Số record xử lý mỗi batch

    try:
        # Khởi tạo pipeline
        print("🔧 Khởi tạo pipeline...")
        pipeline = IntegratedProductPipeline(
            db_config=db_config,
            google_api_key=google_api_key,
            ollama_model=ollama_model,
            milvus_host=milvus_host,
            milvus_port=milvus_port
        )

        print("✅ Pipeline khởi tạo thành công!")

        # Chạy pipeline chính
        print(f"🎯 Bắt đầu crawl data từ {start_date} đến {end_date}")

        stats = pipeline.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            provider=provider,
            batch_size=batch_size
        )

        # Lưu thống kê
        pipeline.save_stats_to_json(stats)

        # Hiển thị kết quả ngắn gọn
        print("\n🎊 KẾT QUẢ CUỐI CÙNG:")
        print(f"🦾 Model: {stats.get('ollama_model', 'N/A')}")
        print(f"✅ Thành công: {stats['inserted_count']}/{stats['crawled_count']} records")
        print(f"🔄 Trùng lặp (bỏ qua): {stats['duplicate_count']} records")
        print(f"⏱️  Thời gian: {stats['total_time_seconds']}s")

        if stats['inserted_ids']:
            print(f"📦 Một số ID đã insert:")
            for i, record_id in enumerate(stats['inserted_ids'][:5]):
                print(f"   {i + 1}. {record_id}")
            if len(stats['inserted_ids']) > 5:
                print(f"   ... và {len(stats['inserted_ids']) - 5} records khác")

        if stats['skipped_duplicates']:
            print(f"🔄 Một số ID trùng lặp (đã bỏ qua):")
            for i, record_id in enumerate(stats['skipped_duplicates'][:5]):
                print(f"   {i + 1}. {record_id}")
            if len(stats['skipped_duplicates']) > 5:
                print(f"   ... và {len(stats['skipped_duplicates']) - 5} records khác")

    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: {str(e)}")

    finally:
        try:
            pipeline.close_connections()
        except:
            pass

        print("\n👋 Pipeline kết thúc!")


if __name__ == "__main__":
    main()