import psycopg2
import os
import json
import requests
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta
import uuid
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import time
from embedding_service import EmbeddingService
import ollama
import concurrent.futures
from functools import lru_cache
import threading


# Configuration
class ModelProvider(Enum):
    OLLAMA = "ollama"


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


class MoELabeler:
    """Mixture of Experts Labeler cho phân tích sản phẩm nhanh"""

    def __init__(self, ollama_model: str = "gpt-oss:20b"):
        self.model = ollama_model
        self.experts = {
            'basic_info': self._create_basic_info_prompt,
            'audience_purpose': self._create_audience_purpose_prompt,
            'design_style': self._create_design_style_prompt,
            'product_classification': self._create_product_classification_prompt
        }

        # Cache cho image data để tránh download nhiều lần
        self._image_cache = {}
        self._cache_lock = threading.Lock()

    @lru_cache(maxsize=4)
    def _create_basic_info_prompt(self) -> str:
        """Expert 1: Thông tin cơ bản của sản phẩm"""
        return """
Bạn là expert phân tích THÔNG TIN CƠ BẢN sản phẩm. Hãy tập trung vào:

**CHỈ PHÂN TÍCH:**
1. **Product Type** (1-2 labels): Loại sản phẩm cụ thể (Mug, Hoodie, Watch, Keychain, etc.)
2. **Main Subject** (1-2 labels): Đối tượng chính trong thiết kế (Rose, Butterfly, Truck, etc.)
3. **Text** (tất cả text): Toàn bộ văn bản trên sản phẩm
4. **Trademark Level** (1 label): No TM, Slight TM, TM, TM resemblance

**OUTPUT JSON:**
```json
{
  "product_type": ["value1", "value2"],
  "main_subject": ["value1", "value2"],
  "text": ["value1", "value2"],
  "trademark_level": "value"
}
```
"""

    @lru_cache(maxsize=4)
    def _create_audience_purpose_prompt(self) -> str:
        """Expert 2: Đối tượng và mục đích"""
        return """
Bạn là expert phân tích ĐỐI TƯỢNG VÀ MỤC ĐÍCH sản phẩm. Hãy tập trung vào:

**CHỈ PHÂN TÍCH:**
1. **Image Recipient** (1-3 labels): Người nhận cụ thể (Mom, Dad, Son, Daughter - KHÔNG dùng Children)
2. **Target Audience** (1-3 labels): Người mua cụ thể (From Daughter, From Son, From Wife, etc.)
3. **Usage Purpose** (1-3 labels): Mục đích sử dụng (Gift, Home Decor, Personal Use, etc.)
4. **Occasion** (1-3 labels): Dịp cụ thể (Mother's Birthday, Father's Day, Christmas Gift, etc.)

**OUTPUT JSON:**
```json
{
  "image_recipient": ["value1", "value2"],
  "target_audience": ["value1", "value2"], 
  "usage_purpose": ["value1", "value2"],
  "occasion": ["value1", "value2"]
}
```
"""

    @lru_cache(maxsize=4)
    def _create_design_style_prompt(self) -> str:
        """Expert 3: Thiết kế và phong cách"""
        return """
Bạn là expert phân tích THIẾT KẾ VÀ PHONG CÁCH sản phẩm. Hãy tập trung vào:

**CHỈ PHÂN TÍCH:**
1. **Design Style** (1-3 labels): Phong cách (Elegant, Vintage, Stained Glass - CHÚ Ý: 3D Rendered chỉ khi thiết kế thực sự là 3D)
2. **Color Aesthetic** (1-2 labels): Màu sắc chủ đạo (Pink, Blue, Gold, Pastel, etc.)
3. **Placement Display Context** (1-2 labels): Bối cảnh trưng bày (Shelf decor, Desk decor, etc.)
4. **Sentiment Tone** (1-2 labels): Cảm xúc (Sentimental, Humorous, Elegant, etc.)

**OUTPUT JSON:**
```json
{
  "design_style": ["value1", "value2"],
  "color_aesthetic": ["value1", "value2"],
  "placement_display_context": ["value1", "value2"],
  "sentiment_tone": ["value1", "value2"]
}
```
"""

    @lru_cache(maxsize=4)
    def _create_product_classification_prompt(self) -> str:
        """Expert 4: Phân loại sản phẩm"""
        return """
Bạn là expert PHÂN LOẠI SAN PHẨM chi tiết. Hãy tập trung vào:

**CHỈ PHÂN TÍCH:**
1. **Niche Theme** (1-2 labels): Chủ đề (Mother, Father, Police, Beer, Cowgirl, etc.)
2. **Message Type** (1 label): Loại thông điệp (No quote, Symbolic Message, etc.)
3. **Personalization Type** (1 label): Cá nhân hóa (Personalized Name, Non-personalized)

**OUTPUT JSON:**
```json
{
  "niche_theme": ["value1", "value2"],
  "message_type": ["value1"],
  "personalization_type": ["value1"]
}
```
"""

    def _get_cached_image(self, image_url: str) -> str:
        """Lấy image từ cache hoặc download và cache"""
        with self._cache_lock:
            if image_url in self._image_cache:
                return self._image_cache[image_url]

            try:
                # Download image
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image_base64 = base64.b64encode(response.content).decode('utf-8')

                # Cache với giới hạn
                if len(self._image_cache) < 50:  # Giới hạn cache
                    self._image_cache[image_url] = image_base64

                return image_base64
            except Exception as e:
                raise Exception(f"Lỗi download ảnh: {str(e)}")

    def _analyze_with_expert(self, expert_name: str, image_url: str) -> Dict:
        """Phân tích với 1 expert cụ thể"""
        try:
            image_base64 = self._get_cached_image(image_url)
            prompt = self.experts[expert_name]()

            # Tối ưu options cho tốc độ
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                images=[image_base64],
                options={
                    'temperature': 0.1,
                    'top_p': 0.8,
                    'num_ctx': 4096,  # Giảm context length
                    'repeat_penalty': 1.05,
                    'num_predict': 512,  # Giảm số token predict
                    'num_thread': 8,  # Tăng số thread
                }
            )

            content = response['response']
            json_start = content.find('{')
            json_end = content.rfind('}') + 1

            if json_start != -1 and json_end != -1:
                json_str = content[json_start:json_end]
                result = json.loads(json_str)
                return result
            else:
                print(f"⚠️  Expert {expert_name} không trả về JSON hợp lệ")
                return {}

        except Exception as e:
            print(f"⚠️  Lỗi expert {expert_name}: {str(e)}")
            return {}

    def analyze_parallel(self, image_url: str, timeout: int = 4) -> Dict:
        """Chạy tất cả experts song song với timeout"""
        results = {}

        def run_expert(expert_name: str) -> tuple:
            try:
                result = self._analyze_with_expert(expert_name, image_url)
                return expert_name, result
            except Exception as e:
                return expert_name, {}

        # Chạy song song với ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_expert = {
                executor.submit(run_expert, expert_name): expert_name
                for expert_name in self.experts.keys()
            }

            completed_futures = concurrent.futures.as_completed(
                future_to_expert,
                timeout=timeout
            )

            for future in completed_futures:
                try:
                    expert_name, result = future.result(timeout=1)
                    results.update(result)
                except concurrent.futures.TimeoutError:
                    expert_name = future_to_expert[future]
                    print(f"⚠️  Expert {expert_name} timeout")
                except Exception as e:
                    expert_name = future_to_expert[future]
                    print(f"⚠️  Expert {expert_name} lỗi: {str(e)}")

        return results

    def analyze_sequential_fallback(self, image_url: str) -> Dict:
        """Fallback: chạy tuần tự nếu song song thất bại"""
        results = {}

        for expert_name in ['basic_info', 'audience_purpose']:  # Chỉ chạy 2 experts quan trọng nhất
            try:
                result = self._analyze_with_expert(expert_name, image_url)
                results.update(result)
            except Exception as e:
                print(f"⚠️  Expert {expert_name} fallback lỗi: {str(e)}")

        return results


class OptimizedProductPipeline:
    """Pipeline tối ưu chỉ sử dụng GPT-OSS 20B với MoE"""

    def __init__(self,
                 db_config: Dict[str, str],
                 ollama_model: str = "gpt-oss:20b",
                 milvus_host: str = "10.10.4.25",
                 milvus_port: str = "19530"):
        """
        Khởi tạo pipeline tối ưu với MoE

        Args:
            db_config: Cấu hình database PostgreSQL
            ollama_model: Model Ollama (GPT_OSS_20B)
            milvus_host: Milvus host
            milvus_port: Milvus port
        """
        # Database config
        self.db_config = db_config
        self.db_connection = None

        # MoE Labeler
        print("🔧 Khởi tạo MoE Labeler...")
        self.moe_labeler = MoELabeler(ollama_model)

        # Khởi tạo EmbeddingService với Jina v4
        print("🔧 Khởi tạo Jina v4 Embedding Service...")
        self.embedding_service = EmbeddingService()
        self.embedding_dim = self.embedding_service.embedding_dim

        # Milvus config
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "product_collection_GP_optimized"
        self.collection = None

        # Log info
        model_info = self.embedding_service.get_model_info()
        print(f"🤖 Embedding Model: {model_info['model_name']}")
        print(f"📊 Embedding Dimensions: {model_info['embedding_dimension']}")
        print(f"🔧 Device: {model_info['device']}")
        print(f"🦙 Ollama Model: {ollama_model}")

        # Initialize connections
        self._connect_db()
        self._connect_milvus()
        self._setup_collection()

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
        """Tạo schema cho collection"""
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
            description=f"Optimized collection với embedding {self.embedding_dim}D và MoE labeling"
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
    def check_ids_exist_batch(self, id_list: List[str]) -> Dict[str, bool]:
        """Kiểm tra nhiều ID cùng lúc"""
        try:
            if not id_list:
                return {}

            # Batch check với chunks để tránh query quá lớn
            chunk_size = 100
            all_existing_ids = set()

            for i in range(0, len(id_list), chunk_size):
                chunk = id_list[i:i + chunk_size]
                id_conditions = [f'id_sanpham == "{id_val}"' for id_val in chunk]
                expr = " or ".join(id_conditions)

                results = self.collection.query(
                    expr=expr,
                    output_fields=["id_sanpham"],
                    limit=len(chunk)
                )

                chunk_existing = {result["id_sanpham"] for result in results}
                all_existing_ids.update(chunk_existing)

            return {id_val: id_val in all_existing_ids for id_val in id_list}

        except Exception as e:
            print(f"⚠️  Lỗi kiểm tra batch IDs: {str(e)}")
            return {id_val: False for id_val in id_list}

    def filter_existing_records(self, raw_data_list: List[Dict[str, Any]]) -> tuple:
        """Lọc bỏ các record đã tồn tại"""
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

            print(f"✅ Kết quả: {len(new_records)} mới, {duplicate_count} trùng lặp")
            return new_records, existing_records, duplicate_count

        except Exception as e:
            print(f"❌ Lỗi lọc records: {str(e)}")
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
            print(f"❌ Lỗi crawl data: {e}")
            return []

    # === OPTIMIZED LABELING METHODS ===
    def label_image_fast(self, image_url: str) -> ProductLabel:
        """Đánh label nhanh với MoE - mục tiêu 5 giây"""
        start_time = time.time()

        try:
            # Thử chạy song song trước (4 giây timeout)
            result = self.moe_labeler.analyze_parallel(image_url, timeout=4)

            # Nếu không đủ data, chạy fallback
            if len(result) < 5:  # Ít nhất 5 fields
                print("🔄 Chạy fallback sequential...")
                fallback_result = self.moe_labeler.analyze_sequential_fallback(image_url)
                result.update(fallback_result)

            # Fill missing fields với defaults
            result = self._fill_missing_fields(result)

            # Tạo ProductLabel
            label = ProductLabel(
                image_url=image_url,
                image_recipient=result.get('image_recipient', ['Family']),
                target_audience=result.get('target_audience', ['General']),
                usage_purpose=result.get('usage_purpose', ['Gift']),
                occasion=result.get('occasion', ['Any Occasion']),
                niche_theme=result.get('niche_theme', ['General']),
                sentiment_tone=result.get('sentiment_tone', ['Positive']),
                message_type=result.get('message_type', ['No quote']),
                personalization_type=result.get('personalization_type', ['Non-personalized']),
                product_type=result.get('product_type', ['General Product']),
                placement_display_context=result.get('placement_display_context', ['Home Decor']),
                design_style=result.get('design_style', ['Modern']),
                color_aesthetic=result.get('color_aesthetic', ['Colorful']),
                trademark_level=result.get('trademark_level', 'No TM'),
                main_subject=result.get('main_subject', ['Product']),
                text=result.get('text', ['No text'])
            )

            elapsed_time = time.time() - start_time
            print(f"⚡ Labeling hoàn thành trong {elapsed_time:.1f}s")

            return label

        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"⚠️  Labeling lỗi sau {elapsed_time:.1f}s: {str(e)}")

            # Return default label
            return self._create_default_label(image_url)

    def _fill_missing_fields(self, result: Dict) -> Dict:
        """Fill các fields bị thiếu với giá trị default"""
        defaults = {
            'image_recipient': ['Family'],
            'target_audience': ['General'],
            'usage_purpose': ['Gift'],
            'occasion': ['Any Occasion'],
            'niche_theme': ['General'],
            'sentiment_tone': ['Positive'],
            'message_type': ['No quote'],
            'personalization_type': ['Non-personalized'],
            'product_type': ['General Product'],
            'placement_display_context': ['Home Decor'],
            'design_style': ['Modern'],
            'color_aesthetic': ['Colorful'],
            'trademark_level': 'No TM',
            'main_subject': ['Product'],
            'text': ['No text']
        }

        for key, default_value in defaults.items():
            if key not in result or not result[key]:
                result[key] = default_value

        return result

    def _create_default_label(self, image_url: str) -> ProductLabel:
        """Tạo label default khi lỗi"""
        return ProductLabel(
            image_url=image_url,
            image_recipient=['Family'],
            target_audience=['General'],
            usage_purpose=['Gift'],
            occasion=['Any Occasion'],
            niche_theme=['General'],
            sentiment_tone=['Positive'],
            message_type=['No quote'],
            personalization_type=['Non-personalized'],
            product_type=['General Product'],
            placement_display_context=['Home Decor'],
            design_style=['Modern'],
            color_aesthetic=['Colorful'],
            trademark_level='No TM',
            main_subject=['Product'],
            text=['No text']
        )

    # === VECTOR GENERATION METHODS ===
    def _generate_vectors(self, text: str, image_url: str = None) -> tuple:
        """Tạo embedding vectors cho text và image"""
        image_vector, text_vector = self.embedding_service._generate_vectors(
            text=text,
            image_url=image_url
        )
        return image_vector, text_vector

    def _create_description(self, label: ProductLabel) -> str:
        """Tạo description ngắn gọn từ ProductLabel"""

        def format_list(items: List[str]) -> str:
            if not items:
                return "Không xác định"
            return ", ".join(items[:3])  # Chỉ lấy 3 items đầu

        # Description ngắn gọn hơn để tăng tốc độ
        description = f"""# {format_list(label.product_type)}

Chủ thể: {format_list(label.main_subject)}
Dành cho: {format_list(label.image_recipient)}
Dịp: {format_list(label.occasion)}
Phong cách: {format_list(label.design_style)}
Màu sắc: {format_list(label.color_aesthetic)}
Text: {format_list(label.text)}

{format_list(label.product_type)} với thiết kế {format_list(label.design_style)}, phù hợp cho {format_list(label.occasion)}.
"""
        return description

    # === MAIN PROCESSING METHODS ===
    def process_single_record(self, raw_data: Dict[str, Any]) -> ProductRecord:
        """Xử lý 1 record với tốc độ cao"""
        record_start_time = time.time()

        try:
            image_url = raw_data.get('image', '')
            if not image_url:
                raise Exception("Không có URL ảnh")

            # 1. Label metadata với MoE (mục tiêu 4s)
            label = self.label_image_fast(image_url)
            metadata = asdict(label)

            # 2. Tạo description ngắn (mục tiêu 0.1s)
            description = self._create_description(label)

            # 3. Tạo embedding vectors (mục tiêu 0.9s)
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

            elapsed_time = time.time() - record_start_time
            print(f"⚡ Record processed in {elapsed_time:.1f}s")

            return record

        except Exception as e:
            elapsed_time = time.time() - record_start_time
            raise Exception(
                f"Lỗi xử lý record {raw_data.get('id_sanpham', 'unknown')} sau {elapsed_time:.1f}s: {str(e)}")

    def insert_record(self, record: ProductRecord) -> str:
        """Insert 1 ProductRecord vào Milvus"""
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
        """Insert nhiều ProductRecord vào Milvus cùng lúc"""
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

    def run_optimized_pipeline(self, start_date: str, end_date: str,
                               limit: int = 1000,
                               batch_size: int = 5,
                               max_workers: int = 3) -> Dict[str, Any]:
        """
        Chạy pipeline tối ưu với xử lý song song có giới hạn

        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            limit: Số lượng record tối đa
            batch_size: Số record xử lý mỗi batch
            max_workers: Số thread tối đa (khuyến nghị 2-3 để tránh quá tải Ollama)

        Returns:
            Dictionary chứa thống kê kết quả
        """
        print("🚀 BẮT ĐẦU OPTIMIZED PIPELINE - GPT-OSS 20B + MoE")
        print(f"📅 Thời gian: {start_date} → {end_date}")
        print(f"📊 Giới hạn: {limit} records, batch_size: {batch_size}, workers: {max_workers}")
        print("-" * 80)

        start_time = time.time()

        # Statistics
        stats = {
            'start_time': datetime.now().isoformat(),
            'crawled_count': 0,
            'duplicate_count': 0,
            'processed_count': 0,
            'inserted_count': 0,
            'failed_count': 0,
            'skipped_duplicates': [],
            'inserted_ids': [],
            'failed_records': [],
            'total_time_seconds': 0,
            'avg_processing_time_per_record': 0,
            'performance_stats': {
                'labeling_times': [],
                'embedding_times': [],
                'insert_times': []
            }
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
                print("⚠️  Tất cả records đã tồn tại trong Milvus")
                return stats

            print(f"✅ Sẽ xử lý {len(new_records)} records mới")

            # STEP 3: Xử lý records với parallel processing có giới hạn
            print(f"🔄 STEP 3: Xử lý với {max_workers} workers song song...")

            def process_record_wrapper(raw_data: Dict[str, Any]) -> tuple:
                """Wrapper function cho parallel processing"""
                record_id = raw_data.get('id_sanpham', 'unknown')
                try:
                    record_start = time.time()
                    record = self.process_single_record(raw_data)
                    processing_time = time.time() - record_start

                    return 'success', record, record_id, processing_time, None
                except Exception as e:
                    return 'error', None, record_id, 0, str(e)

            # Xử lý với ThreadPoolExecutor có giới hạn workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tất cả tasks
                future_to_data = {
                    executor.submit(process_record_wrapper, raw_data): raw_data
                    for raw_data in new_records
                }

                # Collect results
                for i, future in enumerate(concurrent.futures.as_completed(future_to_data)):
                    try:
                        status, record, record_id, processing_time, error = future.result()

                        if status == 'success':
                            # Insert record
                            insert_start = time.time()
                            inserted_id = self.insert_record(record)
                            insert_time = time.time() - insert_start

                            stats['processed_count'] += 1
                            stats['inserted_count'] += 1
                            stats['inserted_ids'].append(inserted_id)
                            stats['performance_stats']['labeling_times'].append(processing_time)
                            stats['performance_stats']['insert_times'].append(insert_time)

                            print(f"✅ [{i + 1}/{len(new_records)}] Thành công: {record_id} ({processing_time:.1f}s)")

                        else:
                            stats['failed_count'] += 1
                            error_info = {
                                'id_sanpham': record_id,
                                'error': error
                            }
                            stats['failed_records'].append(error_info)
                            print(f"❌ [{i + 1}/{len(new_records)}] Lỗi: {record_id} - {error}")

                    except Exception as e:
                        stats['failed_count'] += 1
                        print(f"❌ [{i + 1}/{len(new_records)}] Lỗi nghiêm trọng: {str(e)}")

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng trong pipeline: {str(e)}")

        finally:
            # Tính toán thời gian và thống kê
            end_time = time.time()
            stats['total_time_seconds'] = round(end_time - start_time, 2)
            stats['end_time'] = datetime.now().isoformat()

            # Tính average processing time
            if stats['processed_count'] > 0:
                stats['avg_processing_time_per_record'] = round(
                    sum(stats['performance_stats']['labeling_times']) / stats['processed_count'], 2
                )

            # Log kết quả cuối cùng
            print("=" * 80)
            print("🎊 OPTIMIZED PIPELINE HOÀN THÀNH!")
            print(f"📊 THỐNG KÊ TỔNG KẾT:")
            print(f"   📥 Crawl: {stats['crawled_count']} records")
            print(f"   🔄 Trùng lặp (bỏ qua): {stats['duplicate_count']} records")
            print(f"   🆕 Records mới: {len(new_records) if 'new_records' in locals() else 0} records")
            print(f"   🔄 Xử lý: {stats['processed_count']} records")
            print(f"   ✅ Insert thành công: {stats['inserted_count']} records")
            print(f"   ❌ Thất bại: {stats['failed_count']} records")
            print(f"   ⏱️  Tổng thời gian: {stats['total_time_seconds']}s")
            print(f"   📈 Avg thời gian/record: {stats['avg_processing_time_per_record']}s")

            # Performance analysis
            if stats['performance_stats']['labeling_times']:
                avg_labeling = sum(stats['performance_stats']['labeling_times']) / len(
                    stats['performance_stats']['labeling_times'])
                max_labeling = max(stats['performance_stats']['labeling_times'])
                min_labeling = min(stats['performance_stats']['labeling_times'])
                print(
                    f"   🏷️  Labeling time - Avg: {avg_labeling:.1f}s, Min: {min_labeling:.1f}s, Max: {max_labeling:.1f}s")

            # Tính tỉ lệ thành công
            new_records_count = len(new_records) if 'new_records' in locals() else max(
                stats['crawled_count'] - stats['duplicate_count'], 1)
            success_rate = stats['inserted_count'] / max(new_records_count, 1) * 100
            print(f"   📈 Tỉ lệ thành công: {success_rate:.1f}%")

            # Performance target check
            target_time = 5.0  # 5 seconds per record target
            if stats['avg_processing_time_per_record'] <= target_time:
                print(f"   🎯 MỤC TIÊU ĐẠT: {stats['avg_processing_time_per_record']:.1f}s ≤ {target_time}s")
            else:
                print(f"   ⚠️  MỤC TIÊU CHƯA ĐẠT: {stats['avg_processing_time_per_record']:.1f}s > {target_time}s")

            print("=" * 80)

            return stats

    def search_similar_products(self, query_vector: List[float],
                                field_name: str = "image_vector",
                                top_k: int = 5):
        """Tìm kiếm sản phẩm tương tự"""
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
            filename = f"optimized_pipeline_stats_{timestamp}.json"

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

        # Clear image cache in MoE labeler
        try:
            self.moe_labeler._image_cache.clear()
            print("✅ Đã clear image cache")
        except:
            pass

def main():
    """
    Hàm main tối ưu - chỉ GPT-OSS 20B với MoE
    Mục tiêu: 5 giây/record
    """
    print("🚀 KHỞI ĐỘNG OPTIMIZED PIPELINE - GPT-OSS 20B + MoE")
    print("🎯 Mục tiêu: 5 giây/record")
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

    # Ollama config - Chỉ GPT_OSS_20B
    ollama_model = "gpt-oss:20b"

    # Milvus config
    milvus_host = "10.10.4.25"
    milvus_port = "19530"

    # ========== THỜI GIAN CRAWL ==========
    start_date = "2024-08-12"  # YYYY-MM-DD
    end_date = "2025-08-12"  # YYYY-MM-DD

    # ========== CÀI ĐẶT PIPELINE TỐI ƯU ==========
    limit = 100  # Giới hạn số record để test performance
    batch_size = 5  # Batch size nhỏ hơn để kiểm soát memory
    max_workers = 2  # Giới hạn workers để tránh quá tải Ollama

    try:
        # Khởi tạo pipeline
        print("🔧 Khởi tạo optimized pipeline...")
        pipeline = OptimizedProductPipeline(
            db_config=db_config,
            ollama_model=ollama_model,
            milvus_host=milvus_host,
            milvus_port=milvus_port
        )

        print("✅ Pipeline khởi tạo thành công!")

        # Performance check trước khi chạy
        print("🧪 Đang test performance với 1 record mẫu...")
        test_data = [{'id_sanpham': 'TEST_001', 'image': 'https://example.com/test.jpg'}]
        test_start = time.time()

        # Bỏ qua test nếu không có image URL thực
        print("⚠️  Bỏ qua test performance, chạy pipeline thực...")

        # Chạy pipeline chính
        print(f"🎯 Bắt đầu crawl data từ {start_date} đến {end_date}")

        stats = pipeline.run_optimized_pipeline(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            batch_size=batch_size,
            max_workers=max_workers
        )

        # Lưu thống kê
        pipeline.save_stats_to_json(stats)

        # Performance report
        print("\n🎊 BÁO CÁO PERFORMANCE:")
        print(f"✅ Thành công: {stats['inserted_count']}/{stats['crawled_count']} records")
        print(f"🔄 Trùng lặp: {stats['duplicate_count']} records")
        print(f"⏱️  Avg time/record: {stats['avg_processing_time_per_record']}s")
        print(f"🎯 Target: 5.0s/record")

        if stats['avg_processing_time_per_record'] <= 5.0:
            print("🎉 ĐẠT MỤC TIÊU PERFORMANCE!")
        else:
            print("⚠️  CHƯA ĐẠT MỤC TIÊU - Cần tối ưu thêm")

        # Đề xuất tối ưu
        if stats['avg_processing_time_per_record'] > 5.0:
            print("\n💡 ĐỀ XUẤT TỐI ƯU:")
            print("   - Giảm context length trong Ollama options")
            print("   - Tăng num_thread trong Ollama")
            print("   - Giảm số experts trong MoE")
            print("   - Sử dụng image cache hiệu quả hơn")
            print("   - Giảm timeout cho parallel execution")

    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: {str(e)}")

    finally:
        try:
            pipeline.close_connections()
        except:
            pass

        print("\n👋 Optimized Pipeline kết thúc!")


if __name__ == "__main__":
    main()