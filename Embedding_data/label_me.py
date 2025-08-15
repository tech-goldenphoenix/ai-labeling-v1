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
import torch
import re
import concurrent.futures
from threading import Lock

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
    """Module tích hợp: Crawl Data → Label → Insert Milvus với Jina v4 - OPTIMIZED"""

    def __init__(self,
                 db_config: Dict[str, str],
                 google_api_key: str,
                 ollama_model: str = "gpt-oss:20b",
                 milvus_host: str = "10.10.4.25",
                 milvus_port: str = "19530",
                 max_workers: int = 5):
        """
        Khởi tạo pipeline tích hợp với optimizations

        Args:
            db_config: Cấu hình database PostgreSQL
            google_api_key: API key cho Google Gemini
            ollama_model: Model Ollama để sử dụng
            milvus_host: Milvus host
            milvus_port: Milvus port
            max_workers: Số thread đồng thời để labeling
        """
        # Database config
        self.db_config = db_config
        self.db_connection = None

        # AI Labeler
        self.ollama_model = ollama_model
        self.google_client = None
        self.max_workers = max_workers
        self._lock = Lock()  # Thread lock cho thread-safe operations

        # Khởi tạo EmbeddingService với Jina v4
        print("🔧 Khởi tạo Jina v4 Embedding Service...")
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

        # Cache cho image downloads
        self.image_cache = {}
        self.cache_lock = Lock()

        # Log embedding model info
        model_info = self.embedding_service.get_model_info()
        print(f"🤖 Embedding Model: {model_info['model_name']}")
        print(f"📊 Embedding Dimensions: {model_info['embedding_dimension']}")
        print(f"🔧 Device: {model_info['device']}")
        print(f"🦙 Ollama Model: {self.ollama_model}")
        print(f"🚀 Max Workers: {self.max_workers}")

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
        """Kiểm tra nhiều ID cùng lúc để tối ưu performance"""
        try:
            if not id_list:
                return {}

            # Chia nhỏ để tránh query quá lớn
            batch_size = 100
            all_results = {}
            
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i + batch_size]
                id_conditions = [f'id_sanpham == "{id_val}"' for id_val in batch_ids]
                expr = " or ".join(id_conditions)

                results = self.collection.query(
                    expr=expr,
                    output_fields=["id_sanpham"],
                    limit=len(batch_ids)
                )

                existing_ids = {result["id_sanpham"] for result in results}
                batch_results = {id_val: id_val in existing_ids for id_val in batch_ids}
                all_results.update(batch_results)

            return all_results

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

            return new_records, existing_records, duplicate_count

        except Exception as e:
            print(f"❌ Lỗi khi lọc records trùng lặp: {str(e)}")
            return raw_data_list, [], 0

    # === CRAWL DATA METHODS ===
    def crawl_data_by_date_range(self, start_date: str, end_date: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Crawl data từ database theo khoảng thời gian từ bảng product_marketing_summary"""
        if not self.db_connection:
            if not self._connect_db():
                return []

        try:
            cursor = self.db_connection.cursor()

            query = """
            SELECT 
                COALESCE(product_id::text, CONCAT('SP_', SUBSTRING(MD5(RANDOM()::text), 1, 8))) as id_sanpham,
                COALESCE(image, '') as image,
                COALESCE(CAST(published_at AS text), CAST(NOW() AS text)) as date,
                '' as like,
                '' as comment, 
                '' as share,
                COALESCE(link, '') as link_redirect,
                COALESCE(gp_code, 'Website') as platform,
                COALESCE(store, 'unknown') as name_store,
                COALESCE(title, '') as title,
                COALESCE(spend::text, '0') as spend,
                COALESCE(clicks::text, '0') as clicks,
                COALESCE(unique_atc::text, '0') as unique_atc,
                COALESCE(impression::text, '0') as impression,
                COALESCE(unique_clicks::text, '0') as unique_clicks,
                COALESCE(reach::text, '0') as reach,
                COALESCE(quantity::text, '0') as quantity
            FROM ai_craw.product_marketing_summary
            WHERE published_at BETWEEN %s AND %s
              AND image IS NOT NULL 
              AND image != ''
            ORDER BY published_at DESC
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

    # === OPTIMIZED LABELING METHODS ===
    def _create_simplified_prompt(self) -> str:
        """Tạo prompt đơn giản hơn và đáng tin cậy hơn"""
        return """
Analyze this product image and provide labels in valid JSON format.

Choose 1-3 most relevant labels for each category:

CATEGORIES:
1. **image_recipient**: Who receives this (Mom, Dad, Son, Daughter, Wife, Husband)
2. **target_audience**: Who buys this (From Daughter, From Son, From Wife, From Husband)  
3. **usage_purpose**: Purpose (Gift, Home Decor, Personal Use, Keepsake)
4. **occasion**: Occasion (Christmas, Birthday, Mother's Day, Father's Day, Valentine's Day)
5. **niche_theme**: Theme (Mother, Father, Family, Pet, Sports, etc.)
6. **sentiment_tone**: Tone (Sentimental, Humorous, Elegant, Cute)
7. **message_type**: Message (No quote, Symbolic Message, Personal Identity)
8. **product_type**: Type (Mug, T-shirt, Hoodie, Keychain, Plaque, etc.)
9. **design_style**: Style (Modern, Vintage, Minimalist, Colorful)
10. **color_aesthetic**: Colors (Pink, Blue, Black, Colorful, Pastel)
11. **main_subject**: Subject (Text, Animal, Flower, Symbol)
12. **text**: Exact text on product (or "No text" if none)

OUTPUT ONLY VALID JSON:
```json
{
  "image_recipient": ["value"],
  "target_audience": ["value"], 
  "usage_purpose": ["value"],
  "occasion": ["value"],
  "niche_theme": ["value"],
  "sentiment_tone": ["value"],
  "message_type": ["value"],
  "personalization_type": ["Personalized"],
  "product_type": ["value"],
  "placement_display_context": ["Home decor"],
  "design_style": ["value"],
  "color_aesthetic": ["value"],
  "trademark_level": "No TM",
  "main_subject": ["value"],
  "text": ["value"]
}
```
"""

    def _download_image_cached(self, url: str) -> bytes:
        """Download image với caching để tăng tốc"""
        with self.cache_lock:
            if url in self.image_cache:
                return self.image_cache[url]

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            image_bytes = response.content
            
            with self.cache_lock:
                self.image_cache[url] = image_bytes
            
            return image_bytes
        except Exception as e:
            raise Exception(f"Lỗi download ảnh: {str(e)}")

    def _extract_json_robust(self, content: str) -> Dict:
        """Extract JSON từ response một cách robust hơn"""
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Try to find JSON pattern
        json_patterns = [
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Simple nested
            r'\{.*?\}',  # Basic
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        # If still no JSON, try to extract line by line
        lines = content.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            if '{' in line:
                in_json = True
            if in_json:
                json_lines.append(line)
            if '}' in line and in_json:
                break
        
        if json_lines:
            try:
                json_str = '\n'.join(json_lines)
                return json.loads(json_str)
            except:
                pass
        
        # Fallback: create basic structure
        return self._create_fallback_labels()

    def _create_fallback_labels(self) -> Dict:
        """Tạo labels cơ bản khi không parse được JSON"""
        return {
            "image_recipient": ["Unknown"],
            "target_audience": ["Unknown"],
            "usage_purpose": ["Gift"],
            "occasion": ["General"],
            "niche_theme": ["General"],
            "sentiment_tone": ["Neutral"],
            "message_type": ["No quote"],
            "personalization_type": ["Non-personalized"],
            "product_type": ["Unknown"],
            "placement_display_context": ["Home decor"],
            "design_style": ["Modern"],
            "color_aesthetic": ["Colorful"],
            "trademark_level": "No TM",
            "main_subject": ["Unknown"],
            "text": ["No text"]
        }

    def _analyze_with_ollama_optimized(self, image_url: str) -> Dict:
        """Phân tích với Ollama LLaVA - optimized version"""
        try:
            # Download và encode image với cache
            image_bytes = self._download_image_cached(image_url)
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

            prompt = self._create_simplified_prompt()

            # Optimized parameters
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                images=[image_base64],
                options={
                    'temperature': 0.1,
                    'top_p': 0.8,
                    'num_ctx': 2048,  # Reduced context
                    'num_predict': 512  # Limit output length
                }
            )

            content = response['response']
            result = self._extract_json_robust(content)
            return result

        except Exception as e:
            print(f"⚠️  Ollama error: {str(e)}")
            return self._create_fallback_labels()

    def _analyze_with_google_optimized(self, image_url: str) -> Dict:
        """Phân tích với Google Gemini - optimized version"""
        if not self.google_client:
            return self._create_fallback_labels()

        try:
            image_bytes = self._download_image_cached(image_url)
            image = Image.open(BytesIO(image_bytes))
            
            # Resize image for faster processing
            image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            prompt = self._create_simplified_prompt()

            response = self.google_client.generate_content([prompt, image])
            content = response.text
            result = self._extract_json_robust(content)
            return result

        except Exception as e:
            print(f"⚠️  Google error: {str(e)}")
            return self._create_fallback_labels()

    def label_image_optimized(self, image_url: str, provider: ModelProvider = ModelProvider.OLLAMA) -> ProductLabel:
        """Đánh label cho 1 ảnh sản phẩm - optimized version"""
        try:
            if provider == ModelProvider.OLLAMA:
                result = self._analyze_with_ollama_optimized(image_url)
            elif provider == ModelProvider.GOOGLE:
                result = self._analyze_with_google_optimized(image_url)
            else:  # BOTH - but simplified
                result = self._analyze_with_ollama_optimized(image_url)

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
            print(f"⚠️  Labeling error for {image_url}: {str(e)}")
            # Return fallback label
            return ProductLabel(
                image_url=image_url,
                **self._create_fallback_labels()
            )

    # === PARALLEL PROCESSING METHODS ===
    def _process_single_record_thread_safe(self, raw_data: Dict[str, Any], provider: ModelProvider) -> Optional[ProductRecord]:
        """Thread-safe version của process single record"""
        try:
            image_url = raw_data.get('image', '')
            if not image_url:
                return None

            # 1. Label metadata
            label = self.label_image_optimized(image_url, provider)
            metadata = asdict(label)

            # 2. Tạo description markdown
            description = self._create_description(label)

            # 3. Tạo embedding vectors
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
            print(f"⚠️  Lỗi xử lý record {raw_data.get('id_sanpham', 'unknown')}: {str(e)}")
            return None

    def process_batch_records_parallel(self, raw_data_list: List[Dict[str, Any]], 
                                     provider: ModelProvider = ModelProvider.OLLAMA) -> List[ProductRecord]:
        """Xử lý nhiều records song song để tăng tốc"""
        print(f"🚀 Bắt đầu parallel processing {len(raw_data_list)} records với {self.max_workers} workers")
        
        successful_records = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_data = {
                executor.submit(self._process_single_record_thread_safe, raw_data, provider): raw_data 
                for raw_data in raw_data_list
            }
            
            # Collect results
            for i, future in enumerate(concurrent.futures.as_completed(future_to_data)):
                try:
                    record = future.result(timeout=120)  # 2 minute timeout per record
                    if record:
                        successful_records.append(record)
                        print(f"✅ [{i+1}/{len(raw_data_list)}] Processed: {record.id_sanpham}")
                    else:
                        raw_data = future_to_data[future]
                        print(f"❌ [{i+1}/{len(raw_data_list)}] Failed: {raw_data.get('id_sanpham', 'unknown')}")
                        
                except concurrent.futures.TimeoutError:
                    raw_data = future_to_data[future]
                    print(f"⏰ [{i+1}/{len(raw_data_list)}] Timeout: {raw_data.get('id_sanpham', 'unknown')}")
                except Exception as e:
                    raw_data = future_to_data[future]
                    print(f"❌ [{i+1}/{len(raw_data_list)}] Error: {raw_data.get('id_sanpham', 'unknown')} - {str(e)}")

        print(f"✅ Parallel processing hoàn thành: {len(successful_records)}/{len(raw_data_list)} thành công")
        return successful_records

    # === VECTOR GENERATION METHODS ===
    def _generate_vectors(self, text: str, image_url: str = None) -> tuple:
        """Tạo embedding vectors cho text và image sử dụng Jina v4"""
        image_vector, text_vector = self.embedding_service._generate_vectors(
            text=text,
            image_url=image_url
        )
        return image_vector, text_vector

    def _create_description(self, label: ProductLabel) -> str:
        """Tạo description ngắn gọn hơn từ ProductLabel"""
        def format_list(items: List[str]) -> str:
            if not items:
                return "Unknown"
            return ", ".join(items[:3])  # Chỉ lấy 3 items đầu

        description = f"""Product: {format_list(label.product_type)}
Subject: {format_list(label.main_subject)}
For: {format_list(label.image_recipient)}
Occasion: {format_list(label.occasion)}
Style: {format_list(label.design_style)}
Colors: {format_list(label.color_aesthetic)}
Text: {format_list(label.text)}
Theme: {format_list(label.niche_theme)}
Tone: {format_list(label.sentiment_tone)}
"""
        return description

    # === INSERTION METHODS ===
    def insert_batch_records(self, records: List[ProductRecord]) -> List[str]:
        """Insert nhiều ProductRecord vào Milvus cùng lúc"""
        try:
            if not records:
                return []

            # Batch size để tránh memory issues
            batch_size = 50
            all_inserted_ids = []
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                
                # Chuẩn bị data cho batch insert
                ids = [record.id_sanpham for record in batch]
                image_vectors = [record.image_vector for record in batch]
                description_vectors = [record.description_vector for record in batch]
                images = [record.image for record in batch]
                descriptions = [record.description for record in batch]
                metadatas = [record.metadata for record in batch]
                dates = [record.date for record in batch]
                likes = [record.like for record in batch]
                comments = [record.comment for record in batch]
                shares = [record.share for record in batch]
                link_redirects = [record.link_redirect for record in batch]
                platforms = [record.platform for record in batch]
                name_stores = [record.name_store for record in batch]

                data = [
                    ids, image_vectors, description_vectors, images, descriptions, metadatas,
                    dates, likes, comments, shares, link_redirects, platforms, name_stores
                ]

                mr = self.collection.insert(data)
                self.collection.flush()
                all_inserted_ids.extend(ids)
                
                print(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} records")

            return all_inserted_ids

        except Exception as e:
            raise Exception(f"Lỗi batch insert: {str(e)}")

    # === MAIN PIPELINE METHOD ===
    def run_pipeline_optimized(self, start_date: str, end_date: str,
                              limit: int = 1000,
                              provider: ModelProvider = ModelProvider.OLLAMA,
                              parallel_batch_size: int = 20) -> Dict[str, Any]:
        """
        Chạy pipeline tối ưu với parallel processing

        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            limit: Số lượng record tối đa
            provider: Model provider để labeling
            parallel_batch_size: Số record xử lý song song mỗi batch

        Returns:
            Dictionary chứa thống kê kết quả
        """
        print("🚀 BẮT ĐẦU OPTIMIZED PIPELINE")
        print(f"📅 Thời gian: {start_date} → {end_date}")
        print(f"📊 Giới hạn: {limit} records")
        print(f"🤖 Provider: {provider.value}")
        print(f"⚡ Parallel batch size: {parallel_batch_size}")
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

            # Chạy trên CPU cho duplicate check
            if torch.cuda.is_available():
                torch.set_default_device("cpu")

            new_records, existing_records, duplicate_count = self.filter_existing_records(raw_data_list)

            stats['duplicate_count'] = duplicate_count
            stats['skipped_duplicates'] = [record.get('id_sanpham', 'unknown') for record in existing_records]

            if not new_records:
                print("⚠️  Tất cả records đã tồn tại trong Milvus, không có gì để xử lý")
                return stats

            print(f"✅ Sẽ xử lý {len(new_records)} records mới")

            # Chuyển lại sang GPU cho embedding
            if torch.cuda.is_available():
                torch.set_default_device("cuda")

            # STEP 3: Parallel processing trong batches
            print(f"🚀 STEP 3: Parallel processing với batch_size={parallel_batch_size}")
            
            all_processed_records = []
            
            for i in range(0, len(new_records), parallel_batch_size):
                batch = new_records[i:i + parallel_batch_size]
                batch_num = (i // parallel_batch_size) + 1
                total_batches = (len(new_records) + parallel_batch_size - 1) // parallel_batch_size

                print(f"📦 Batch {batch_num}/{total_batches}: Processing {len(batch)} records...")

                # Process batch in parallel
                batch_records = self.process_batch_records_parallel(batch, provider)
                all_processed_records.extend(batch_records)
                
                stats['processed_count'] = len(all_processed_records)
                stats['failed_count'] = len(new_records) - len(all_processed_records)

                print(f"📦 Batch {batch_num} completed: {len(batch_records)}/{len(batch)} successful")
                print(f"📊 Overall progress: {len(all_processed_records)}/{len(new_records)} processed")

            # STEP 4: Batch insert vào Milvus
            if all_processed_records:
                print(f"💾 STEP 4: Batch insert {len(all_processed_records)} records vào Milvus...")
                
                inserted_ids = self.insert_batch_records(all_processed_records)
                stats['inserted_count'] = len(inserted_ids)
                stats['inserted_ids'] = inserted_ids
                
                print(f"✅ Insert thành công {len(inserted_ids)} records")
            else:
                print("⚠️  Không có records nào để insert")

        except Exception as e:
            print(f"❌ Lỗi nghiêm trọng trong pipeline: {str(e)}")

        finally:
            # Tính toán thời gian
            end_time = time.time()
            stats['total_time_seconds'] = round(end_time - start_time, 2)
            stats['end_time'] = datetime.now().isoformat()

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

            # Tính tỉ lệ thành công trên records mới
            new_records_count = len(new_records) if 'new_records' in locals() else max(
                stats['crawled_count'] - stats['duplicate_count'], 1)
            success_rate = stats['inserted_count'] / max(new_records_count, 1) * 100
            print(f"   📈 Tỉ lệ thành công: {success_rate:.1f}%")
            
            # Tính tốc độ xử lý
            if stats['total_time_seconds'] > 0:
                processing_rate = stats['inserted_count'] / stats['total_time_seconds']
                print(f"   🚄 Tốc độ xử lý: {processing_rate:.2f} records/second")

            # Hiển thị collection stats
            try:
                total_entities = self.collection.num_entities
                print(f"   💾 Tổng entities trong Milvus: {total_entities}")
            except:
                pass

            print("=" * 80)
            return stats

    # === UTILITY METHODS ===
    def save_stats_to_json(self, stats: Dict[str, Any], filename: str = None):
        """Lưu thống kê vào file JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_stats_optimized_{timestamp}.json"

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

    def clear_cache(self):
        """Clear image cache để giải phóng memory"""
        with self.cache_lock:
            self.image_cache.clear()
        print("🧹 Đã clear image cache")


def main_optimized():
    """
    Hàm main tối ưu - nhanh hơn với parallel processing
    """
    print("🚀 KHỞI ĐỘNG OPTIMIZED INTEGRATED PRODUCT PIPELINE")
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

    # Model config
    ollama_model = "gpt-oss:20b"
    
    # Parallel processing config
    max_workers = 3  # Số thread đồng thời (adjust based on your system)

    # Milvus config
    milvus_host = "10.10.4.25"
    milvus_port = "19530"

    # ========== THỜI GIAN CRAWL ==========
    start_date = "2023-08-04"  # YYYY-MM-DD
    end_date = "2025-08-14"    # YYYY-MM-DD

    # ========== CÀI ĐẶT PIPELINE ==========
    limit = 3                           # Giới hạn số record
    provider = ModelProvider.OLLAMA         # Chỉ dùng OLLAMA để nhanh hơn
    parallel_batch_size = 3                # Số record xử lý song song mỗi batch

    try:
        # Khởi tạo pipeline
        print("🔧 Khởi tạo optimized pipeline...")
        pipeline = IntegratedProductPipeline(
            db_config=db_config,
            google_api_key=google_api_key,
            ollama_model=ollama_model,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
            max_workers=max_workers
        )

        print("✅ Pipeline khởi tạo thành công!")

        # Chạy optimized pipeline
        print(f"🎯 Bắt đầu crawl data từ {start_date} đến {end_date}")

        stats = pipeline.run_pipeline_optimized(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            provider=provider,
            parallel_batch_size=parallel_batch_size
        )

        # Lưu thống kê
        pipeline.save_stats_to_json(stats)

        # Clear cache để giải phóng memory
        pipeline.clear_cache()

        # Hiển thị kết quả ngắn gọn
        print("\n🎊 KẾT QUẢ CUỐI CÙNG:")
        print(f"✅ Thành công: {stats['inserted_count']}/{stats['crawled_count']} records")
        print(f"🔄 Trùng lặp (bỏ qua): {stats['duplicate_count']} records")
        print(f"⏱️  Thời gian: {stats['total_time_seconds']}s")
        
        if stats['total_time_seconds'] > 0:
            processing_rate = stats['inserted_count'] / stats['total_time_seconds']
            print(f"🚄 Tốc độ: {processing_rate:.2f} records/second")

        if stats['inserted_ids']:
            print(f"📦 Một số ID đã insert:")
            for i, record_id in enumerate(stats['inserted_ids'][:5]):
                print(f"   {i + 1}. {record_id}")
            if len(stats['inserted_ids']) > 5:
                print(f"   ... và {len(stats['inserted_ids']) - 5} records khác")

    except Exception as e:
        print(f"❌ LỖI NGHIÊM TRỌNG: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        try:
            pipeline.close_connections()
        except:
            pass

        print("\n👋 Optimized Pipeline kết thúc!")


if __name__ == "__main__":
    main_optimized()