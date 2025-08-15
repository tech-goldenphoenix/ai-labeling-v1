import streamlit as st
import pandas as pd
import json
from pymilvus import connections, Collection, utility
import time
from datetime import datetime, timedelta
import hashlib


# ==================== ENHANCED CACHING ====================

# Connection caching với retry logic
@st.cache_resource
def get_milvus_connection():
    """Cache Milvus connection - chỉ tạo 1 lần"""
    try:
        connections.connect(
            alias="default",
            host="10.10.10.140",
            port="19530"
        )
        return True
    except Exception as e:
        st.error(f"❌ Lỗi kết nối Milvus: {e}")
        return False


def connect_to_milvus():
    """Wrapper function cho connection với caching"""
    return get_milvus_connection()


# Collection data caching với TTL dài hơn
@st.cache_data(
    ttl=7200,  # Cache 2 tiếng thay vì 50 phút
    max_entries=3,  # Giới hạn số cache entries
    show_spinner="🔄 Đang tải dữ liệu từ Milvus..."
)
def load_collection_data():
    """Load dữ liệu từ collection với caching tối ưu"""
    try:
        collection_name = "filtered_collection_from_chinh"

        if not utility.has_collection(collection_name):
            st.error(f"❌ Collection '{collection_name}' không tồn tại!")
            return None

        collection = Collection(collection_name)
        collection.load()

        # Query toàn bộ dữ liệu
        results = collection.query(
            expr="",  # Query tất cả
            output_fields=["id_sanpham", "platform", "description", "metadata", "date", "like", "comment", "share",
                           "name_store"],
            limit=16384  # Giới hạn max
        )

        return results
    except Exception as e:
        st.error(f"❌ Lỗi load dữ liệu: {e}")
        return None


# Cached collection status check
@st.cache_data(ttl=1800)  # Cache 30 phút
def check_collection_exists(collection_name):
    """Cache collection existence check"""
    try:
        return utility.has_collection(collection_name)
    except:
        return False


# ==================== OPTIMIZED DATA PARSING ====================

@st.cache_data(
    ttl=3600,  # Cache 1 tiếng
    max_entries=5,
    show_spinner="📊 Đang xử lý metadata..."
)
def parse_metadata_cached(data_hash, data):
    """Parse metadata với caching based on data hash"""
    return parse_metadata_internal(data)


def parse_metadata(data):
    """Parse metadata từ JSON và tạo DataFrame với caching"""
    if not data:
        return pd.DataFrame()

    # Tạo hash của data để cache
    data_str = json.dumps(data, sort_keys=True)
    data_hash = hashlib.md5(data_str.encode()).hexdigest()

    return parse_metadata_cached(data_hash, data)


def parse_metadata_internal(data):
    """Internal parsing function"""
    parsed_data = []

    for item in data:
        try:
            metadata = json.loads(item.get('metadata', '{}')) if isinstance(item.get('metadata'), str) else item.get(
                'metadata', {})

            row = {
                'id_sanpham': item.get('id_sanpham', ''),
                'platform': item.get('platform', ''),
                'description': item.get('description', ''),
                'name_store': item.get('name_store', ''),
                'date': item.get('date', ''),
                'like': int(item.get('like', 0)) if str(item.get('like', 0)).isdigit() else 0,
                'comment': int(item.get('comment', 0)) if str(item.get('comment', 0)).isdigit() else 0,
                'share': int(item.get('share', 0)) if str(item.get('share', 0)).isdigit() else 0,
            }

            # Thêm các trường metadata
            for key, value in metadata.items():
                if isinstance(value, list):
                    row[key] = ', '.join(value)
                else:
                    row[key] = str(value)

            parsed_data.append(row)
        except Exception as e:
            st.warning(f"Lỗi parse metadata cho item {item.get('id_sanpham', 'unknown')}: {e}")
            continue

    return pd.DataFrame(parsed_data)


# ==================== CACHED UTILITY FUNCTIONS ====================

@st.cache_data
def safe_int_convert(value):
    """Safely convert value to integer - cached"""
    try:
        if isinstance(value, (int, float)):
            return int(value)
        elif isinstance(value, str):
            # Remove commas and spaces, then convert
            clean_value = value.replace(',', '').replace(' ', '').strip()
            return int(clean_value) if clean_value.isdigit() else 0
        else:
            return 0
    except:
        return 0


@st.cache_data
def parse_engagement_string(engagement_str):
    """Parse engagement string với caching"""
    if not engagement_str:
        return {"like": 0, "comment": 0, "share": 0}

    try:
        # Handle different possible formats
        engagement_dict = {"like": 0, "comment": 0, "share": 0}

        # Case 1: JSON-like string format
        if engagement_str.strip().startswith('{') and engagement_str.strip().endswith('}'):
            import json
            try:
                parsed = json.loads(engagement_str)
                if isinstance(parsed, dict):
                    return {
                        "like": safe_int_convert(parsed.get("like", 0)),
                        "comment": safe_int_convert(parsed.get("comment", 0)),
                        "share": safe_int_convert(parsed.get("share", 0))
                    }
            except:
                pass

        # Case 2: Dictionary-like string format (like, comment, share)
        if "like" in engagement_str.lower() or "comment" in engagement_str.lower() or "share" in engagement_str.lower():
            # Extract numbers after keywords
            import re

            like_match = re.search(r'like[\'\":\s]*(\d+)', engagement_str, re.IGNORECASE)
            comment_match = re.search(r'comment[\'\":\s]*(\d+)', engagement_str, re.IGNORECASE)
            share_match = re.search(r'share[\'\":\s]*(\d+)', engagement_str, re.IGNORECASE)

            if like_match:
                engagement_dict["like"] = int(like_match.group(1))
            if comment_match:
                engagement_dict["comment"] = int(comment_match.group(1))
            if share_match:
                engagement_dict["share"] = int(share_match.group(1))

            return engagement_dict

        # Case 3: Simple number format (fallback)
        clean_str = engagement_str.replace(',', '').replace(' ', '').strip()
        if clean_str.isdigit():
            total = int(clean_str)
            # Distribute as before for backward compatibility
            return {
                "like": int(total * 0.7),
                "comment": int(total * 0.2),
                "share": int(total * 0.1)
            }

        return engagement_dict

    except Exception as e:
        print(f"Error parsing engagement: {e}")
        return {"like": 0, "comment": 0, "share": 0}


# ==================== BATCH PROCESSING WITH CACHING ====================

@st.cache_data(
    ttl=1800,
    max_entries=10,
    show_spinner="⚡ Đang xử lý batch data..."
)
def process_batch_data(data_batch_hash, data_batch):
    """Process batch data với caching"""
    processed_items = []

    for item in data_batch:
        try:
            processed_item = {
                'id': item.get('id_sanpham', ''),
                'platform': item.get('platform', ''),
                'engagement_score': calculate_engagement_score(item),
                'processed_at': datetime.now().isoformat()
            }
            processed_items.append(processed_item)
        except Exception as e:
            continue

    return processed_items


def calculate_engagement_score(item):
    """Calculate engagement score for item"""
    try:
        likes = safe_int_convert(item.get('like', 0))
        comments = safe_int_convert(item.get('comment', 0))
        shares = safe_int_convert(item.get('share', 0))

        return likes + comments * 5 + shares * 10
    except:
        return 0


# ==================== CACHE MANAGEMENT ====================

def clear_data_cache():
    """Clear all data-related caches"""
    st.cache_data.clear()
    st.success("✅ Đã xóa cache dữ liệu!")


def get_cache_stats():
    """Get cache statistics"""
    return {
        'cache_hits': 'Not available in current Streamlit version',
        'cache_size': 'Not available in current Streamlit version',
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


# ==================== HEALTH CHECK ====================

@st.cache_data(ttl=300)  # Cache 5 phút
def health_check():
    """Health check cho data processor"""
    try:
        # Test connection
        connection_ok = connect_to_milvus()

        # Test collection
        collection_ok = check_collection_exists("filtered_collection_from_chinh")

        return {
            'status': 'healthy' if connection_ok and collection_ok else 'degraded',
            'connection': connection_ok,
            'collection': collection_ok,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }