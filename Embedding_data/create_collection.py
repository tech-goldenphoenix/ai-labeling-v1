from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import time


# Kết nối tới Milvus
def connect_milvus():
    connections.connect(
        alias="default",
        host="localhost",  # Hoặc sử dụng IP của bạn: "0.10.10.64"
        port="19530"
    )
    print("✅ Kết nối Milvus thành công!")


def create_collection_schema():
    fields = [
        FieldSchema(name="id_sanpham", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="description_vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="image", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="metadata", dtype=DataType.JSON),
        FieldSchema(name="date", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="like", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="comment", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="share", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="link_redirect", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="platform", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="name_store", dtype=DataType.VARCHAR, max_length=200)
    ]

    schema = CollectionSchema(
        fields=fields,
        description="Collection chứa thông tin sản phẩm với embedding"
    )
    return schema


def create_collection():
    collection_name = "product_collection"

    # Kiểm tra collection đã tồn tại chưa
    if utility.has_collection(collection_name):
        print(f"⚠️  Collection '{collection_name}' đã tồn tại!")
        return Collection(collection_name)

    # Tạo collection
    schema = create_collection_schema()
    collection = Collection(collection_name, schema)

    print(f"✅ Tạo collection '{collection_name}' thành công!")
    return collection


def create_indexes(collection):
    """Tạo index cho vector fields để tăng tốc độ tìm kiếm"""

    # Index cho image_vector
    image_index_params = {
        "metric_type": "COSINE",  # Inner Product (có thể dùng L2, COSINE)
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1536}
    }

    # Index cho description_vector
    desc_index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1536}
    }

    collection.create_index(
        field_name="image_vector",
        index_params=image_index_params,
        index_name="image_vector_index"
    )

    collection.create_index(
        field_name="description_vector",
        index_params=desc_index_params,
        index_name="description_vector_index"
    )

    print("✅ Tạo indexes thành công!")


def main():
    try:
        # Kết nối
        connect_milvus()

        # Tạo collection
        collection = create_collection()

        # Tạo indexes
        create_indexes(collection)

        # Load collection vào memory
        collection.load()
        print("✅ Load collection vào memory thành công!")

        # Hiển thị thông tin collection
        print("\n📊 Thông tin Collection:")
        print(f"Tên: {collection.name}")
        print(f"Số lượng entities: {collection.num_entities}")
        print(f"Schema: {collection.schema}")

        # Liệt kê tất cả collections
        print(f"\n📋 Danh sách Collections: {utility.list_collections()}")

    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    main()