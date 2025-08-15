import psycopg2
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json
import csv


class DataCrawler:
    """Module để crawl data từ PostgreSQL và format theo structure mong muốn"""

    def __init__(self, db_config: Dict[str, str]):
        """
        Khởi tạo DataCrawler với config database

        Args:
            db_config: Dictionary chứa thông tin kết nối DB
                      {'host': 'localhost', 'database': 'your_db', 'user': 'user', 'password': 'pass'}
        """
        self.db_config = db_config
        self.connection = None

    def connect_db(self) -> bool:
        """Kết nối đến PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            logging.info("Kết nối database thành công")
            return True
        except Exception as e:
            logging.error(f"Lỗi kết nối database: {e}")
            return False

    def disconnect_db(self):
        """Đóng kết nối database"""
        if self.connection:
            self.connection.close()
            logging.info("Đã đóng kết nối database")

    def crawl_random_data(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Crawl ngẫu nhiên data từ bảng toidispy_full và format theo structure mong muốn

        Args:
            limit: Số lượng record ngẫu nhiên cần lấy (default: 200)

        Returns:
            List dictionary chứa data theo format mong muốn
        """
        if not self.connection:
            if not self.connect_db():
                return []

        try:
            cursor = self.connection.cursor()

            # Query để lấy data ngẫu nhiên và format theo structure mong muốn
            # Sửa tên bảng và column name dựa trên schema thực tế
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
            ORDER BY RANDOM()
            LIMIT %s
            """

            cursor.execute(query, (limit,))
            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                results.append(row_dict)

            cursor.close()
            logging.info(f"Đã crawl {len(results)} records ngẫu nhiên từ database")
            return results

        except Exception as e:
            logging.error(f"Lỗi khi crawl data: {e}")
            return []

    def crawl_data_from_db(self, limit: int = 100, conditions: str = "") -> List[Dict[str, Any]]:
        """
        Crawl data từ bảng toidispy_full và format theo structure mong muốn

        Args:
            limit: Số lượng record tối đa cần lấy
            conditions: Điều kiện WHERE thêm vào query (optional)

        Returns:
            List dictionary chứa data theo format mong muốn
        """
        if not self.connection:
            if not self.connect_db():
                return []

        try:
            cursor = self.connection.cursor()

            # Query được sửa để tránh lỗi với reserved keywords và sửa column name
            base_query = """
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
            """

            # Thêm điều kiện WHERE nếu có
            if conditions:
                base_query += f" WHERE {conditions}"

            base_query += " ORDER BY created_at_std DESC"

            # Thêm LIMIT nếu có
            if limit > 0:
                base_query += f" LIMIT {limit}"

            cursor.execute(base_query)
            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                row_dict = dict(zip(columns, row))
                results.append(row_dict)

            cursor.close()
            logging.info(f"Đã crawl {len(results)} records từ database")
            return results

        except Exception as e:
            logging.error(f"Lỗi khi crawl data: {e}")
            return []

    def get_product_data(self, product_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lấy data sản phẩm với filter theo ID (optional)

        Args:
            product_id: ID sản phẩm cần lấy (optional)
            limit: Số lượng record tối đa

        Returns:
            List dictionary chứa product data
        """
        conditions = ""
        if product_id:
            conditions = f"_id = '{product_id}'"

        return self.crawl_data_from_db(limit=limit, conditions=conditions)

    def get_data_by_platform(self, platform: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lấy data theo platform cụ thể

        Args:
            platform: Tên platform (Facebook, Instagram, etc.)
            limit: Số lượng record tối đa

        Returns:
            List dictionary chứa data theo platform
        """
        conditions = f"platform = '{platform}'"
        return self.crawl_data_from_db(limit=limit, conditions=conditions)

    def get_data_by_date_range(self, start_date: str, end_date: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Lấy data theo khoảng thời gian

        Args:
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            limit: Số lượng record tối đa

        Returns:
            List dictionary chứa data trong khoảng thời gian
        """
        conditions = f"created_at_std BETWEEN '{start_date}' AND '{end_date}'"
        return self.crawl_data_from_db(limit=limit, conditions=conditions)

    def get_popular_posts(self, min_likes: int = 100, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Lấy các post phổ biến (nhiều like)

        Args:
            min_likes: Số like tối thiểu
            limit: Số lượng record tối đa

        Returns:
            List dictionary chứa popular posts
        """
        conditions = f'"like" >= {min_likes}'  # Thêm quotes cho reserved keyword
        return self.crawl_data_from_db(limit=limit, conditions=conditions)

    def save_to_json(self, data: List[Dict[str, Any]], filename: str = "crawled_data.json"):
        """
        Lưu data vào file JSON

        Args:
            data: List data cần lưu
            filename: Tên file
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"Đã lưu {len(data)} records vào {filename}")
        except Exception as e:
            logging.error(f"Lỗi khi lưu file: {e}")

    def convert_to_fake_data_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert data sang format fake_data structure mong muốn

        Args:
            data: List data từ database

        Returns:
            List data theo format fake_data
        """
        fake_data_list = []

        for item in data:
            fake_data = {
                "id_sanpham": item.get("id_sanpham", ""),
                "image": item.get("image", ""),
                "date": item.get("date", ""),
                "like": item.get("like", "0"),
                "comment": item.get("comment", "0"),
                "share": item.get("share", "0"),
                "link_redirect": item.get("link_redirect", ""),
                "platform": item.get("platform", ""),
                "name_store": item.get("name_store", "")
            }
            fake_data_list.append(fake_data)

        return fake_data_list


# Sử dụng module
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Config database connection
    db_config = {
        'host': '45.79.189.110',
        'database': 'ai_db',
        'user': 'ai_engineer',
        'password': 'StrongPassword123',
        'port': 5432
    }

    # Khởi tạo crawler
    crawler = DataCrawler(db_config)

    try:
        # Lấy 200 mẫu ngẫu nhiên
        print("=== Lấy 200 mẫu ngẫu nhiên ===")
        random_data = crawler.crawl_random_data(limit=200)

        print(f"Đã lấy được {len(random_data)} records ngẫu nhiên")

        if random_data:
            # Convert sang fake_data format
            fake_data_list = crawler.convert_to_fake_data_format(random_data)

            # Hiển thị 2 mẫu đầu tiên để kiểm tra
            for i, item in enumerate(fake_data_list[:2]):
                print(f"\nRecord {i + 1} (fake_data format):")
                for key, value in item.items():
                    print(f"  {key}: {value}")

            # Lưu vào file JSON
            crawler.save_to_json(fake_data_list, "fake_data_200_samples.json")

            # In ra một mẫu fake_data structure để kiểm tra
            print("\n=== Cấu trúc fake_data mẫu ===")
            sample_fake_data = fake_data_list[0] if fake_data_list else {}
            print(json.dumps(sample_fake_data, ensure_ascii=False, indent=2))

            print(f"\n=== Thống kê ===")
            print(f"Tổng số records: {len(fake_data_list)}")
            print(f"Đã lưu vào file: fake_data_200_samples.json")

    except Exception as e:
        logging.error(f"Lỗi trong quá trình xử lý: {e}")

    finally:
        # Đóng kết nối
        crawler.disconnect_db()