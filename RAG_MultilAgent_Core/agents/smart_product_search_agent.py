"""
Smart product search agent for Enhanced RnD Assistant với metadata analysis
"""
import re
from typing import Dict, Any, List

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from tools.search_tools import (
    search_by_description_tool,
    search_by_image_tool,
    search_multimodal_tool
)
from config.settings import Config


class SmartProductSearchAgent(BaseAgent):
    """Agent for smart product search with image support and metadata analysis"""

    def __init__(self):
        super().__init__(temperature=0.2)
        self.vision_llm = ChatOpenAI(
            model=Config.VISION_MODEL,
            temperature=0.2,
            api_key=""
        )
        # Khởi tạo metadata mappings
        self.metadata_mappings = self._init_metadata_mappings()

    def _init_metadata_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Khởi tạo mapping từ keywords đến metadata fields"""
        return {
            "main_subject": {
                "Family": ["family", "gia đình", "relatives", "họ hàng"],
                "Police": ["police", "cảnh sát", "officer", "công an"],
                "Halloween": ["halloween", "ma quỷ", "scary", "kinh dị", "pumpkin", "bí ngô"],
                "Christmas": ["christmas", "noel", "giáng sinh", "santa", "xmas"],
                "Mom": ["mom", "mother", "mẹ", "mama", "mommy"],
                "Dad": ["dad", "father", "bố", "papa", "daddy"],
                "Teacher": ["teacher", "giáo viên", "educator", "thầy", "cô"],
                "Doctor": ["doctor", "bác sĩ", "nurse", "y tá", "medical"],
                "Love": ["love", "yêu", "tình yêu", "romantic", "lãng mạn"],
                "Pet": ["pet", "thú cưng", "dog", "chó", "cat", "mèo"],
                "Sports": ["sports", "thể thao", "football", "basketball", "soccer"],
                "Music": ["music", "âm nhạc", "guitar", "piano", "song"]
            },
            "product_type": {
                "Desk Plaque": ["plaque", "bảng", "desk", "bàn làm việc"],
                "Mug": ["mug", "cốc", "cup", "ly"],
                "T-Shirt": ["tshirt", "shirt", "áo", "clothing"],
                "Canvas": ["canvas", "tranh", "painting", "wall art"],
                "Pillow": ["pillow", "gối", "cushion", "throw pillow"],
                "Keychain": ["keychain", "móc khóa", "key ring"],
                "Tumbler": ["tumbler", "bottle", "chai", "water bottle"],
                "Frame": ["frame", "khung", "photo frame", "khung ảnh"]
            },
            "recipient": {
                "Mom": ["mom", "mother", "mẹ", "mama", "mommy"],
                "Dad": ["dad", "father", "bố", "papa", "daddy"],
                "Grandma": ["grandma", "grandmother", "bà", "ngoại", "bà ngoại"],
                "Grandpa": ["grandpa", "grandfather", "ông", "nội", "ông nội"],
                "Wife": ["wife", "vợ", "spouse", "partner"],
                "Husband": ["husband", "chồng", "spouse", "partner"],
                "Daughter": ["daughter", "con gái", "girl"],
                "Son": ["son", "con trai", "boy"],
                "Sister": ["sister", "chị", "em gái"],
                "Brother": ["brother", "anh", "em trai"],
                "Teacher": ["teacher", "giáo viên", "thầy", "cô"],
                "Friend": ["friend", "bạn", "buddy", "pal"]
            },
            "buyer": {
                "From Daughter": ["from daughter", "từ con gái", "daughter to", "con gái tặng"],
                "From Son": ["from son", "từ con trai", "son to", "con trai tặng"],
                "From Wife": ["from wife", "từ vợ", "wife to", "vợ tặng"],
                "From Husband": ["from husband", "từ chồng", "husband to", "chồng tặng"],
                "From Kids": ["from kids", "từ con", "children to", "con tặng"],
                "From Family": ["from family", "từ gia đình", "family gift"]
            },
            "occasion": {
                "Mother's Day": ["mother's day", "ngày của mẹ", "mom's day"],
                "Father's Day": ["father's day", "ngày của bố", "dad's day"],
                "Halloween": ["halloween", "lễ hội ma quỷ"],
                "Christmas": ["christmas", "noel", "giáng sinh", "xmas"],
                "Valentine": ["valentine", "ngày valentine", "tình nhân"],
                "Birthday": ["birthday", "sinh nhật", "bday"],
                "Anniversary": ["anniversary", "kỷ niệm", "ngày cưới"],
                "Graduation": ["graduation", "tốt nghiệp", "graduate"],
                "Retirement": ["retirement", "nghỉ hưu", "retire"],
                "New Year": ["new year", "năm mới", "tết"]
            },
            "purpose": {
                "Gift": ["gift", "quà", "present", "tặng"],
                "Home Decor": ["decor", "trang trí", "decoration", "home"],
                "Office Decor": ["office", "văn phòng", "workplace", "desk"],
                "Personal Use": ["personal", "cá nhân", "for myself", "own use"]
            },
            "theme": {
                "Family": ["family", "gia đình"],
                "Police": ["police", "cảnh sát"],
                "Halloween": ["halloween", "scary", "spooky"],
                "Christmas": ["christmas", "holiday", "festive"],
                "Love": ["love", "romantic", "heart"],
                "Funny": ["funny", "humor", "joke", "hài hước"],
                "Inspirational": ["inspirational", "motivational", "inspiring"]
            },
            "emotion_tone": {
                "Sentimental": ["sentimental", "cảm động", "touching", "heartfelt"],
                "Funny": ["funny", "hài hước", "humorous", "joke"],
                "Inspirational": ["inspirational", "cảm hứng", "motivational"],
                "Romantic": ["romantic", "lãng mạn", "love", "sweet"],
                "Serious": ["serious", "nghiêm túc", "formal", "professional"]
            },
            "personalization": {
                "Personalized Name": ["name", "tên", "personalized", "custom name"],
                "Custom Text": ["custom text", "văn bản tùy chỉnh", "your text"],
                "Photo": ["photo", "ảnh", "picture", "hình"],
                "No Personalization": ["no personalization", "không cá nhân hóa"]
            },
            "design_style": {
                "Elegant": ["elegant", "thanh lịch", "classy", "sophisticated"],
                "Cute": ["cute", "dễ thương", "adorable", "sweet"],
                "Modern": ["modern", "hiện đại", "contemporary"],
                "Vintage": ["vintage", "cổ điển", "retro", "classic"],
                "Minimalist": ["minimalist", "tối giản", "simple", "clean"]
            },
            "color_aesthetic": {
                "Pastel": ["pastel", "màu nhạt", "soft colors"],
                "Bright": ["bright", "màu sáng", "vivid", "colorful"],
                "Dark": ["dark", "màu tối", "black", "gothic"],
                "Natural": ["natural", "tự nhiên", "earth tones"],
                "Monochrome": ["monochrome", "đen trắng", "black and white"]
            }
        }

    def _analyze_text_metadata(self, text: str) -> Dict[str, List[str]]:
        """Phân tích text để trích xuất metadata"""
        text_lower = text.lower().strip()
        detected_metadata = {}
        
        # Mapping từ field names sang tên hiển thị
        field_display_names = {
            "main_subject": "Chủ Thể Chính",
            "product_type": "Loại Sản Phẩm", 
            "recipient": "Người Nhận",
            "buyer": "Người Mua",
            "occasion": "Dịp Sử Dụng",
            "purpose": "Mục Đích Sử Dụng",
            "theme": "Chủ Đề/Ngách",
            "emotion_tone": "Cảm Xúc/Tông Điệu",
            "personalization": "Cá Nhân Hóa",
            "design_style": "Phong Cách Thiết Kế",
            "color_aesthetic": "Thẩm Mỹ Màu Sắc"
        }

        # Duyệt qua từng category
        for category, items in self.metadata_mappings.items():
            detected_values = []
            
            # Duyệt qua từng item trong category
            for item_name, keywords in items.items():
                # Kiểm tra xem có keyword nào match không
                for keyword in keywords:
                    if keyword in text_lower:
                        detected_values.append(item_name)
                        break  # Thoát khỏi vòng lặp keywords khi đã tìm thấy match
            
            # Chỉ thêm vào kết quả nếu có phát hiện
            if detected_values:
                display_name = field_display_names.get(category, category)
                detected_metadata[display_name] = list(set(detected_values))  # Remove duplicates

        return detected_metadata

    def _format_metadata_description(self, metadata: Dict[str, List[str]]) -> str:
        """Format metadata thành description dạng cấu trúc"""
        if not metadata:
            return ""
        
        description_parts = []
        description_parts.append("# Thông Tin Sản Phẩm Được Phân Tích")
        
        for field_name, values in metadata.items():
            values_str = ", ".join(values)
            description_parts.append(f"- **{field_name}**: {values_str}")
        
        return "\n".join(description_parts)

    def _is_image_url(self, text: str) -> bool:
        """Kiểm tra xem text có phải là URL hình ảnh không"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
        text_lower = text.lower().strip()

        # Kiểm tra URL pattern
        url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
        if re.search(url_pattern, text_lower):
            # Kiểm tra có extension hình ảnh
            return any(ext in text_lower for ext in image_extensions)

        return False

    def _is_base64_image(self, text: str) -> bool:
        """Kiểm tra xem text có phải là base64 image không"""
        text = text.strip()
        # Kiểm tra data URL format
        if text.startswith('data:image/'):
            return True
        # Kiểm tra base64 pattern (chuỗi dài, chỉ chứa A-Z, a-z, 0-9, +, /, =)
        if len(text) > 100 and re.match(r'^[A-Za-z0-9+/=]+$', text):
            return True
        return False

    async def determine_search_type(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Xác định loại tìm kiếm dựa trên input và phân tích metadata"""
        query = state["query"].lower()
        original_query = state["query"]
        has_image = state.get("input_image") is not None

        # Phân tích metadata từ text query
        detected_metadata = self._analyze_text_metadata(original_query)
        if detected_metadata:
            state["detected_metadata"] = detected_metadata
            state["metadata_description"] = self._format_metadata_description(detected_metadata)
            # Cập nhật query với metadata information
            metadata_context = self._format_metadata_description(detected_metadata)
            state["enriched_query"] = f"{original_query}\n\n{metadata_context}"
        else:
            state["enriched_query"] = original_query

        # Kiểm tra xem query có chứa URL hình ảnh hoặc base64 không
        has_image_in_query = self._is_image_url(original_query) or self._is_base64_image(original_query)

        # Logic xác định search type
        if has_image and any(keyword in query for keyword in ["tương tự", "giống", "similar", "tìm sản phẩm tương tự"]):
            search_type = "image_to_image"

        elif has_image_in_query and any(keyword in query for keyword in ["tương tự", "giống", "similar", "tìm sản phẩm tương tự"]):
            search_type = "url_to_image"
            # Extract image từ query và lưu vào state
            if self._is_image_url(original_query):
                # Tách URL từ query
                url_match = re.search(r'https?://[^\s<>"]+', original_query)
                if url_match:
                    state["image_url"] = url_match.group()
                    # Loại bỏ URL khỏi query để lấy phần mô tả
                    remaining_text = original_query.replace(state["image_url"], "").strip()
                    state["query"] = remaining_text if remaining_text else "tìm sản phẩm tương tự"

        elif has_image and any(keyword in query for keyword in ["mô tả", "describe", "phân tích", "này là gì"]):
            search_type = "image_to_text"

        elif has_image_in_query and any(keyword in query for keyword in ["mô tả", "describe", "phân tích", "này là gì"]):
            search_type = "url_to_text"
            if self._is_image_url(original_query):
                url_match = re.search(r'https?://[^\s<>"]+', original_query)
                if url_match:
                    state["image_url"] = url_match.group()
                    remaining_text = original_query.replace(state["image_url"], "").strip()
                    state["query"] = remaining_text if remaining_text else "mô tả sản phẩm"

        elif any(keyword in query for keyword in ["tìm hình", "show image", "hình ảnh của"]):
            search_type = "text_to_image"

        # NEW: Multimodal search detection
        elif has_image and len(query.strip()) > 5:  # Có cả image và text có ý nghĩa
            search_type = "multimodal_search"

        elif has_image_in_query and len(original_query.replace(self._extract_url(original_query), "").strip()) > 5:
            search_type = "multimodal_url_search"
            if self._is_image_url(original_query):
                url_match = re.search(r'https?://[^\s<>"]+', original_query)
                if url_match:
                    state["image_url"] = url_match.group()
                    remaining_text = original_query.replace(state["image_url"], "").strip()
                    state["query"] = remaining_text

        elif has_image_in_query:
            # Nếu có hình ảnh trong query mà không có keyword đặc biệt, mặc định tìm sản phẩm tương tự
            search_type = "url_to_image"
            if self._is_image_url(original_query):
                url_match = re.search(r'https?://[^\s<>"]+', original_query)
                if url_match:
                    state["image_url"] = url_match.group()
                    remaining_text = original_query.replace(state["image_url"], "").strip()
                    if remaining_text:
                        state["query"] = remaining_text
                    else:
                        state["query"] = "tìm sản phẩm tương tự"
        else:
            search_type = "text_to_text"

        state["search_type"] = search_type
        
        # Log metadata detection
        metadata_info = ""
        if detected_metadata:
            metadata_info = f" | Phát hiện metadata: {list(detected_metadata.keys())}"
        
        state["messages"].append(AIMessage(content=f"Xác định loại tìm kiếm: {search_type}{metadata_info}"))
        return state

    def _extract_url(self, text: str) -> str:
        """Extract URL from text"""
        url_match = re.search(r'https?://[^\s<>"]+', text)
        return url_match.group() if url_match else ""

    async def execute_smart_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Thực hiện tìm kiếm thông minh với metadata enrichment"""
        search_type = state["search_type"]
        # Sử dụng enriched_query thay vì query thông thường để tận dụng metadata
        query = state.get("enriched_query", state["query"])

        if search_type == "text_to_image":
            results = search_by_description_tool.invoke({"description": query})
            state["search_results"] = results

        elif search_type == "image_to_image":
            if state.get("input_image"):
                results = search_by_image_tool.invoke({"image_base64": state["input_image"]})
                state["search_results"] = results
            else:
                state["search_results"] = [{"error": "Không có hình ảnh input"}]

        elif search_type == "url_to_image":
            if state.get("image_url"):
                # Tải hình ảnh từ URL và convert thành base64, sau đó search
                try:
                    import requests
                    import base64

                    response = requests.get(state["image_url"])
                    if response.status_code == 200:
                        image_base64 = base64.b64encode(response.content).decode('utf-8')
                        results = search_by_image_tool.invoke({"image_base64": image_base64})
                        state["search_results"] = results
                    else:
                        state["search_results"] = [{"error": f"Không thể tải hình ảnh từ URL: {response.status_code}"}]
                except Exception as e:
                    state["search_results"] = [{"error": f"Lỗi khi xử lý URL hình ảnh: {str(e)}"}]
            else:
                state["search_results"] = [{"error": "Không có URL hình ảnh"}]

        elif search_type == "image_to_text":
            if state.get("input_image"):
                description = await self._image_to_text_description(state["input_image"])
                state["search_description"] = description
                # Also search for similar products with metadata context
                enhanced_description = f"{description}\n\n{query}" if state.get("metadata_description") else description
                results = search_by_description_tool.invoke({"description": enhanced_description})
                state["search_results"] = results
            else:
                state["search_description"] = "Không có hình ảnh để mô tả"
                state["search_results"] = []

        elif search_type == "url_to_text":
            if state.get("image_url"):
                try:
                    import requests
                    import base64

                    response = requests.get(state["image_url"])
                    if response.status_code == 200:
                        image_base64 = base64.b64encode(response.content).decode('utf-8')
                        description = await self._image_to_text_description(image_base64)
                        state["search_description"] = description
                        enhanced_description = f"{description}\n\n{query}" if state.get("metadata_description") else description
                        results = search_by_description_tool.invoke({"description": enhanced_description})
                        state["search_results"] = results
                    else:
                        state["search_description"] = f"Không thể tải hình ảnh: {response.status_code}"
                        state["search_results"] = []
                except Exception as e:
                    state["search_description"] = f"Lỗi xử lý hình ảnh: {str(e)}"
                    state["search_results"] = []
            else:
                state["search_description"] = "Không có URL hình ảnh"
                state["search_results"] = []

        # NEW: Multimodal search cases with metadata
        elif search_type == "multimodal_search":
            if state.get("input_image"):
                # Sử dụng original query cho multimodal, metadata được thêm vào context
                search_text = state["query"]
                if state.get("metadata_description"):
                    search_text = f"{state['query']}\n\nContext: {state['metadata_description']}"
                
                results = search_multimodal_tool.invoke({
                    "text": search_text,
                    "image_base64": state["input_image"],
                    "top_k": 12
                })
                state["search_results"] = results
            else:
                state["search_results"] = [{"error": "Không có hình ảnh input cho multimodal search"}]

        elif search_type == "multimodal_url_search":
            if state.get("image_url"):
                try:
                    import requests
                    import base64

                    response = requests.get(state["image_url"])
                    if response.status_code == 200:
                        image_base64 = base64.b64encode(response.content).decode('utf-8')
                        search_text = state["query"]
                        if state.get("metadata_description"):
                            search_text = f"{state['query']}\n\nContext: {state['metadata_description']}"
                        
                        results = search_multimodal_tool.invoke({
                            "text": search_text,
                            "image_base64": image_base64,
                            "top_k": 12
                        })
                        state["search_results"] = results
                    else:
                        state["search_results"] = [{"error": f"Không thể tải hình ảnh từ URL: {response.status_code}"}]
                except Exception as e:
                    state["search_results"] = [{"error": f"Lỗi khi xử lý URL trong multimodal search: {str(e)}"}]
            else:
                state["search_results"] = [{"error": "Không có URL hình ảnh cho multimodal search"}]

        else:  # text_to_text with metadata enhancement
            results = search_by_description_tool.invoke({"description": query})
            state["search_results"] = results

        # Log kết quả với thông tin metadata
        metadata_info = ""
        if state.get("detected_metadata"):
            metadata_count = len(state["detected_metadata"])
            metadata_info = f" (với {metadata_count} metadata fields)"

        state["messages"].append(
            AIMessage(content=f"Hoàn thành {search_type} search với {len(state.get('search_results', []))} kết quả{metadata_info}"))
        return state

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process smart search - determine type, analyze metadata and execute"""
        state = await self.determine_search_type(state)
        return await self.execute_smart_search(state)