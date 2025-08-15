"""
Chatbot interface for Enhanced RnD Assistant
"""
import asyncio
import base64
from typing import Optional

from workflow.rag_multi_agent_workflow import RAGMultiAgentWorkflow


class RnDChatbot:
    def __init__(self):
        self.workflow = RAGMultiAgentWorkflow()
        print("🤖 Enhanced RnD Assistant với Single Collection đã sẵn sàng!")

    async def chat(self, user_input: str, image_base64: Optional[str] = None) -> str:
        """Main chat interface with image support"""
        if not user_input.strip():
            return "Vui lòng nhập câu hỏi của bạn."

        print(f"\n🔍 Đang xử lý: {user_input}")
        if image_base64:
            print("🖼️ Có hình ảnh đính kèm")
        print("⏳ Vui lòng chờ...")

        response = await self.workflow.process_query(user_input, image_base64)
        return response

    def run_interactive(self):
        """Run interactive chat session"""
        print("\n" + "=" * 80)
        print("🎯 ENHANCED RnD ASSISTANT - Single Collection RAG với Image URLs")
        print("=" * 80)
        print("Chức năng hỗ trợ:")
        print("1. 📊 Benchmark Analysis - Phân tích đối thủ cạnh tranh")
        print("2. 🕳️ Market Gap Discovery - Tìm khoảng trống thị trường")
        print("3. ✅ Idea Verification - Xác minh ý tưởng kinh doanh")
        print("4. 📈 Audience Volume Estimation - Ước tính quy mô khách hàng")
        print("5. 🔍 Smart Search với Image URLs:")
        print("   - Text → Image: 'Tìm hình ảnh keychain Star Wars'")
        print("   - Image → Image: 'Tìm sản phẩm tương tự' (attach image)")
        print("   - Image → Text: 'Mô tả hình này' (attach image)")
        print("\n🖼️ **NEW: Tất cả kết quả đều bao gồm Image URLs để tham khảo visual**")
        print("\nLệnh đặc biệt:")
        print("- 'image:[path]' để load hình ảnh từ file")
        print("- 'quit' để thoát")
        print("=" * 80)

        while True:
            try:
                user_input = input("\n🤔 Câu hỏi của bạn: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Cảm ơn bạn đã sử dụng Enhanced RnD Assistant!")
                    break

                if not user_input:
                    continue

                # Handle image input
                image_base64 = None
                if user_input.startswith('image:'):
                    image_path = user_input.split(':', 1)[1].strip()
                    try:
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                        user_input = input("🖼️ Đã load hình. Câu hỏi về hình này: ").strip()
                    except Exception as e:
                        print(f"❌ Lỗi load hình: {e}")
                        continue

                # Process query
                response = asyncio.run(self.chat(user_input, image_base64))
                print(f"\n🤖 **Enhanced RnD Assistant:**\n{response}")

            except KeyboardInterrupt:
                print("\n👋 Cảm ơn bạn đã sử dụng Enhanced RnD Assistant!")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {e}")



# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    # You can run the interactive chatbot
    chatbot = RnDChatbot()
    chatbot.run_interactive()