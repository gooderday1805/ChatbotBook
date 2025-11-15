"""
chatbot_app.py

STREAMLIT CHATBOT - RAG với GROQ + GEMINI
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
import json
import requests
import os
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedders.single_embedder import SingleEmbedder
from retrievers.dense_retriever import DenseRetriever

# Load environment
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RAGChatbot:
    """RAG Chatbot với retrieval + LLM"""

    def __init__(self, strategy_name="bge-m3"):
        self.strategy_name = strategy_name
        self.embedder = None
        self.retriever = None
        self.chunks = None

    def load_rag_system(self):
        """Load embeddings và build retriever"""
        # Load embeddings
        emb_file = f"output/embeddings/{self.strategy_name}_embeddings.npy"
        chunks_file = f"output/embeddings/{self.strategy_name}_chunks.json"

        if not Path(emb_file).exists():
            return False, f"Embeddings not found: {emb_file}"

        embeddings = np.load(emb_file)

        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # Load embedder
        if self.strategy_name == "bge-m3":
            self.embedder = SingleEmbedder(
                model_name="BAAI/bge-m3",
                device="cpu",
                batch_size=32
            )
        else:
            return False, f"Strategy {self.strategy_name} not supported yet"

        # Build retriever
        self.retriever = DenseRetriever(self.embedder)
        self.retriever.fit(self.chunks, embeddings)

        return True, f"Loaded {len(self.chunks)} chunks"

    def retrieve(self, query: str, top_k: int = 3):
        """Retrieve relevant chunks"""
        if self.retriever is None:
            return []

        results = self.retriever.search(query, top_k=top_k)
        return results

    def generate_with_groq(self, query: str, context: str) -> tuple[str, bool]:
        """Generate answer with GROQ"""
        if not GROQ_API_KEY:
            return "❌ GROQ API key not found!", False

        prompt = f"""Bạn là trợ lý giáo viên Toán. Dựa vào NGỮ CẢNH dưới đây, trả lời câu hỏi của học sinh.

NGUYÊN TẮC:
- Chỉ dựa vào NGỮ CẢNH để trả lời
- Nếu không tìm thấy thông tin → nói "Tôi không tìm thấy thông tin này trong sách"
- Giải thích đơn giản, dễ hiểu
- Có thể đưa ví dụ minh họa

NGỮ CẢNH:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 2048
                },
                timeout=30
            )

            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content']
                return answer, True
            else:
                return f"❌ GROQ Error: {response.status_code}", False

        except Exception as e:
            return f"❌ GROQ Exception: {str(e)}", False

    def generate_with_gemini(self, query: str, context: str) -> tuple[str, bool]:
        """Generate answer with GEMINI"""
        if not GEMINI_API_KEY:
            return "❌ GEMINI API key not found!", False

        prompt = f"""Bạn là trợ lý giáo viên Toán. Dựa vào NGỮ CẢNH dưới đây, trả lời câu hỏi của học sinh.

NGUYÊN TẮC:
- Chỉ dựa vào NGỮ CẢNH để trả lời
- Nếu không tìm thấy thông tin → nói "Tôi không tìm thấy thông tin này trong sách"
- Giải thích đơn giản, dễ hiểu
- Có thể đưa ví dụ minh họa

NGỮ CẢNH:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 2048
                    }
                },
                timeout=30
            )

            if response.status_code == 200:
                answer = response.json()['candidates'][0]['content']['parts'][0]['text']
                return answer, True
            else:
                return f"❌ GEMINI Error: {response.status_code}", False

        except Exception as e:
            return f"❌ GEMINI Exception: {str(e)}", False

    def answer(self, query: str, llm_choice: str = "groq", top_k: int = 3):
        """
        Answer query với RAG

        Returns:
            dict with 'answer', 'sources', 'success'
        """
        # 1. Retrieve
        results = self.retrieve(query, top_k=top_k)

        if not results:
            return {
                'answer': "❌ Không tìm thấy thông tin liên quan.",
                'sources': [],
                'success': False
            }

        # 2. Build context
        context_parts = []
        for i, r in enumerate(results, 1):
            page = r.get('page_num', '?')
            text = r['text']
            context_parts.append(f"[Nguồn {i} - Trang {page}]\n{text}\n")

        context = "\n".join(context_parts)

        # 3. Generate with LLM
        if llm_choice == "groq":
            answer, success = self.generate_with_groq(query, context)
        elif llm_choice == "gemini":
            answer, success = self.generate_with_gemini(query, context)
        else:
            answer = "❌ Invalid LLM choice"
            success = False

        return {
            'answer': answer,
            'sources': results,
            'success': success
        }


def main():
    st.set_page_config(
        page_title="Chatbot Toán Học",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 Chatbot Sách Giáo Khoa Toán")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài Đặt")

        # LLM choice
        llm_choice = st.radio(
            "Chọn LLM:",
            ["groq", "gemini"],
            help="GROQ: Nhanh hơn | GEMINI: Chất lượng cao hơn"
        )

        # Top-k
        top_k = st.slider(
            "Số nguồn tham khảo:",
            min_value=1,
            max_value=5,
            value=3,
            help="Số chunks để làm context"
        )

        # Show sources
        show_sources = st.checkbox("Hiển thị nguồn", value=True)

        st.markdown("---")
        st.markdown("### 📊 Thống Kê")

        if 'chatbot' in st.session_state and st.session_state.chatbot.chunks:
            st.metric("Số chunks", len(st.session_state.chatbot.chunks))
            st.metric("Model", "BGE-M3")

    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("🔄 Đang load RAG system..."):
            chatbot = RAGChatbot(strategy_name="bge-m3")
            success, message = chatbot.load_rag_system()

            if success:
                st.session_state.chatbot = chatbot
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
                st.info("💡 Chạy `python embed_gpu.py` trước để tạo embeddings")
                st.stop()

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if exists
            if "sources" in message and show_sources:
                with st.expander("📚 Nguồn tham khảo"):
                    for src in message["sources"]:
                        st.markdown(
                            f"**Rank {src['rank']} - Page {src.get('page_num', '?')} - Score: {src['score']:.3f}**")
                        st.text(src['text'][:200] + "...")
                        st.markdown("---")

    # Chat input
    if prompt := st.chat_input("Hỏi gì về sách Toán?"):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Đang suy nghĩ..."):
                result = st.session_state.chatbot.answer(
                    prompt,
                    llm_choice=llm_choice,
                    top_k=top_k
                )

                st.markdown(result['answer'])

                # Show sources
                if result['sources'] and show_sources:
                    with st.expander("📚 Nguồn tham khảo"):
                        for src in result['sources']:
                            st.markdown(
                                f"**Rank {src['rank']} - Page {src.get('page_num', '?')} - Score: {src['score']:.3f}**")
                            st.text(src['text'][:200] + "...")
                            st.markdown("---")

                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "sources": result['sources']
                })

    # Clear button
    if st.sidebar.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()