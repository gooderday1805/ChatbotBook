# """
# app_chatbot_modular.py
#
# MODULAR RAG CHATBOT WITH PERSISTENT STORAGE
# - Upload PDF once → Save retriever
# - Next times → Load instantly!
# """
#
# import streamlit as st
# import sys
# from pathlib import Path
# import numpy as np
# import json
# import requests
# import os
# from dotenv import load_dotenv
# import time
# import torch
#
# sys.path.insert(0, str(Path(__file__).resolve().parent))
#
# try:
#     from embedders.single_embedder import SingleEmbedder
#     from embedders.vietnamese_embedder import VietnameseEmbedder
#     from chunkers.recursive_chunker import RecursiveChunker
#     from chunkers.page_aware_chunker import PageAwareChunker
#     from chunkers.semantic_chunker import SemanticChunker
#     from retrievers.dense_retriever import DenseRetriever
# except ImportError as e:
#     st.error(f"❌ Import error: {e}")
#     st.stop()
#
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#
#
# # ==================== PIPELINE CONFIGS ====================
# PIPELINE_CONFIGS = {
#     "fast": {
#         "name": "⚡ Fast Pipeline",
#         "description": "Nhanh nhất - Recursive + MiniLM",
#         "chunking": "recursive",
#         "embedding": "minilm",
#         "chunker_class": RecursiveChunker,
#         "chunker_params": {"chunk_size": 512, "overlap": 50},
#         "retriever_path": "output/retrievers/fast_retriever",
#         "color": "🟢"
#     },
#     "balanced": {
#         "name": "⚖️ Balanced Pipeline",
#         "description": "Cân bằng - Page-Aware + GTE-Viet",
#         "chunking": "page_aware",
#         "embedding": "gte-viet",
#         "chunker_class": PageAwareChunker,
#         "chunker_params": {"max_chars": 2000},
#         "retriever_path": "output/retrievers/balanced_retriever",
#         "color": "🟡"
#     },
#     "quality": {
#         "name": "🎯 Quality Pipeline",
#         "description": "Chất lượng cao - Semantic + BGE-M3",
#         "chunking": "semantic",
#         "embedding": "bge-m3",
#         "chunker_class": SemanticChunker,
#         "chunker_params": {"detect_headings": True},
#         "retriever_path": "output/retrievers/quality_retriever",
#         "color": "🔴"
#     }
# }
#
#
# # ==================== HÀM 1: CORE PIPELINE ====================
# class CorePipeline:
#     """OCR + LLM Cleaning"""
#
#     @staticmethod
#     def process_pdf(pdf_file, progress_callback=None):
#         """Process PDF: OCR + LLM Clean"""
#         pdf_path = Path("output/temp_upload.pdf")
#         pdf_path.parent.mkdir(exist_ok=True)
#
#         with open(pdf_path, "wb") as f:
#             f.write(pdf_file.read())
#
#         if progress_callback:
#             progress_callback("Step 1/2: OCR + Extract text...", 0.25)
#
#         # Run OCR
#         try:
#             import core.ocr_text_processor as ocr_module
#             processor = ocr_module.OCRProcessor()
#             processor.process_pdf(str(pdf_path))
#
#             if not Path("output/before_llm.txt").exists():
#                 raise Exception("OCR failed")
#         except Exception as e:
#             raise Exception(f"OCR failed: {str(e)}")
#
#         if progress_callback:
#             progress_callback("Step 2/2: LLM Cleaning...", 0.50)
#
#         # Run LLM cleaner
#         try:
#             import core.llm_clean_optimal as llm_module
#             cleaner = llm_module.OptimalLLMCleaner()
#             cleaner.process()
#
#             if not Path("output/after_llm.txt").exists():
#                 raise Exception("LLM cleaning failed")
#         except Exception as e:
#             raise Exception(f"LLM cleaning failed: {str(e)}")
#
#         if progress_callback:
#             progress_callback("Core pipeline completed!", 1.0)
#
#         return Path("output/after_llm.txt")
#
#
# # ==================== HÀM 2: CHUNKING ====================
# class ChunkingPipeline:
#     """Chunking"""
#
#     @staticmethod
#     def parse_after_llm(file_path):
#         """Parse after_llm.txt"""
#         import re
#
#         content = Path(file_path).read_text(encoding='utf-8')
#         separator = '=' * 80
#         blocks = content.split(separator)
#
#         pages = []
#         current_page_num = None
#
#         for block in blocks:
#             if not block.strip():
#                 continue
#
#             lines = block.strip().split('\n')
#
#             page_match = None
#             for line in lines:
#                 match = re.match(r'PAGE\s+(\d+)', line)
#                 if match:
#                     page_match = match
#                     break
#
#             if page_match:
#                 current_page_num = int(page_match.group(1))
#                 pages.append({'page_num': current_page_num, 'text': ''})
#             else:
#                 text = block.strip()
#                 if text and current_page_num is not None:
#                     for page in pages:
#                         if page['page_num'] == current_page_num:
#                             if page['text']:
#                                 page['text'] += '\n\n' + text
#                             else:
#                                 page['text'] = text
#                             break
#
#         pages = [p for p in pages if p['text'].strip()]
#         return pages
#
#     @staticmethod
#     def chunk_pages(pages, config):
#         """Chunk pages"""
#         chunker = config['chunker_class'](**config['chunker_params'])
#         chunks = chunker.chunk_document(pages)
#         return chunks
#
#
# # ==================== HÀM 3: EMBEDDING ====================
# class EmbeddingPipeline:
#     """Embedding with GPU priority"""
#
#     @staticmethod
#     def create_embedder(embedding_model):
#         """Create embedder instance"""
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#         if device == "cuda":
#             batch_sizes = {"bge-m3": 32, "gte-viet": 48, "minilm": 64}
#         else:
#             batch_sizes = {"bge-m3": 32, "gte-viet": 32, "minilm": 64}
#
#         batch_size = batch_sizes.get(embedding_model, 32)
#
#         if embedding_model == "bge-m3":
#             embedder = SingleEmbedder(
#                 model_name="BAAI/bge-m3",
#                 device=device,
#                 batch_size=batch_size
#             )
#         elif embedding_model == "gte-viet":
#             embedder = VietnameseEmbedder(
#                 model_key="gte-viet",
#                 device=device,
#                 batch_size=batch_size
#             )
#         elif embedding_model == "minilm":
#             embedder = SingleEmbedder(
#                 model_name="sentence-transformers/all-MiniLM-L6-v2",
#                 device=device,
#                 batch_size=batch_size
#             )
#         else:
#             raise ValueError(f"Unknown embedding model: {embedding_model}")
#
#         return embedder
#
#     @staticmethod
#     def embed_chunks(chunks, embedding_model):
#         """Embed chunks"""
#         embedder = EmbeddingPipeline.create_embedder(embedding_model)
#         embeddings = embedder.encode_chunks(chunks, text_key='text')
#         return embeddings, embedder
#
#
# # ==================== HÀM 4: CHATBOT ====================
# class ChatbotApp:
#     """Main chatbot with RAG"""
#
#     def __init__(self, pipeline_config):
#         self.config = pipeline_config
#         self.retriever = None
#
#     def build_and_save_retriever(self, chunks, embeddings, embedder):
#         """Build retriever and save to disk"""
#         self.retriever = DenseRetriever(embedder)
#         self.retriever.fit(chunks, embeddings)
#
#         # Save retriever
#         self.retriever.save(self.config['retriever_path'])
#
#         return self.retriever
#
#     def load_retriever(self, embedder):
#         """Load retriever from disk"""
#         retriever_path = self.config['retriever_path']
#
#         if not DenseRetriever.exists(retriever_path):
#             raise FileNotFoundError(f"Retriever not found at {retriever_path}")
#
#         self.retriever = DenseRetriever.load(retriever_path, embedder)
#         return self.retriever
#
#     def retrieve(self, query, top_k=3):
#         """Retrieve"""
#         if self.retriever is None:
#             return []
#         return self.retriever.search(query, top_k=top_k)
#
#     def generate_answer(self, query, top_k=3):
#         """Generate answer with GEMINI"""
#         results = self.retrieve(query, top_k=top_k)
#
#         if not results:
#             return {
#                 'answer': "❌ Không tìm thấy thông tin liên quan.",
#                 'sources': [],
#                 'success': False
#             }
#
#         context_parts = []
#         for i, r in enumerate(results, 1):
#             page = r.get('page_num', '?')
#             text = r['text']
#             context_parts.append(f"[Nguồn {i} - Trang {page}]\n{text}\n")
#
#         context = "\n".join(context_parts)
#         answer, success = self._generate_gemini(query, context)
#
#         return {
#             'answer': answer,
#             'sources': results,
#             'success': success
#         }
#
#     def _generate_gemini(self, query, context):
#         """Generate with GEMINI"""
#         if not GEMINI_API_KEY:
#             return "❌ GEMINI API key not found!", False
#
#         prompt = f"""Bạn là trợ lý giáo viên Toán. Dựa vào NGỮ CẢNH, trả lời câu hỏi.
#
# NGUYÊN TẮC:
# - Chỉ dùng thông tin từ NGỮ CẢNH
# - Nếu không có info → nói "Không tìm thấy trong sách"
# - Giải thích đơn giản, dễ hiểu
#
# NGỮ CẢNH:
# {context}
#
# CÂU HỎI: {query}
#
# TRẢ LỜI:"""
#
#         try:
#             response = requests.post(
#                 f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}",
#                 json={
#                     "contents": [{"parts": [{"text": prompt}]}],
#                     "generationConfig": {
#                         "temperature": 0.3,
#                         "maxOutputTokens": 2048
#                     }
#                 },
#                 timeout=30
#             )
#
#             if response.status_code == 200:
#                 return response.json()['candidates'][0]['content']['parts'][0]['text'], True
#             else:
#                 return f"❌ Error: {response.status_code}", False
#         except Exception as e:
#             return f"❌ Exception: {str(e)}", False
#
#
# # ==================== STREAMLIT UI ====================
# def main():
#     if 'pipeline_mode' not in st.session_state:
#         query_params = st.query_params
#         if 'mode' in query_params:
#             st.session_state.pipeline_mode = query_params['mode']
#         else:
#             st.session_state.pipeline_mode = "balanced"
#
#     mode = st.session_state.pipeline_mode
#     config = PIPELINE_CONFIGS.get(mode, PIPELINE_CONFIGS['balanced'])
#
#     st.set_page_config(
#         page_title=f"Chatbot - {config['name']}",
#         page_icon="🤖",
#         layout="wide"
#     )
#
#     # Header
#     st.title(f"🤖 RAG Chatbot - {config['color']} {config['name']}")
#     st.markdown(f"**{config['description']}**")
#
#     # Check if retriever exists
#     retriever_exists = DenseRetriever.exists(config['retriever_path'])
#
#     if retriever_exists:
#         st.success(f"✅ Retriever đã tồn tại! Có thể load ngay hoặc process lại.")
#
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.info(f"**Chunking:** {config['chunking'].title()}")
#     with col2:
#         st.info(f"**Embedding:** {config['embedding'].upper()}")
#     with col3:
#         st.info(f"**Status:** {'💾 Saved' if retriever_exists else '🆕 New'}")
#
#     st.markdown("---")
#
#     # Sidebar
#     with st.sidebar:
#         st.header("⚙️ Settings")
#
#         top_k = st.slider("Top-K sources:", 1, 5, 3)
#         show_sources = st.checkbox("Show sources", value=True)
#
#         st.markdown("---")
#         st.markdown("### 📊 Pipeline Info")
#         st.markdown(f"**Mode:** {mode}")
#         st.markdown(f"**Chunking:** {config['chunking']}")
#         st.markdown(f"**Embedding:** {config['embedding']}")
#         st.markdown(f"**Retriever:** {'Saved ✅' if retriever_exists else 'Not created'}")
#
#     # Initialize
#     if 'chatbot' not in st.session_state:
#         st.session_state.chatbot = None
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'processed' not in st.session_state:
#         st.session_state.processed = False
#
#     # Tabs
#     tab1, tab2 = st.tabs(["📤 Process / Load", "💬 Chat"])
#
#     with tab1:
#         st.header("📤 Upload PDF or Load Existing")
#
#         # Option 1: Load existing
#         if retriever_exists:
#             st.success("💾 Retriever đã được lưu trước đó!")
#
#             if st.button("⚡ Load Retriever (Instant!)", type="primary"):
#                 with st.spinner("Loading..."):
#                     try:
#                         st.session_state.chatbot = ChatbotApp(config)
#
#                         # Create embedder
#                         embedder = EmbeddingPipeline.create_embedder(config['embedding'])
#
#                         # Load retriever
#                         st.session_state.chatbot.load_retriever(embedder)
#
#                         st.session_state.processed = True
#                         st.success("✅ Loaded! Go to Chat tab →")
#
#                     except Exception as e:
#                         st.error(f"❌ Load failed: {str(e)}")
#
#         st.markdown("---")
#
#         # Option 2: Process new PDF
#         st.subheader("🆕 Process New PDF")
#
#         uploaded_file = st.file_uploader(
#             "Choose PDF file",
#             type=['pdf'],
#             help="Upload sách giáo khoa Toán"
#         )
#
#         if uploaded_file is not None:
#             st.success(f"✅ Uploaded: {uploaded_file.name}")
#
#             if st.button("🚀 Process Full Pipeline", type="secondary"):
#                 st.session_state.processed = False
#                 st.session_state.chatbot = ChatbotApp(config)
#
#                 progress_bar = st.progress(0)
#                 status_text = st.empty()
#
#                 try:
#                     # Hàm 1: Core
#                     status_text.text("Hàm 1/4: Core Pipeline (OCR + LLM)...")
#
#                     def core_progress(msg, progress):
#                         status_text.text(f"Hàm 1/4: {msg}")
#                         progress_bar.progress(progress * 0.5)
#
#                     after_llm_file = CorePipeline.process_pdf(
#                         uploaded_file,
#                         progress_callback=core_progress
#                     )
#                     st.success(f"✅ Core completed")
#
#                     # Hàm 2: Chunking
#                     status_text.text(f"Hàm 2/4: Chunking...")
#                     progress_bar.progress(0.6)
#
#                     pages = ChunkingPipeline.parse_after_llm(after_llm_file)
#                     chunks = ChunkingPipeline.chunk_pages(pages, config)
#                     st.success(f"✅ {len(chunks)} chunks")
#
#                     # Hàm 3: Embedding
#                     status_text.text(f"Hàm 3/4: Embedding...")
#                     progress_bar.progress(0.7)
#
#                     embeddings, embedder = EmbeddingPipeline.embed_chunks(
#                         chunks,
#                         config['embedding']
#                     )
#
#                     device = "GPU" if torch.cuda.is_available() else "CPU"
#                     st.success(f"✅ Embedded on {device}")
#
#                     # Hàm 4: Build & Save
#                     status_text.text("Hàm 4/4: Building & Saving retriever...")
#                     progress_bar.progress(0.9)
#
#                     st.session_state.chatbot.build_and_save_retriever(
#                         chunks, embeddings, embedder
#                     )
#
#                     progress_bar.progress(1.0)
#                     st.success("✅ Retriever saved!")
#
#                     status_text.text("✅ All done! Go to Chat tab →")
#                     st.session_state.processed = True
#
#                     # Stats
#                     st.markdown("---")
#                     col1, col2, col3, col4 = st.columns(4)
#                     col1.metric("Pages", len(pages))
#                     col2.metric("Chunks", len(chunks))
#                     col3.metric("Device", device)
#                     col4.metric("Saved", "✅")
#
#                 except Exception as e:
#                     st.error(f"❌ Error: {str(e)}")
#                     st.exception(e)
#
#     with tab2:
#         if not st.session_state.processed:
#             st.warning("⚠️ Please load or process first")
#             st.stop()
#
#         st.header("💬 Chat with your document")
#
#         # Display history
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])
#
#                 if "sources" in message and show_sources and message["sources"]:
#                     with st.expander("📚 Sources"):
#                         for src in message["sources"]:
#                             st.markdown(f"**Rank {src['rank']} - Page {src.get('page_num', '?')} - Score: {src['score']:.3f}**")
#                             st.text(src['text'][:200] + "...")
#                             st.markdown("---")
#
#         # Chat input
#         if prompt := st.chat_input("Ask about the document..."):
#             st.session_state.messages.append({
#                 "role": "user",
#                 "content": prompt
#             })
#
#             with st.chat_message("user"):
#                 st.markdown(prompt)
#
#             with st.chat_message("assistant"):
#                 with st.spinner("🤔 Thinking..."):
#                     result = st.session_state.chatbot.generate_answer(
#                         prompt,
#                         top_k=top_k
#                     )
#
#                     st.markdown(result['answer'])
#
#                     if result['sources'] and show_sources:
#                         with st.expander("📚 Sources"):
#                             for src in result['sources']:
#                                 st.markdown(f"**Rank {src['rank']} - Page {src.get('page_num', '?')} - Score: {src['score']:.3f}**")
#                                 st.text(src['text'][:200] + "...")
#                                 st.markdown("---")
#
#                     st.session_state.messages.append({
#                         "role": "assistant",
#                         "content": result['answer'],
#                         "sources": result['sources']
#                     })
#
#         if st.button("🗑️ Clear chat"):
#             st.session_state.messages = []
#             st.rerun()
#
#
# if __name__ == "__main__":
#     main()
"""
app_chatbot_modular.py - FINAL 100% - GEMINI FULL ANSWER (VN - 16/11/2025)
- Output dài, chi tiết, đầy đủ
- Đưa cả 3 đoạn top retrieval vào prompt
- Bắt buộc: DỰA VÀO SÁCH, KHÔNG BỊA
- Dựa 100% trên curl: X-goog-api-key + gemini-2.0-flash
- Retry 2 lần, delay 2s → không 429
- SKIP OCR+LLM + GPU + chunk.txt + metadata.json
"""

import streamlit as st
import sys
from pathlib import Path
import torch
import os
import re
import json
import time
import numpy as np
import requests

# === ĐƯỜNG DẪN GỐC ===
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# === IMPORT ===
try:
    from core.text_ocr_processor import OCRProcessor
    from core.llm_clean_ocr import OptimalLLMCleaner
    from embedders.single_embedder import SingleEmbedder
    from embedders.vietnamese_embedder import VietnameseEmbedder
    from chunkers.recursive_chunker import RecursiveChunker
    from chunkers.page_aware_chunker import PageAwareChunker
    from chunkers.semantic_chunker import SemanticChunker
    from retrievers.dense_retriever import DenseRetriever
except ImportError as e:
    st.error(f"Import lỗi: {e}")
    st.code("""
    Đảm bảo:
    1. core/text_ocr_processor.py
    2. core/llm_clean_ocr.py
    3. core/__init__.py có:
       from .text_ocr_processor import OCRProcessor
       from .llm_clean_ocr import OptimalLLMCleaner
    """)
    st.stop()

from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Thiếu GEMINI_API_KEY trong .env!")
    st.stop()


# ==================== CONFIGS ====================
PIPELINES = {
    "fast": {
        "name": "Fast",
        "desc": "Recursive + MiniLM",
        "chunker": RecursiveChunker,
        "params": {"chunk_size": 512, "overlap": 50},
        "embed": "minilm",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "path": "output/retrievers/fast",
    },
    "balanced": {
        "name": "Balanced",
        "desc": "Page-Aware + GTE-Viet",
        "chunker": PageAwareChunker,
        "params": {"max_chars": 2000},
        "embed": "gte-viet",
        "model": "Alibaba-NLP/gte-multilingual-base",
        "path": "output/retrievers/balanced",
    },
    "quality": {
        "name": "Quality",
        "desc": "Semantic + BGE-M3",
        "chunker": SemanticChunker,
        "params": {"detect_headings": True},
        "embed": "bge-m3",
        "model": "BAAI/bge-m3",
        "path": "output/retrievers/quality",
    }
}


# ==================== PARSER ====================
def parse_pages_correctly(file_path):
    content = Path(file_path).read_text(encoding='utf-8')
    separator = '=' * 80
    blocks = content.split(separator)
    pages = []
    current_page_num = None

    for block in blocks:
        if not block.strip(): continue
        lines = block.strip().split('\n')
        page_match = None
        for line in lines:
            match = re.match(r'PAGE\s+(\d+)', line)
            if match:
                page_match = match
                break
        if page_match:
            page_num = int(page_match.group(1))
            current_page_num = page_num
            pages.append({'page_num': page_num, 'text': ''})
        else:
            text = block.strip()
            if text and current_page_num is not None:
                for page in pages:
                    if page['page_num'] == current_page_num:
                        if page['text']:
                            page['text'] += '\n\n' + text
                        else:
                            page['text'] = text
                        break
    return [p for p in pages if p['text'].strip()]


# ==================== SAVE CHUNKS ====================
def save_chunks_to_file(chunks, strategy_name):
    filepath = project_root / "output" / f"chunks_{strategy_name}.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"CHUNKS - {strategy_name.upper()}\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")
        sizes = [len(c['text']) for c in chunks]
        f.write(f"Average size: {sum(sizes)/len(sizes):.0f} chars\n")
        f.write(f"Min size: {min(sizes)} chars\n")
        f.write(f"Max size: {max(sizes)} chars\n\n")
        f.write("="*80 + "\n\n")
        for i, c in enumerate(chunks, 1):
            f.write("="*80 + "\n")
            f.write(f"CHUNK {i}/{len(chunks)}\n")
            f.write("="*80 + "\n")
            f.write(f"Page: {c.get('page_num', 'N/A')}\n")
            f.write(f"Size: {len(c['text'])} chars\n")
            f.write(f"Type: {c.get('chunk_type', 'N/A')}\n\n")
            f.write(c['text'].strip() + "\n\n")
    return filepath


# ==================== CORE PIPELINE ====================
class CorePipeline:
    @staticmethod
    def run(pdf_file, callback=None):
        pdf_path = project_root / "output" / "temp.pdf"
        pdf_path.write_bytes(pdf_file.read())

        if callback: callback("OCR + Extract...", 0.3)
        OCRProcessor().process_pdf(str(pdf_path))

        if callback: callback("LLM Cleaning...", 0.6)
        OptimalLLMCleaner().process()

        after_path = project_root / "output" / "after_llm.txt"
        if not after_path.exists():
            raise RuntimeError("after_llm.txt not created")
        return after_path


# ==================== CHUNK & EMBED ====================
def chunk_and_embed(after_path, config):
    pages = parse_pages_correctly(after_path)
    if not pages:
        raise ValueError("Không parse được trang nào!")

    chunker = config['chunker'](**config['params'])
    chunks = chunker.chunk_document(pages)
    if not chunks:
        raise ValueError("Không tạo được chunk!")

    chunk_file = save_chunks_to_file(chunks, config['embed'])
    st.success(f"Đã tạo {len(chunks)} chunks → {chunk_file.name}")

    has_gpu = torch.cuda.is_available()
    device = "cuda" if has_gpu else "cpu"
    batch_size_map = {"bge-m3": 32, "gte-viet": 48, "minilm": 64}
    batch_size = batch_size_map.get(config['embed'], 32)
    if not has_gpu:
        batch_size = max(8, batch_size // 2)

    if config['embed'] == "bge-m3":
        embedder = SingleEmbedder(model_name="BAAI/bge-m3", device=device, batch_size=batch_size)
    elif config['embed'] == "gte-viet":
        embedder = VietnameseEmbedder(model_key="gte-viet", device=device, batch_size=batch_size)
    else:
        embedder = SingleEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device, batch_size=batch_size)

    start = time.time()
    embeddings = embedder.encode_chunks(chunks, text_key='text')
    elapsed = time.time() - start

    if len(embeddings) == 0:
        raise ValueError("Embeddings rỗng!")

    emb_dir = project_root / "output" / "embeddings"
    emb_dir.mkdir(exist_ok=True, parents=True)

    emb_file = emb_dir / f"{config['embed']}_embeddings.npy"
    embedder.save_embeddings(embeddings, str(emb_file))

    meta = {
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "strategy": config['embed'],
        "device": device,
        "batch_size": batch_size,
        "time_seconds": elapsed,
        "chunk_file": str(chunk_file),
    }
    meta_file = emb_dir / f"{config['embed']}_metadata.json"
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    st.info(f"Embedding: {embeddings.shape} | {elapsed:.1f}s | {device.upper()}")

    return chunks, embeddings, embedder


# ==================== CHATBOT – FULL ANSWER + 3 SOURCES ====================
class Chatbot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.retriever = None

    def build(self, chunks, embs, embedder):
        self.retriever = DenseRetriever(embedder)
        self.retriever.fit(chunks, embs)
        self.retriever.save(self.cfg['path'])

    def load(self, embedder):
        if not DenseRetriever.exists(self.cfg['path']):
            raise FileNotFoundError("Retriever not found")
        self.retriever = DenseRetriever.load(self.cfg['path'], embedder)

    def ask(self, q, k=3):
        if not self.retriever:
            return {"answer": "Bot chưa sẵn sàng", "sources": []}
        results = self.retriever.search(q, top_k=k)
        if not results:
            return {"answer": "Không tìm thấy thông tin trong sách", "sources": []}

        # ĐƯA CẢ 3 ĐOẠN TOP VÀO PROMPT
        ctx = "\n\n".join([
            f"--- NGUỒN {i+1} (Trang {r.get('page_num','?')} | Độ tương đồng: {r['score']:.3f}) ---\n{r['text']}"
            for i, r in enumerate(results)
        ])

        ans = self._gemini_full_answer(q, ctx)
        return {"answer": ans, "sources": results}

    def _gemini_full_answer(self, q, ctx):
        """
        BẮT BUỘC: DỰA VÀO SÁCH, KHÔNG BỊA, TRẢ LỜI CHI TIẾT
        Dựa 100% trên curl
        """
        prompt = f"""Bạn là chuyên gia giáo dục, trả lời câu hỏi DỰA HOÀN TOÀN VÀO 3 đoạn trích từ sách giáo khoa dưới đây.

YÊU CẦU NGHIÊM NGẶT:
1. CHỈ DỰA VÀO NỘI DUNG TRONG 3 NGUỒN DƯỚI ĐÂY.
2. KHÔNG BỊA THÊM, KHÔNG SUY DIỄN NGOÀI SÁCH.
3. TRẢ LỜI CHI TIẾT, ĐẦY ĐỦ, CÓ CẤU TRÚC RÕ RÀNG.
4. TRÍCH DẪN NGUỒN (Trang X) khi dùng thông tin.
5. Nếu không đủ thông tin → nói rõ: "Không có trong sách".

NGUỒN TỪ SÁCH:
{ctx}

CÂU HỎI: {q}

TRẢ LỜI CHI TIẾT:"""

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": GEMINI_API_KEY
        }

        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048  # TĂNG LÊN ĐỂ TRẢ LỜI DÀI
            }
        }

        for attempt in range(1, 3):
            try:
                with st.spinner(f"Gemini đang trả lời chi tiết (lần {attempt}/2)..."):
                    resp = requests.post(url, headers=headers, json=data, timeout=60)

                if resp.status_code == 200:
                    try:
                        answer = resp.json()['candidates'][0]['content']['parts'][0]['text']
                        return answer.strip()
                    except:
                        st.error("Gemini JSON lỗi")
                        return "Lỗi phân tích phản hồi"

                elif resp.status_code == 429:
                    st.warning("Gemini 429: Đợi 2s...")
                    time.sleep(2)
                    continue
                else:
                    st.error(f"Gemini lỗi {resp.status_code}")
                    if attempt == 2:
                        return "Lỗi server"

            except requests.exceptions.Timeout:
                st.warning("Timeout. Thử lại...")
                time.sleep(2)
            except Exception as e:
                st.error(f"Lỗi: {str(e)[:60]}")
                if attempt == 2:
                    return "Gemini không phản hồi"

        return "Không thể lấy câu trả lời sau 2 lần thử"


# ==================== UI ====================
def main():
    st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="Robot")

    st.sidebar.header("Pipeline")
    mode = st.sidebar.radio("Mode", list(PIPELINES.keys()),
                            format_func=lambda x: f"{PIPELINES[x]['name']} - {PIPELINES[x]['desc']}",
                            index=1)
    cfg = PIPELINES[mode]

    st.title(f"RAG Chatbot - {cfg['name']}")
    st.markdown(f"*{cfg['desc']}*")

    saved = DenseRetriever.exists(cfg['path'])
    st.info(f"**Status:** {'Đã lưu' if saved else 'Chưa xử lý'}")

    for k in ['bot', 'msg', 'ready']:
        if k not in st.session_state:
            st.session_state[k] = None if k == 'bot' else ([] if k == 'msg' else False)

    t1, t2 = st.tabs(["Xử lý / Load", "Chat"])

    with t1:
        col1, col2 = st.columns([1, 2])
        with col1:
            if saved and st.button("Load Retriever", type="primary"):
                with st.spinner("Loading..."):
                    bot = Chatbot(cfg)
                    embedder = SingleEmbedder(model_name=cfg['model'],
                                              device="cuda" if torch.cuda.is_available() else "cpu")
                    bot.load(embedder)
                    st.session_state.bot = bot
                    st.session_state.ready = True
                    st.success("Loaded!")

            skip_file = project_root / "output" / "after_llm.txt"
            if skip_file.exists() and st.button("SKIP OCR+LLM (Dùng file có sẵn)", type="secondary"):
                with st.spinner("Đang dùng after_llm.txt..."):
                    after = skip_file
                    bot = Chatbot(cfg)
                    st.session_state.bot = bot
                    prog = st.progress(0)
                    stat = st.empty()

                    try:
                        stat.text("2. Chunking + Embedding...")
                        chunks, embs, embedder = chunk_and_embed(after, cfg)
                        prog.progress(0.7)

                        stat.text("3. Saving...")
                        bot.build(chunks, embs, embedder)
                        prog.progress(1.0)
                        st.session_state.ready = True
                        st.success("HOÀN THÀNH (SKIP OCR+LLM)!")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Chunks", len(chunks))
                        c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
                        c3.metric("Saved", "Yes")

                    except Exception as e:
                        st.error(f"Lỗi: {e}")
                        st.exception(e)

            uploaded = st.file_uploader("Upload PDF", type=['pdf'])

        if uploaded and st.button("Process Full Pipeline", type="secondary"):
            bot = Chatbot(cfg)
            st.session_state.bot = bot
            prog = st.progress(0)
            stat = st.empty()

            try:
                stat.text("1. OCR + LLM...")
                after = CorePipeline.run(uploaded, lambda m, p: (stat.text(m), prog.progress(p * 0.4)))
                prog.progress(0.4)

                stat.text("2. Chunking + Embedding...")
                chunks, embs, embedder = chunk_and_embed(after, cfg)
                prog.progress(0.7)

                stat.text("3. Saving...")
                bot.build(chunks, embs, embedder)
                prog.progress(1.0)
                st.session_state.ready = True
                st.success("HOÀN THÀNH!")

                c1, c2, c3 = st.columns(3)
                c1.metric("Chunks", len(chunks))
                c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
                c3.metric("Saved", "Yes")

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Xem chunk.txt"):
                        f = project_root / "output" / f"chunks_{cfg['embed']}.txt"
                        if f.exists():
                            st.code(f.read_text(encoding='utf-8')[:2000])
                with col_b:
                    if st.button("Xem metadata.json"):
                        f = project_root / "output" / "embeddings" / f"{cfg['embed']}_metadata.json"
                        if f.exists():
                            st.json(json.loads(f.read_text(encoding='utf-8')))

            except Exception as e:
                st.error(f"Lỗi: {e}")
                st.exception(e)

    with t2:
        if not st.session_state.ready:
            st.warning("Vui lòng xử lý trước")
            st.stop()

        for m in st.session_state.msg:
            with st.chat_message(m['role']):
                st.markdown(m['content'])
                if m.get('src'):
                    with st.expander("Nguồn (3 đoạn trích)"):
                        for s in m['src']:
                            st.caption(f"Trang {s.get('page_num','?')} | Score: {s['score']:.3f}")
                            st.text(s['text'][:300] + ("..." if len(s['text']) > 300 else ""))

        if q := st.chat_input("Hỏi nội dung về sách..."):
            st.session_state.msg.append({"role": "user", "content": q})
            with st.chat_message("user"): st.markdown(q)
            with st.chat_message("assistant"):
                res = st.session_state.bot.ask(q)
                st.markdown(res['answer'])
                st.session_state.msg.append({
                    "role": "assistant",
                    "content": res['answer'],
                    "src": res['sources']
                })

        if st.button("Xóa"):
            st.session_state.msg = []
            st.rerun()


if __name__ == "__main__":
    main()