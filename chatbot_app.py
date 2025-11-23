# """
# app_chatbot_modular.py - FINAL 100% - ICON ĐẸP + KHÔNG LỖI (VN - 16/11/2025 08:53 AM)
# - Icon hiển thị 100%: bolt, sliders, star
# - Sửa lỗi HTML escape → dùng st.button + st.markdown
# - UI đẹp, responsive, không crash
# """
#
# import streamlit as st
# import sys
# from pathlib import Path
# import torch
# import os
# import re
# import json
# import time
# import numpy as np
# import requests
#
# # === ĐƯỜNG DẪN GỐC ===
# project_root = Path(__file__).resolve().parent
# sys.path.insert(0, str(project_root))
#
# # === IMPORT ===
# try:
#     from core.text_ocr_processor import OCRProcessor
#     from core.llm_clean_ocr import OptimalLLMCleaner
#     from embedders.single_embedder import SingleEmbedder
#     from embedders.vietnamese_embedder import VietnameseEmbedder
#     from chunkers.recursive_chunker import RecursiveChunker
#     from chunkers.page_aware_chunker import PageAwareChunker
#     from chunkers.semantic_chunker import SemanticChunker
#     from retrievers.dense_retriever import DenseRetriever
# except ImportError as e:
#     st.error(f"Import lỗi: {e}")
#     st.code("""
#     Đảm bảo:
#     1. core/text_ocr_processor.py
#     2. core/llm_clean_ocr.py
#     3. core/__init__.py có:
#        from .text_ocr_processor import OCRProcessor
#        from .llm_clean_ocr import OptimalLLMCleaner
#     """)
#     st.stop()
#
# from dotenv import load_dotenv
#
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#
# if not GEMINI_API_KEY:
#     st.error("Thiếu GEMINI_API_KEY trong .env!")
#     st.stop()
#
# # ==================== CONFIGS ====================
# PIPELINES = {
#     "fast": {
#         "name": "Fast",
#         "desc": "Recursive + MiniLM",
#         "chunker": RecursiveChunker,
#         "params": {"chunk_size": 512, "overlap": 50},
#         "embed": "minilm",
#         "model": "sentence-transformers/all-MiniLM-L6-v2",
#         "path": "output/retrievers/fast",
#         "icon": "bolt",
#         "color": "#10B981"
#     },
#     "balanced": {
#         "name": "Balanced",
#         "desc": "Page-Aware + GTE-Viet",
#         "chunker": PageAwareChunker,
#         "params": {"max_chars": 2000},
#         "embed": "gte-viet",
#         "model": "Alibaba-NLP/gte-multilingual-base",
#         "path": "output/retrievers/balanced",
#         "icon": "sliders",
#         "color": "#3B82F6"
#     },
#     "quality": {
#         "name": "Quality",
#         "desc": "Semantic + BGE-M3",
#         "chunker": SemanticChunker,
#         "params": {"detect_headings": True},
#         "embed": "bge-m3",
#         "model": "BAAI/bge-m3",
#         "path": "output/retrievers/quality",
#         "icon": "star",
#         "color": "#8B5CF6"
#     }
# }
#
# # ==================== PAGE CONFIG + FONT AWESOME ====================
# st.set_page_config(
#     page_title="MIBook BOT",
#     page_icon="robot",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# st.markdown("""
# <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
# """, unsafe_allow_html=True)
#
# st.markdown("""
# <style>
#     .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; }
#     .title { font-size: 3rem !important; font-weight: 800; background: linear-gradient(90deg, #FFF, #FBBF24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
#     .pipeline-btn { text-align: center; padding: 1rem; border-radius: 16px; transition: all 0.3s; }
#     .pipeline-btn:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0,0,0,0.2); }
#     .stButton > button { width: 100%; height: 100%; border-radius: 12px; font-weight: 600; }
#     .metric-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid rgba(255,255,255,0.15); }
#     .source-box { background: rgba(255,255,255,0.08); border-left: 4px solid #FBBF24; padding: 0.75rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; font-size: 0.9rem; }
# </style>
# """, unsafe_allow_html=True)
#
#
# # ==================== HELPER FUNCTIONS ====================
# def parse_pages_correctly(file_path):
#     content = Path(file_path).read_text(encoding='utf-8')
#     separator = '=' * 80
#     blocks = content.split(separator)
#     pages = []
#     current_page_num = None
#     for block in blocks:
#         if not block.strip(): continue
#         lines = block.strip().split('\n')
#         page_match = None
#         for line in lines:
#             match = re.match(r'PAGE\s+(\d+)', line)
#             if match:
#                 page_match = match
#                 break
#         if page_match:
#             page_num = int(page_match.group(1))
#             current_page_num = page_num
#             pages.append({'page_num': page_num, 'text': ''})
#         else:
#             text = block.strip()
#             if text and current_page_num is not None:
#                 for page in pages:
#                     if page['page_num'] == current_page_num:
#                         if page['text']:
#                             page['text'] += '\n\n' + text
#                         else:
#                             page['text'] = text
#                         break
#     return [p for p in pages if p['text'].strip()]
#
#
# def save_chunks_to_file(chunks, strategy_name):
#     filepath = project_root / "output" / f"chunks_{strategy_name}.txt"
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write("=" * 80 + "\n")
#         f.write(f"CHUNKS - {strategy_name.upper()}\n")
#         f.write(f"=" * 80 + "\n\n")
#         f.write(f"Total chunks: {len(chunks)}\n\n")
#         sizes = [len(c['text']) for c in chunks]
#         f.write(f"Average size: {sum(sizes) / len(sizes):.0f} chars\n")
#         f.write(f"Min size: {min(sizes)} chars\n")
#         f.write(f"Max size: {max(sizes)} chars\n\n")
#         f.write("=" * 80 + "\n\n")
#         for i, c in enumerate(chunks, 1):
#             f.write("=" * 80 + "\n")
#             f.write(f"CHUNK {i}/{len(chunks)}\n")
#             f.write("=" * 80 + "\n")
#             f.write(f"Page: {c.get('page_num', 'N/A')}\n")
#             f.write(f"Size: {len(c['text'])} chars\n")
#             f.write(f"Type: {c.get('chunk_type', 'N/A')}\n\n")
#             f.write(c['text'].strip() + "\n\n")
#     return filepath
#
#
# # ==================== CORE PIPELINE ====================
# class CorePipeline:
#     @staticmethod
#     def run(pdf_file, callback=None):
#         pdf_path = project_root / "output" / "temp.pdf"
#         pdf_path.write_bytes(pdf_file.read())
#         if callback: callback("OCR + Extract...", 0.3)
#         OCRProcessor().process_pdf(str(pdf_path))
#         if callback: callback("LLM Cleaning...", 0.6)
#         OptimalLLMCleaner().process()
#         after_path = project_root / "output" / "after_llm.txt"
#         if not after_path.exists():
#             raise RuntimeError("after_llm.txt not created")
#         return after_path
#
#
# # ==================== CHUNK & EMBED ====================
# def chunk_and_embed(after_path, config):
#     pages = parse_pages_correctly(after_path)
#     if not pages:
#         raise ValueError("Không parse được trang nào!")
#     chunker = config['chunker'](**config['params'])
#     chunks = chunker.chunk_document(pages)
#     if not chunks:
#         raise ValueError("Không tạo được chunk!")
#     chunk_file = save_chunks_to_file(chunks, config['embed'])
#     st.success(f"Đã tạo {len(chunks)} chunks → {chunk_file.name}")
#     has_gpu = torch.cuda.is_available()
#     device = "cuda" if has_gpu else "cpu"
#     batch_size_map = {"bge-m3": 32, "gte-viet": 48, "minilm": 64}
#     batch_size = batch_size_map.get(config['embed'], 32)
#     if not has_gpu:
#         batch_size = max(8, batch_size // 2)
#     if config['embed'] == "bge-m3":
#         embedder = SingleEmbedder(model_name="BAAI/bge-m3", device=device, batch_size=batch_size)
#     elif config['embed'] == "gte-viet":
#         embedder = VietnameseEmbedder(model_key="gte-viet", device=device, batch_size=batch_size)
#     else:
#         embedder = SingleEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device,
#                                   batch_size=batch_size)
#     start = time.time()
#     embeddings = embedder.encode_chunks(chunks, text_key='text')
#     elapsed = time.time() - start
#     if len(embeddings) == 0:
#         raise ValueError("Embeddings rỗng!")
#     emb_dir = project_root / "output" / "embeddings"
#     emb_dir.mkdir(exist_ok=True, parents=True)
#     emb_file = emb_dir / f"{config['embed']}_embeddings.npy"
#     embedder.save_embeddings(embeddings, str(emb_file))
#     meta = {
#         "num_chunks": len(chunks),
#         "embedding_dim": embeddings.shape[1],
#         "strategy": config['embed'],
#         "device": device,
#         "batch_size": batch_size,
#         "time_seconds": elapsed,
#         "chunk_file": str(chunk_file),
#     }
#     meta_file = emb_dir / f"{config['embed']}_metadata.json"
#     with open(meta_file, 'w', encoding='utf-8') as f:
#         json.dump(meta, f, ensure_ascii=False, indent=2)
#     st.info(f"Embedding: {embeddings.shape} | {elapsed:.1f}s | {device.upper()}")
#     return chunks, embeddings, embedder
#
#
# # ==================== CHATBOT ====================
# class Chatbot:
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.retriever = None
#
#     def build(self, chunks, embs, embedder):
#         self.retriever = DenseRetriever(embedder)
#         self.retriever.fit(chunks, embs)
#         self.retriever.save(self.cfg['path'])
#
#     def load(self, embedder):
#         if not DenseRetriever.exists(self.cfg['path']):
#             raise FileNotFoundError("Retriever not found")
#         self.retriever = DenseRetriever.load(self.cfg['path'], embedder)
#
#     def ask(self, q, k=3):
#         if not self.retriever:
#             return {"answer": "Bot chưa sẵn sàng", "sources": []}
#         results = self.retriever.search(q, top_k=k)
#         if not results:
#             return {"answer": "Không tìm thấy thông tin trong sách", "sources": []}
#         ctx = "\n\n".join([
#             f"--- NGUỒN {i + 1} (Trang {r.get('page_num', '?')} | Độ tương đồng: {r['score']:.3f}) ---\n{r['text']}"
#             for i, r in enumerate(results)
#         ])
#         ans = self._gemini_full_answer(q, ctx)
#         return {"answer": ans, "sources": results}
#
#     def _gemini_full_answer(self, q, ctx):
#         prompt = f"""Bạn là chuyên gia giáo dục, trả lời câu hỏi DỰA HOÀN TOÀN VÀO 3 đoạn trích từ sách giáo khoa dưới đây.
#
# YÊU CẦU NGHIÊM NGẶT:
# 1. CHỈ DỰA VÀO NỘI DUNG TRONG 3 NGUỒN DƯỚI ĐÂY.
# 2. KHÔNG BỊA THÊM, KHÔNG SUY DIỄN NGOÀI SÁCH.
# 3. TRẢ LỜI CHI TIẾT, ĐẦY ĐỦ, CÓ CẤU TRÚC RÕ RÀNG.
# 4. TRÍCH DẪN NGUỒN (Trang X) khi dùng thông tin.
# 5. Nếu không đủ thông tin → nói rõ: "Không có trong sách".
#
# NGUỒN TỪ SÁCH:
# {ctx}
#
# CÂU HỎI: {q}
#
# TRẢ LỜI CHI TIẾT:"""
#         url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
#         headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
#         data = {
#             "contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
#         }
#         for attempt in range(1, 3):
#             try:
#                 with st.spinner(f"MIBook BOT đang suy nghĩ... (lần {attempt}/2)"):
#                     resp = requests.post(url, headers=headers, json=data, timeout=60)
#                 if resp.status_code == 200:
#                     return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
#                 elif resp.status_code == 429:
#                     st.warning("Quota hết, đợi 2s...")
#                     time.sleep(2)
#                 else:
#                     if attempt == 2: return "Lỗi server"
#             except:
#                 if attempt == 2: return "Không phản hồi"
#                 time.sleep(2)
#         return "Không thể trả lời"
#
#
# # ==================== MAIN UI ====================
# def main():
#     # === HEADER ===
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.markdown("<h1 class='title'>MIBook BOT</h1>", unsafe_allow_html=True)
#         st.markdown("<p style='text-align:center; color:#94A3B8; font-size:1.1rem;'>AI-powered textbook assistant</p>",
#                     unsafe_allow_html=True)
#
#     # === SIDEBAR - PIPELINE ===
#     with st.sidebar:
#         st.markdown("## Pipeline")
#
#         # SỬA LỖI: DÙNG st.button + st.markdown
#         if "pipeline" not in st.session_state:
#             st.session_state.pipeline = "balanced"
#
#         col1, col2, col3 = st.columns(3)
#         for idx, (key, cfg) in enumerate(PIPELINES.items()):
#             with [col1, col2, col3][idx]:
#                 if st.button(
#                         f"**{cfg['name']}**\n_{cfg['desc']}_",
#                         key=key,
#                         use_container_width=True,
#                         type="primary" if st.session_state.pipeline == key else "secondary"
#                 ):
#                     st.session_state.pipeline = key
#                 st.markdown(
#                     f"<div style='text-align:center; margin-top:8px;'><i class='fa-solid fa-{cfg['icon']}' style='color:{cfg['color']}; font-size:1.8rem;'></i></div>",
#                     unsafe_allow_html=True
#                 )
#
#         cfg = PIPELINES[st.session_state.pipeline]
#         st.markdown("---")
#         st.markdown(
#             f"### <i class='fa-solid fa-{cfg['icon']}' style='color:{cfg['color']}; margin-right:8px;'></i> **{cfg['name']}**",
#             unsafe_allow_html=True)
#         st.markdown(f"*{cfg['desc']}*")
#
#         saved = DenseRetriever.exists(cfg['path'])
#         status_icon = "check-circle" if saved else "circle-xmark"
#         status_color = "#10B981" if saved else "#EF4444"
#         st.markdown(
#             f"**Trạng thái:** <i class='fa-solid fa-{status_icon}' style='color:{status_color}; margin-right:6px;'></i> `{'Đã lưu' if saved else 'Chưa xử lý'}`",
#             unsafe_allow_html=True)
#
#     # === TABS ===
#     tab1, tab2 = st.tabs(["Xử lý PDF", "Chat"])
#
#     # === TAB 1: XỬ LÝ ===
#     with tab1:
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             st.markdown("### Tải lên & Xử lý")
#             if saved:
#                 if st.button("Load Retriever", type="primary", use_container_width=True):
#                     with st.spinner("Đang tải..."):
#                         bot = Chatbot(cfg)
#                         embedder = SingleEmbedder(model_name=cfg['model'],
#                                                   device="cuda" if torch.cuda.is_available() else "cpu")
#                         bot.load(embedder)
#                         st.session_state.bot = bot
#                         st.session_state.ready = True
#                         st.success("Đã tải!")
#             skip_file = project_root / "output" / "after_llm.txt"
#             if skip_file.exists():
#                 if st.button("SKIP OCR+LLM", type="secondary", use_container_width=True):
#                     with st.spinner("Chuẩn bị..."):
#                         after = skip_file
#                         bot = Chatbot(cfg)
#                         st.session_state.bot = bot
#                         prog = st.progress(0)
#                         stat = st.empty()
#                         try:
#                             stat.text("Chunking + Embedding...")
#                             chunks, embs, embedder = chunk_and_embed(after, cfg)
#                             prog.progress(0.7)
#                             stat.text("Lưu...")
#                             bot.build(chunks, embs, embedder)
#                             prog.progress(1.0)
#                             st.session_state.ready = True
#                             st.success("HOÀN THÀNH!")
#                             c1, c2, c3 = st.columns(3)
#                             c1.metric("Chunks", len(chunks))
#                             c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
#                             c3.metric("Saved", "Yes")
#                         except Exception as e:
#                             st.error(f"Lỗi: {e}")
#             uploaded = st.file_uploader("Upload PDF", type=['pdf'], label_visibility="collapsed")
#         if uploaded:
#             if st.button("Process Full Pipeline", type="primary", use_container_width=True):
#                 bot = Chatbot(cfg)
#                 st.session_state.bot = bot
#                 prog = st.progress(0)
#                 stat = st.empty()
#                 try:
#                     stat.text("OCR + LLM...")
#                     after = CorePipeline.run(uploaded, lambda m, p: (stat.text(m), prog.progress(p * 0.4)))
#                     prog.progress(0.4)
#                     stat.text("Chunking + Embedding...")
#                     chunks, embs, embedder = chunk_and_embed(after, cfg)
#                     prog.progress(0.7)
#                     stat.text("Lưu...")
#                     bot.build(chunks, embs, embedder)
#                     prog.progress(1.0)
#                     st.session_state.ready = True
#                     st.success("HOÀN THÀNH!")
#                     c1, c2, c3 = st.columns(3)
#                     c1.metric("Chunks", len(chunks))
#                     c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
#                     c3.metric("Saved", "Yes")
#                 except Exception as e:
#                     st.error(f"Lỗi: {e}")
#
#     # === TAB 2: CHAT ===
#     with tab2:
#         if not st.session_state.get('ready', False):
#             st.warning("Vui lòng xử lý PDF trước")
#             st.stop()
#         for k in ['bot', 'msg', 'ready']:
#             if k not in st.session_state:
#                 st.session_state[k] = None if k == 'bot' else ([] if k == 'msg' else False)
#         for m in st.session_state.msg:
#             with st.chat_message(m['role'], avatar="user" if m['role'] == "user" else "assistant"):
#                 st.markdown(f"**{m['content']}**")
#                 if m.get('src'):
#                     with st.expander("Nguồn (3 đoạn trích)", expanded=False):
#                         for i, s in enumerate(m['src']):
#                             st.markdown(f"""
#                             <div class='source-box'>
#                                 <strong>Nguồn {i + 1}</strong> | Trang {s.get('page_num', '?')} | Score: {s['score']:.3f}<br>
#                                 <small>{s['text'][:400]}{'...' if len(s['text']) > 400 else ''}</small>
#                             </div>
#                             """, unsafe_allow_html=True)
#         if q := st.chat_input("Hỏi gì về sách..."):
#             st.session_state.msg.append({"role": "user", "content": q})
#             with st.chat_message("user", avatar="user"):
#                 st.markdown(f"**{q}**")
#             with st.chat_message("assistant", avatar="assistant"):
#                 # with st.spinner("Đang suy nghĩ..."):
#                     res = st.session_state.bot.ask(q)
#                     st.markdown(res['answer'])
#                     st.session_state.msg.append({
#                         "role": "assistant",
#                         "content": res['answer'],
#                         "src": res['sources']
#                     })
#         if st.button("Xóa lịch sử", type="secondary"):
#             st.session_state.msg = []
#             st.rerun()
#
#
# if __name__ == "__main__":
#     main()
"""
app_chatbot_modular.py - FINAL 100% - ICON ĐẸP + KHÔNG LỖI + SMART RETRIEVAL + FALLBACK (VN - 18/11/2025)
- Icon hiển thị 100%: bolt, sliders, star
- Sửa lỗi HTML escape → dùng st.button + st.markdown
- UI đẹp, responsive, không crash
- RE-RANKING: LLM chọn 2 nguồn tốt nhất từ 5 candidates
- TRẢ LỜI NGẮN GỌN: 150-200 từ, tập trung, không dài dòng
- FALLBACK: Hỏi user có muốn trả lời từ kiến thức chung không
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
        "icon": "bolt",
        "color": "#10B981"
    },
    "balanced": {
        "name": "Balanced",
        "desc": "Page-Aware + GTE-Viet",
        "chunker": PageAwareChunker,
        "params": {"max_chars": 2000},
        "embed": "gte-viet",
        "model": "Alibaba-NLP/gte-multilingual-base",
        "path": "output/retrievers/balanced",
        "icon": "sliders",
        "color": "#3B82F6"
    },
    "quality": {
        "name": "Quality",
        "desc": "Semantic + BGE-M3",
        "chunker": SemanticChunker,
        "params": {"detect_headings": True},
        "embed": "bge-m3",
        "model": "BAAI/bge-m3",
        "path": "output/retrievers/quality",
        "icon": "star",
        "color": "#8B5CF6"
    }
}

# ==================== PAGE CONFIG + FONT AWESOME ====================
st.set_page_config(
    page_title="MIBook BOT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; }
    .title { font-size: 3rem !important; font-weight: 800; background: linear-gradient(90deg, #FFF, #FBBF24); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .pipeline-btn { text-align: center; padding: 1rem; border-radius: 16px; transition: all 0.3s; }
    .pipeline-btn:hover { transform: translateY(-5px); box-shadow: 0 12px 40px rgba(0,0,0,0.2); }
    .stButton > button { width: 100%; height: 100%; border-radius: 12px; font-weight: 600; }
    .metric-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 12px; padding: 1rem; text-align: center; border: 1px solid rgba(255,255,255,0.15); }
    .source-box { background: rgba(255,255,255,0.08); border-left: 4px solid #FBBF24; padding: 0.75rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ==================== HELPER FUNCTIONS ====================
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


def save_chunks_to_file(chunks, strategy_name):
    filepath = project_root / "output" / f"chunks_{strategy_name}.txt"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"CHUNKS - {strategy_name.upper()}\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Total chunks: {len(chunks)}\n\n")
        sizes = [len(c['text']) for c in chunks]
        f.write(f"Average size: {sum(sizes) / len(sizes):.0f} chars\n")
        f.write(f"Min size: {min(sizes)} chars\n")
        f.write(f"Max size: {max(sizes)} chars\n\n")
        f.write("=" * 80 + "\n\n")
        for i, c in enumerate(chunks, 1):
            f.write("=" * 80 + "\n")
            f.write(f"CHUNK {i}/{len(chunks)}\n")
            f.write("=" * 80 + "\n")
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
        embedder = SingleEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device,
                                  batch_size=batch_size)
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


# ==================== CHATBOT WITH SMART RETRIEVAL ====================
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

    def ask(self, q, k=5):
        """Tìm 5 nguồn, LLM chọn 2 tốt nhất, rồi trả lời ngắn gọn"""
        if not self.retriever:
            return {"answer": "Bot chưa sẵn sàng", "sources": [], "fallback_available": False}

        try:
            # Bước 1: Retrieve top 5
            results = self.retriever.search(q, top_k=k)
            if not results:
                return {
                    "answer": "Không tìm thấy thông tin trong sách",
                    "sources": [],
                    "fallback_available": True
                }

            # Bước 2: LLM re-ranking → chọn 2 nguồn tốt nhất
            best_sources = self._select_best_sources(q, results, top_n=2)

            # Bước 3: Tạo context ngắn gọn
            ctx = "\n\n".join([
                f"[NGUỒN {i+1} - Trang {r.get('page_num', '?')}]\n{r['text']}"
                for i, r in enumerate(best_sources)
            ])

            # Bước 4: Trả lời ngắn gọn
            ans = self._gemini_concise_answer(q, ctx)

            # ✨ Kiểm tra nếu sách không đề cập
            if "không đề cập" in ans.lower() or "không có trong sách" in ans.lower():
                return {
                    "answer": ans,
                    "sources": best_sources,
                    "fallback_available": True
                }

            return {"answer": ans, "sources": best_sources, "fallback_available": False}

        except Exception as e:
            st.error(f"Lỗi trong ask(): {e}")
            return {
                "answer": f"❌ Lỗi xử lý: {str(e)}",
                "sources": [],
                "fallback_available": False
            }

    def _select_best_sources(self, q, results, top_n=2):
        """LLM đánh giá và chọn nguồn phù hợp nhất"""
        try:
            sources_text = "\n\n".join([
                f"NGUỒN {i+1} (Score: {r['score']:.3f}, Trang {r.get('page_num', '?')}):\n{r['text'][:500]}"
                for i, r in enumerate(results[:5])
            ])

            prompt = f"""Phân tích nguồn nào TRỰC TIẾP trả lời câu hỏi sau:

CÂU HỎI: {q}

CÁC NGUỒN:
{sources_text}

YÊU CẦU:
- Chọn {top_n} nguồn PHÙ HỢP NHẤT
- Ưu tiên nguồn có thông tin CỤ THỂ, CHÍNH XÁC
- Bỏ qua nguồn chung chung, không liên quan

TRẢ LỜI JSON (chỉ số nguồn, bắt đầu từ 1):
{{
  "selected": [1, 3],
  "reason": "Nguồn 1 có định nghĩa rõ ràng, Nguồn 3 có ví dụ minh họa"
}}"""

            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
            headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 512}
            }

            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code == 200:
                result_text = resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                # Parse JSON từ response
                result_text = result_text.replace("```json", "").replace("```", "").strip()
                selected_data = json.loads(result_text)
                selected_indices = selected_data.get("selected", [1, 2])
                # Chuyển từ 1-indexed sang 0-indexed
                selected = [results[i-1] for i in selected_indices if 0 < i <= len(results)]
                if len(selected) >= top_n:
                    return selected[:top_n]
        except Exception as e:
            st.warning(f"")

        # Fallback: lấy top_n nguồn có score cao nhất
        return results[:top_n]

    def _gemini_general_answer(self, q):
        """Trả lời từ kiến thức chung (không dựa vào sách)"""
        prompt = f"""Bạn là trợ lý AI thông minh, trả lời câu hỏi dựa trên kiến thức chung.

YÊU CẦU:
1. TRẢ LỜI NGẮN GỌN, DỄ HIỂU (150-200 từ)
2. CẤU TRÚC: Định nghĩa → Giải thích → Ví dụ (nếu có)
3. Không cần trích dẫn nguồn (vì không có sách)
4. Nếu là bài toán → giải chi tiết từng bước

CÂU HỎI: {q}

TRẢ LỜI:"""

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024,
                "topP": 0.9,
                "topK": 40
            }
        }

        try:
            with st.spinner("🧠 MIBook BOT đang suy nghĩ từ kiến thức chung..."):
                resp = requests.post(url, headers=headers, json=data, timeout=60)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                return f"❌ Không thể trả lời (Status: {resp.status_code})"
        except Exception as e:
            return f"❌ Lỗi: {str(e)}"

    def _gemini_concise_answer(self, q, ctx):
        """Trả lời NGẮN GỌN, TẬP TRUNG + GHI TRANG"""
        prompt = f"""Bạn là trợ lý AI, trả lời NGẮN GỌN dựa trên sách giáo khoa.

NGUYÊN TẮC:
1. ĐI THẲNG VÀO VẤN ĐỀ - không dài dòng, không luyên thuyên
2. CHỈ DÙNG THÔNG TIN TỪ 2 NGUỒN DƯỚI ĐÂY
3. TRẢ LỜI TỐI ĐA 150-200 TỪ (khoảng 4-6 câu)
4. CẤU TRÚC: Định nghĩa/Khái niệm → Giải thích ngắn → Ví dụ (nếu có)
5. **BẮT BUỘC GHI TRANG**: Mỗi khi dùng thông tin từ nguồn nào, ghi **(Trang X)** ngay sau câu đó
6. **QUAN TRỌNG**: Nếu 2 nguồn KHÔNG ĐỦ thông tin để trả lời đầy đủ câu hỏi → Nói rõ "Sách không đề cập chi tiết về [vấn đề]" rồi DỪNG LẠI
7. KHÔNG ĐƯỢC lặp lại câu hỏi, KHÔNG ĐƯỢC nói "Theo như sách...", "Dựa vào..."
8. KHÔNG ĐƯỢC SUY LUẬN hay TRẢ LỜI NGOÀI nội dung có sẵn trong 2 nguồn

VÍ DỤ GHI TRANG:
- "Quang hợp là quá trình tạo chất hữu cơ từ CO₂ và H₂O **(Trang 45)**."
- "Enzim giúp tăng tốc độ phản ứng hóa học **(Trang 12)**."

VÍ DỤ KHI THIẾU THÔNG TIN:
- "Sách không đề cập chi tiết về số tự nhiên lớn nhất có ba chữ số. Tuy nhiên, sách có đề cập đến kí hiệu số tự nhiên có ba chữ số là abc = a . 100 + b . 10 + c với a ≠ 0 **(Trang 9)**."

NGUỒN TỪ SÁCH:
{ctx}

CÂU HỎI: {q}

TRẢ LỜI NGẮN GỌN (150-200 từ) VÀ GHI RÕ TRANG:"""

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
                "topP": 0.8,
                "topK": 40
            }
        }

        for attempt in range(1, 3):
            try:
                with st.spinner(f"🤖 MIBook BOT đang suy nghĩ... ({attempt}/2)"):
                    resp = requests.post(url, headers=headers, json=data, timeout=60)
                if resp.status_code == 200:
                    answer = resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()

                    # ✨ Kiểm tra xem có ghi trang không
                    if not re.search(r'\(Trang \d+\)', answer) and "không đề cập" not in answer.lower():
                        st.warning("⚠️ Câu trả lời thiếu trích dẫn trang")

                    # Kiểm tra nếu câu trả lời quá dài
                    if len(answer) > 800:
                        st.warning("⚠️ Câu trả lời hơi dài, bạn có thể hỏi cụ thể hơn")
                    return answer
                elif resp.status_code == 429:
                    st.warning("⏳ API quota hết, đợi 2s...")
                    time.sleep(2)
                else:
                    if attempt == 2:
                        return f"❌ Lỗi API (Status: {resp.status_code})"
            except Exception as e:
                if attempt == 2:
                    return f"❌ Không phản hồi: {str(e)}"
                time.sleep(2)

        return "❌ Không thể trả lời sau 2 lần thử"


# ==================== MAIN UI ====================
def main():
    # === HEADER ===
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='title'>🤖 MIBook BOT</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#94A3B8; font-size:1.1rem;'>AI-powered textbook assistant with smart retrieval</p>",
                    unsafe_allow_html=True)

    # === SIDEBAR - PIPELINE ===
    with st.sidebar:
        st.markdown("## 🔧 Pipeline Selection")

        if "pipeline" not in st.session_state:
            st.session_state.pipeline = "balanced"

        col1, col2, col3 = st.columns(3)
        for idx, (key, cfg) in enumerate(PIPELINES.items()):
            with [col1, col2, col3][idx]:
                if st.button(
                        f"**{cfg['name']}**\n_{cfg['desc']}_",
                        key=key,
                        use_container_width=True,
                        type="primary" if st.session_state.pipeline == key else "secondary"
                ):
                    st.session_state.pipeline = key
                st.markdown(
                    f"<div style='text-align:center; margin-top:8px;'><i class='fa-solid fa-{cfg['icon']}' style='color:{cfg['color']}; font-size:1.8rem;'></i></div>",
                    unsafe_allow_html=True
                )

        cfg = PIPELINES[st.session_state.pipeline]
        st.markdown("---")
        st.markdown(
            f"### <i class='fa-solid fa-{cfg['icon']}' style='color:{cfg['color']}; margin-right:8px;'></i> **{cfg['name']}**",
            unsafe_allow_html=True)
        st.markdown(f"*{cfg['desc']}*")

        saved = DenseRetriever.exists(cfg['path'])
        status_icon = "check-circle" if saved else "circle-xmark"
        status_color = "#10B981" if saved else "#EF4444"
        st.markdown(
            f"**Trạng thái:** <i class='fa-solid fa-{status_icon}' style='color:{status_color}; margin-right:6px;'></i> `{'Đã lưu' if saved else 'Chưa xử lý'}`",
            unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 Smart Features")
        st.markdown("""
        - 🔍 **Re-ranking**: LLM chọn 2 nguồn tốt nhất
        - ⚡ **Concise Answer**: 150-200 từ
        - 🎯 **Focused**: Đi thẳng vào vấn đề
        - 🧠 **Fallback**: Kiến thức chung khi cần
        """)

    # === TABS ===
    tab1, tab2 = st.tabs(["📄 Xử lý PDF", "💬 Chat"])

    # === TAB 1: XỬ LÝ ===
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### 📤 Tải lên & Xử lý")
            if saved:
                if st.button("🔄 Load Retriever", type="primary", use_container_width=True):
                    with st.spinner("Đang tải..."):
                        bot = Chatbot(cfg)
                        embedder = SingleEmbedder(model_name=cfg['model'],
                                                  device="cuda" if torch.cuda.is_available() else "cpu")
                        bot.load(embedder)
                        st.session_state.bot = bot
                        st.session_state.ready = True
                        st.success("✅ Đã tải!")
            skip_file = project_root / "output" / "after_llm.txt"
            if skip_file.exists():
                if st.button("⏭️ SKIP OCR+LLM", type="secondary", use_container_width=True):
                    with st.spinner("Chuẩn bị..."):
                        after = skip_file
                        bot = Chatbot(cfg)
                        st.session_state.bot = bot
                        prog = st.progress(0)
                        stat = st.empty()
                        try:
                            stat.text("Chunking + Embedding...")
                            chunks, embs, embedder = chunk_and_embed(after, cfg)
                            prog.progress(0.7)
                            stat.text("Lưu...")
                            bot.build(chunks, embs, embedder)
                            prog.progress(1.0)
                            st.session_state.ready = True
                            st.success("✅ HOÀN THÀNH!")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Chunks", len(chunks))
                            c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
                            c3.metric("Saved", "✅")
                        except Exception as e:
                            st.error(f"❌ Lỗi: {e}")
            uploaded = st.file_uploader("📁 Upload PDF", type=['pdf'], label_visibility="collapsed")

        with col2:
            if uploaded:
                if st.button("🚀 Process Full Pipeline", type="primary", use_container_width=True):
                    bot = Chatbot(cfg)
                    st.session_state.bot = bot
                    prog = st.progress(0)
                    stat = st.empty()
                    try:
                        stat.text("OCR + LLM...")
                        after = CorePipeline.run(uploaded, lambda m, p: (stat.text(m), prog.progress(p * 0.4)))
                        prog.progress(0.4)
                        stat.text("Chunking + Embedding...")
                        chunks, embs, embedder = chunk_and_embed(after, cfg)
                        prog.progress(0.7)
                        stat.text("Lưu...")
                        bot.build(chunks, embs, embedder)
                        prog.progress(1.0)
                        st.session_state.ready = True
                        st.success("✅ HOÀN THÀNH!")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("📦 Chunks", len(chunks))
                        c2.metric("💻 Device", "GPU" if torch.cuda.is_available() else "CPU")
                        c3.metric("💾 Saved", "✅")
                    except Exception as e:
                        st.error(f"❌ Lỗi: {e}")

    # === TAB 2: CHAT ===
    with tab2:
        if not st.session_state.get('ready', False):
            st.warning("⚠️ Vui lòng xử lý PDF trước ở tab 'Xử lý PDF'")
            st.stop()

        # Initialize session state
        for k in ['bot', 'msg', 'ready']:
            if k not in st.session_state:
                st.session_state[k] = None if k == 'bot' else ([] if k == 'msg' else False)

        # Display chat history
        for idx, m in enumerate(st.session_state.msg):
            with st.chat_message(m['role'], avatar="👤" if m['role'] == "user" else "🤖"):
                st.markdown(m['content'])

                # ✨ Nút "Trả lời từ kiến thức chung" nếu có fallback
                if m['role'] == 'assistant' and m.get('fallback_available'):
                    fallback_key = f"fallback_{idx}"
                    if fallback_key not in st.session_state:
                        st.session_state[fallback_key] = False

                    if not st.session_state[fallback_key]:
                        if st.button("🧠 Bạn muốn MIBook BOT trả lời từ kiến thức chung?", key=f"btn_{fallback_key}"):
                            st.session_state[fallback_key] = True
                            # Lấy câu hỏi gốc (message trước đó)
                            if idx > 0:
                                original_question = st.session_state.msg[idx-1]['content']
                                general_ans = st.session_state.bot._gemini_general_answer(original_question)

                                # Thêm vào lịch sử
                                st.session_state.msg.append({
                                    "role": "assistant",
                                    "content": f"**📚 Trả lời từ kiến thức chung:**\n\n{general_ans}",
                                    "src": [],
                                    "fallback_available": False
                                })
                            st.rerun()
                    else:
                        st.info("✅ Đã trả lời từ kiến thức chung bên dưới")

                # Hiển thị nguồn
                if m.get('src'):
                    with st.expander("📚 Nguồn tham khảo (2 đoạn trích được chọn)", expanded=False):
                        for i, s in enumerate(m['src']):
                            st.markdown(f"""
                            <div class='source-box'>
                                <strong>📄 Nguồn {i + 1}</strong> | Trang {s.get('page_num', '?')} | Score: {s['score']:.3f}<br>
                                <small>{s['text'][:400]}{'...' if len(s['text']) > 400 else ''}</small>
                            </div>
                            """, unsafe_allow_html=True)

        # Chat input
        if q := st.chat_input("💬 Hỏi gì về sách..."):
            st.session_state.msg.append({"role": "user", "content": q})
            with st.chat_message("user", avatar="👤"):
                st.markdown(q)

            with st.chat_message("assistant", avatar="🤖"):
                res = st.session_state.bot.ask(q)
                st.markdown(res['answer'])
                st.session_state.msg.append({
                    "role": "assistant",
                    "content": res['answer'],
                    "src": res.get('sources', []),
                    "fallback_available": res.get('fallback_available', False)
                })
            st.rerun()  # Rerun để hiển thị nút fallback

        # Clear history button
        if st.button("🗑️ Xóa lịch sử chat", type="secondary"):
            st.session_state.msg = []
            # Clear fallback states
            keys_to_delete = [k for k in st.session_state.keys() if k.startswith('fallback_')]
            for k in keys_to_delete:
                del st.session_state[k]
            st.rerun()


if __name__ == "__main__":
    main()