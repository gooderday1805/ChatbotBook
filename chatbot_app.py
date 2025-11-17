# """
# app_chatbot_modular.py - FINAL 100% - GEMINI FULL ANSWER (VN - 16/11/2025)
# - Output dài, chi tiết, đầy đủ
# - Đưa cả 3 đoạn top retrieval vào prompt
# - Bắt buộc: DỰA VÀO SÁCH, KHÔNG BỊA
# - Dựa 100% trên curl: X-goog-api-key + gemini-2.0-flash
# - Retry 2 lần, delay 2s → không 429
# - SKIP OCR+LLM + GPU + chunk.txt + metadata.json
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
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#
# if not GEMINI_API_KEY:
#     st.error("Thiếu GEMINI_API_KEY trong .env!")
#     st.stop()
#
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
#     },
#     "balanced": {
#         "name": "Balanced",
#         "desc": "Page-Aware + GTE-Viet",
#         "chunker": PageAwareChunker,
#         "params": {"max_chars": 2000},
#         "embed": "gte-viet",
#         "model": "Alibaba-NLP/gte-multilingual-base",
#         "path": "output/retrievers/balanced",
#     },
#     "quality": {
#         "name": "Quality",
#         "desc": "Semantic + BGE-M3",
#         "chunker": SemanticChunker,
#         "params": {"detect_headings": True},
#         "embed": "bge-m3",
#         "model": "BAAI/bge-m3",
#         "path": "output/retrievers/quality",
#     }
# }
#
#
# # ==================== PARSER ====================
# def parse_pages_correctly(file_path):
#     content = Path(file_path).read_text(encoding='utf-8')
#     separator = '=' * 80
#     blocks = content.split(separator)
#     pages = []
#     current_page_num = None
#
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
# # ==================== SAVE CHUNKS ====================
# def save_chunks_to_file(chunks, strategy_name):
#     filepath = project_root / "output" / f"chunks_{strategy_name}.txt"
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write("="*80 + "\n")
#         f.write(f"CHUNKS - {strategy_name.upper()}\n")
#         f.write(f"="*80 + "\n\n")
#         f.write(f"Total chunks: {len(chunks)}\n\n")
#         sizes = [len(c['text']) for c in chunks]
#         f.write(f"Average size: {sum(sizes)/len(sizes):.0f} chars\n")
#         f.write(f"Min size: {min(sizes)} chars\n")
#         f.write(f"Max size: {max(sizes)} chars\n\n")
#         f.write("="*80 + "\n\n")
#         for i, c in enumerate(chunks, 1):
#             f.write("="*80 + "\n")
#             f.write(f"CHUNK {i}/{len(chunks)}\n")
#             f.write("="*80 + "\n")
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
#
#         if callback: callback("OCR + Extract...", 0.3)
#         OCRProcessor().process_pdf(str(pdf_path))
#
#         if callback: callback("LLM Cleaning...", 0.6)
#         OptimalLLMCleaner().process()
#
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
#
#     chunker = config['chunker'](**config['params'])
#     chunks = chunker.chunk_document(pages)
#     if not chunks:
#         raise ValueError("Không tạo được chunk!")
#
#     chunk_file = save_chunks_to_file(chunks, config['embed'])
#     st.success(f"Đã tạo {len(chunks)} chunks → {chunk_file.name}")
#
#     has_gpu = torch.cuda.is_available()
#     device = "cuda" if has_gpu else "cpu"
#     batch_size_map = {"bge-m3": 32, "gte-viet": 48, "minilm": 64}
#     batch_size = batch_size_map.get(config['embed'], 32)
#     if not has_gpu:
#         batch_size = max(8, batch_size // 2)
#
#     if config['embed'] == "bge-m3":
#         embedder = SingleEmbedder(model_name="BAAI/bge-m3", device=device, batch_size=batch_size)
#     elif config['embed'] == "gte-viet":
#         embedder = VietnameseEmbedder(model_key="gte-viet", device=device, batch_size=batch_size)
#     else:
#         embedder = SingleEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device, batch_size=batch_size)
#
#     start = time.time()
#     embeddings = embedder.encode_chunks(chunks, text_key='text')
#     elapsed = time.time() - start
#
#     if len(embeddings) == 0:
#         raise ValueError("Embeddings rỗng!")
#
#     emb_dir = project_root / "output" / "embeddings"
#     emb_dir.mkdir(exist_ok=True, parents=True)
#
#     emb_file = emb_dir / f"{config['embed']}_embeddings.npy"
#     embedder.save_embeddings(embeddings, str(emb_file))
#
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
#
#     st.info(f"Embedding: {embeddings.shape} | {elapsed:.1f}s | {device.upper()}")
#
#     return chunks, embeddings, embedder
#
#
# # ==================== CHATBOT – FULL ANSWER + 3 SOURCES ====================
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
#
#         # ĐƯA CẢ 3 ĐOẠN TOP VÀO PROMPT
#         ctx = "\n\n".join([
#             f"--- NGUỒN {i+1} (Trang {r.get('page_num','?')} | Độ tương đồng: {r['score']:.3f}) ---\n{r['text']}"
#             for i, r in enumerate(results)
#         ])
#
#         ans = self._gemini_full_answer(q, ctx)
#         return {"answer": ans, "sources": results}
#
#     def _gemini_full_answer(self, q, ctx):
#         """
#         BẮT BUỘC: DỰA VÀO SÁCH, KHÔNG BỊA, TRẢ LỜI CHI TIẾT
#         Dựa 100% trên curl
#         """
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
#
#         url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
#
#         headers = {
#             "Content-Type": "application/json",
#             "X-goog-api-key": GEMINI_API_KEY
#         }
#
#         data = {
#             "contents": [{"parts": [{"text": prompt}]}],
#             "generationConfig": {
#                 "temperature": 0.3,
#                 "maxOutputTokens": 2048  # TĂNG LÊN ĐỂ TRẢ LỜI DÀI
#             }
#         }
#
#         for attempt in range(1, 3):
#             try:
#                 with st.spinner(f"Gemini đang trả lời chi tiết (lần {attempt}/2)..."):
#                     resp = requests.post(url, headers=headers, json=data, timeout=60)
#
#                 if resp.status_code == 200:
#                     try:
#                         answer = resp.json()['candidates'][0]['content']['parts'][0]['text']
#                         return answer.strip()
#                     except:
#                         st.error("Gemini JSON lỗi")
#                         return "Lỗi phân tích phản hồi"
#
#                 elif resp.status_code == 429:
#                     st.warning("Gemini 429: Đợi 2s...")
#                     time.sleep(2)
#                     continue
#                 else:
#                     st.error(f"Gemini lỗi {resp.status_code}")
#                     if attempt == 2:
#                         return "Lỗi server"
#
#             except requests.exceptions.Timeout:
#                 st.warning("Timeout. Thử lại...")
#                 time.sleep(2)
#             except Exception as e:
#                 st.error(f"Lỗi: {str(e)[:60]}")
#                 if attempt == 2:
#                     return "Gemini không phản hồi"
#
#         return "Không thể lấy câu trả lời sau 2 lần thử"
#
#
# # ==================== UI ====================
# def main():
#     st.set_page_config(page_title="RAG Chatbot", layout="wide", page_icon="Robot")
#
#     st.sidebar.header("Pipeline")
#     mode = st.sidebar.radio("Mode", list(PIPELINES.keys()),
#                             format_func=lambda x: f"{PIPELINES[x]['name']} - {PIPELINES[x]['desc']}",
#                             index=1)
#     cfg = PIPELINES[mode]
#
#     st.title(f"RAG Chatbot - {cfg['name']}")
#     st.markdown(f"*{cfg['desc']}*")
#
#     saved = DenseRetriever.exists(cfg['path'])
#     st.info(f"**Status:** {'Đã lưu' if saved else 'Chưa xử lý'}")
#
#     for k in ['bot', 'msg', 'ready']:
#         if k not in st.session_state:
#             st.session_state[k] = None if k == 'bot' else ([] if k == 'msg' else False)
#
#     t1, t2 = st.tabs(["Xử lý / Load", "Chat"])
#
#     with t1:
#         col1, col2 = st.columns([1, 2])
#         with col1:
#             if saved and st.button("Load Retriever", type="primary"):
#                 with st.spinner("Loading..."):
#                     bot = Chatbot(cfg)
#                     embedder = SingleEmbedder(model_name=cfg['model'],
#                                               device="cuda" if torch.cuda.is_available() else "cpu")
#                     bot.load(embedder)
#                     st.session_state.bot = bot
#                     st.session_state.ready = True
#                     st.success("Loaded!")
#
#             skip_file = project_root / "output" / "after_llm.txt"
#             if skip_file.exists() and st.button("SKIP OCR+LLM (Dùng file có sẵn)", type="secondary"):
#                 with st.spinner("Đang dùng after_llm.txt..."):
#                     after = skip_file
#                     bot = Chatbot(cfg)
#                     st.session_state.bot = bot
#                     prog = st.progress(0)
#                     stat = st.empty()
#
#                     try:
#                         stat.text("2. Chunking + Embedding...")
#                         chunks, embs, embedder = chunk_and_embed(after, cfg)
#                         prog.progress(0.7)
#
#                         stat.text("3. Saving...")
#                         bot.build(chunks, embs, embedder)
#                         prog.progress(1.0)
#                         st.session_state.ready = True
#                         st.success("HOÀN THÀNH (SKIP OCR+LLM)!")
#
#                         c1, c2, c3 = st.columns(3)
#                         c1.metric("Chunks", len(chunks))
#                         c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
#                         c3.metric("Saved", "Yes")
#
#                     except Exception as e:
#                         st.error(f"Lỗi: {e}")
#                         st.exception(e)
#
#             uploaded = st.file_uploader("Upload PDF", type=['pdf'])
#
#         if uploaded and st.button("Process Full Pipeline", type="secondary"):
#             bot = Chatbot(cfg)
#             st.session_state.bot = bot
#             prog = st.progress(0)
#             stat = st.empty()
#
#             try:
#                 stat.text("1. OCR + LLM...")
#                 after = CorePipeline.run(uploaded, lambda m, p: (stat.text(m), prog.progress(p * 0.4)))
#                 prog.progress(0.4)
#
#                 stat.text("2. Chunking + Embedding...")
#                 chunks, embs, embedder = chunk_and_embed(after, cfg)
#                 prog.progress(0.7)
#
#                 stat.text("3. Saving...")
#                 bot.build(chunks, embs, embedder)
#                 prog.progress(1.0)
#                 st.session_state.ready = True
#                 st.success("HOÀN THÀNH!")
#
#                 c1, c2, c3 = st.columns(3)
#                 c1.metric("Chunks", len(chunks))
#                 c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
#                 c3.metric("Saved", "Yes")
#
#                 col_a, col_b = st.columns(2)
#                 with col_a:
#                     if st.button("Xem chunk.txt"):
#                         f = project_root / "output" / f"chunks_{cfg['embed']}.txt"
#                         if f.exists():
#                             st.code(f.read_text(encoding='utf-8')[:2000])
#                 with col_b:
#                     if st.button("Xem metadata.json"):
#                         f = project_root / "output" / "embeddings" / f"{cfg['embed']}_metadata.json"
#                         if f.exists():
#                             st.json(json.loads(f.read_text(encoding='utf-8')))
#
#             except Exception as e:
#                 st.error(f"Lỗi: {e}")
#                 st.exception(e)
#
#     with t2:
#         if not st.session_state.ready:
#             st.warning("Vui lòng xử lý trước")
#             st.stop()
#
#         for m in st.session_state.msg:
#             with st.chat_message(m['role']):
#                 st.markdown(m['content'])
#                 if m.get('src'):
#                     with st.expander("Nguồn (3 đoạn trích)"):
#                         for s in m['src']:
#                             st.caption(f"Trang {s.get('page_num','?')} | Score: {s['score']:.3f}")
#                             st.text(s['text'][:300] + ("..." if len(s['text']) > 300 else ""))
#
#         if q := st.chat_input("Hỏi nội dung về sách..."):
#             st.session_state.msg.append({"role": "user", "content": q})
#             with st.chat_message("user"): st.markdown(q)
#             with st.chat_message("assistant"):
#                 res = st.session_state.bot.ask(q)
#                 st.markdown(res['answer'])
#                 st.session_state.msg.append({
#                     "role": "assistant",
#                     "content": res['answer'],
#                     "src": res['sources']
#                 })
#
#         if st.button("Xóa"):
#             st.session_state.msg = []
#             st.rerun()
#
#
# if __name__ == "__main__":
#     main()
"""
app_chatbot_modular.py - FINAL 100% - ICON ĐẸP + KHÔNG LỖI (VN - 16/11/2025 08:53 AM)
- Icon hiển thị 100%: bolt, sliders, star
- Sửa lỗi HTML escape → dùng st.button + st.markdown
- UI đẹp, responsive, không crash
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
    page_icon="robot",
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


# ==================== CHATBOT ====================
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
        ctx = "\n\n".join([
            f"--- NGUỒN {i + 1} (Trang {r.get('page_num', '?')} | Độ tương đồng: {r['score']:.3f}) ---\n{r['text']}"
            for i, r in enumerate(results)
        ])
        ans = self._gemini_full_answer(q, ctx)
        return {"answer": ans, "sources": results}

    def _gemini_full_answer(self, q, ctx):
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
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
        }
        for attempt in range(1, 3):
            try:
                with st.spinner(f"MIBook BOT đang suy nghĩ... (lần {attempt}/2)"):
                    resp = requests.post(url, headers=headers, json=data, timeout=60)
                if resp.status_code == 200:
                    return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                elif resp.status_code == 429:
                    st.warning("Quota hết, đợi 2s...")
                    time.sleep(2)
                else:
                    if attempt == 2: return "Lỗi server"
            except:
                if attempt == 2: return "Không phản hồi"
                time.sleep(2)
        return "Không thể trả lời"


# ==================== MAIN UI ====================
def main():
    # === HEADER ===
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='title'>MIBook BOT</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#94A3B8; font-size:1.1rem;'>AI-powered textbook assistant</p>",
                    unsafe_allow_html=True)

    # === SIDEBAR - PIPELINE ===
    with st.sidebar:
        st.markdown("## Pipeline")

        # SỬA LỖI: DÙNG st.button + st.markdown
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

    # === TABS ===
    tab1, tab2 = st.tabs(["Xử lý PDF", "Chat"])

    # === TAB 1: XỬ LÝ ===
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Tải lên & Xử lý")
            if saved:
                if st.button("Load Retriever", type="primary", use_container_width=True):
                    with st.spinner("Đang tải..."):
                        bot = Chatbot(cfg)
                        embedder = SingleEmbedder(model_name=cfg['model'],
                                                  device="cuda" if torch.cuda.is_available() else "cpu")
                        bot.load(embedder)
                        st.session_state.bot = bot
                        st.session_state.ready = True
                        st.success("Đã tải!")
            skip_file = project_root / "output" / "after_llm.txt"
            if skip_file.exists():
                if st.button("SKIP OCR+LLM", type="secondary", use_container_width=True):
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
                            st.success("HOÀN THÀNH!")
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Chunks", len(chunks))
                            c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
                            c3.metric("Saved", "Yes")
                        except Exception as e:
                            st.error(f"Lỗi: {e}")
            uploaded = st.file_uploader("Upload PDF", type=['pdf'], label_visibility="collapsed")
        if uploaded:
            if st.button("Process Full Pipeline", type="primary", use_container_width=True):
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
                    st.success("HOÀN THÀNH!")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Chunks", len(chunks))
                    c2.metric("Device", "GPU" if torch.cuda.is_available() else "CPU")
                    c3.metric("Saved", "Yes")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

    # === TAB 2: CHAT ===
    with tab2:
        if not st.session_state.get('ready', False):
            st.warning("Vui lòng xử lý PDF trước")
            st.stop()
        for k in ['bot', 'msg', 'ready']:
            if k not in st.session_state:
                st.session_state[k] = None if k == 'bot' else ([] if k == 'msg' else False)
        for m in st.session_state.msg:
            with st.chat_message(m['role'], avatar="user" if m['role'] == "user" else "assistant"):
                st.markdown(f"**{m['content']}**")
                if m.get('src'):
                    with st.expander("Nguồn (3 đoạn trích)", expanded=False):
                        for i, s in enumerate(m['src']):
                            st.markdown(f"""
                            <div class='source-box'>
                                <strong>Nguồn {i + 1}</strong> | Trang {s.get('page_num', '?')} | Score: {s['score']:.3f}<br>
                                <small>{s['text'][:400]}{'...' if len(s['text']) > 400 else ''}</small>
                            </div>
                            """, unsafe_allow_html=True)
        if q := st.chat_input("Hỏi gì về sách..."):
            st.session_state.msg.append({"role": "user", "content": q})
            with st.chat_message("user", avatar="user"):
                st.markdown(f"**{q}**")
            with st.chat_message("assistant", avatar="assistant"):
                # with st.spinner("Đang suy nghĩ..."):
                    res = st.session_state.bot.ask(q)
                    st.markdown(res['answer'])
                    st.session_state.msg.append({
                        "role": "assistant",
                        "content": res['answer'],
                        "src": res['sources']
                    })
        if st.button("Xóa lịch sử", type="secondary"):
            st.session_state.msg = []
            st.rerun()


if __name__ == "__main__":
    main()