"""
test_batch_questions.py - Script test hàng loạt 50 câu hỏi
Chạy độc lập, không cần UI, xuất kết quả ra Excel
"""

import pandas as pd
import torch
import time
import json
from pathlib import Path
from datetime import datetime
import sys
import os
import re
import requests
from dotenv import load_dotenv

# === SETUP PATH ===
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from embedders.single_embedder import SingleEmbedder
from embedders.vietnamese_embedder import VietnameseEmbedder
from retrievers.dense_retriever import DenseRetriever

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Thiếu GEMINI_API_KEY trong .env!")
    sys.exit(1)

# ==================== CONFIG ====================
PIPELINES = {
    "fast": {
        "name": "Fast (Recursive + MiniLM)",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "path": "output/retrievers/fast",
    },
    "balanced": {
        "name": "Balanced (Page-Aware + GTE-Viet)",
        "model": "Alibaba-NLP/gte-multilingual-base",
        "path": "output/retrievers/balanced",
    },
    "quality": {
        "name": "Quality (Semantic + BGE-M3)",
        "model": "BAAI/bge-m3",
        "path": "output/retrievers/quality",
    }
}

# 50 câu hỏi test
TEST_QUESTIONS = [
    "Giao của hai tập hợp là gì?",
    "Kí hiệu nào dùng để chỉ thuộc tập hợp?",
    "Tập hợp N ký hiệu cho tập hợp nào?",
    "Hai số tự nhiên liên tiếp hơn kém nhau bao nhiêu đơn vị?",
    "ab được hiểu là số có mấy chữ số và ý nghĩa?",
    "Số tự nhiên lớn nhất có ba chữ số là bao nhiêu?",
    "Số 9 được viết trong hệ La Mã là gì?",
    "Số nguyên tố chẵn duy nhất là số nào?",
    "Tập hợp rỗng được ký hiệu là gì?",
    "Nếu mọi phần tử của tập hợp A đều thuộc tập hợp B thì kết luận gì?",
    "Nếu A ⊂ B và B ⊂ A thì kết luận gì?",
    "Tập D = {0} có bao nhiêu phần tử?",
    "Phép nhân hai số tự nhiên cho ta kết quả là gì?",
    "Tính chất giao hoán của phép cộng phát biểu thế nào?",
    "Tính chất phân phối của phép nhân đối với phép cộng là gì?",
    "Tích của một số với 0 bằng bao nhiêu?",
    "Trong phép chia cho 2, số dư có thể là những số nào?",
    "Nhân hai luỹ thừa cùng cơ số ta làm gì?",
    "Nếu a chia hết cho m và b chia hết cho m thì a + b có đặc điểm gì?",
    "Nếu a ⋮ b và b ⋮ c thì suy ra điều gì?",
    "Nếu a không chia hết cho m và b chia hết cho m thì a + b có chia hết cho m không?",
    "Số có chữ số tận cùng chẵn thì có tính chất gì?",
    "Số có chữ số tận cùng là 0 hoặc 5 thì sao?",
    "Trong các số 652, 850, 1546, 785, 6321 thì số nào chia hết cho 2?",
    "Số có chữ số tận cùng 4 chia hết cho 2 đúng hay sai?",
    "Số tự nhiên nhỏ nhất là số nào?",
    "Lũy thừa bậc n của a là gì?",
    "Tập hợp các số tự nhiên được kí hiệu là gì?",
    "Điều kiện để có phép trừ a – b trong tập số tự nhiên là gì?",
    "Tính chất chia hết của một tổng là gì?",
    "Dấu hiệu chia hết cho 3 là gì?",
    "Số nguyên tố là gì?",
    "Hợp số là gì?",
    "Ước chung của hai hay nhiều số là gì?",
    "Bội chung của hai hay nhiều số là gì?",
    "Cách tìm ƯCLN bằng cách phân tích ra thừa số nguyên tố?",
    "Cách tìm BCNN bằng cách phân tích ra thừa số nguyên tố?",
    "Điểm là gì?",
    "Đường thẳng là gì?",
    "Ba điểm thẳng hàng là gì?",
    "Điểm nằm giữa hai điểm là gì?",
    "Tia là gì?",
    "Đoạn thẳng AB là gì?",
    "Khi nào thì AM + MB = AB?",
    "Tập hợp các số nguyên được kí hiệu là gì?",
    "Quy tắc cộng hai số nguyên cùng dấu?",
    "Khi nào thì tích hai số nguyên là số dương?",
    "Tổng của hai số nguyên đối nhau bằng bao nhiêu?",
]


# ==================== CHATBOT CLASS ====================
class Chatbot:
    def __init__(self, cfg):
        self.cfg = cfg
        self.retriever = None

    def load(self, embedder):
        if not DenseRetriever.exists(self.cfg['path']):
            raise FileNotFoundError(f"❌ Retriever not found: {self.cfg['path']}")
        self.retriever = DenseRetriever.load(self.cfg['path'], embedder)
        print(f"✅ Loaded: {self.cfg['name']}")

    def ask(self, q, k=5):
        if not self.retriever:
            return "Bot chưa sẵn sàng"

        try:
            results = self.retriever.search(q, top_k=k)
            if not results:
                return "Không tìm thấy thông tin trong sách"

            best_sources = self._select_best_sources(q, results, top_n=2)

            ctx = "\n\n".join([
                f"[NGUỒN {i + 1} - Trang {r.get('page_num', '?')}]\n{r['text']}"
                for i, r in enumerate(best_sources)
            ])

            return self._gemini_concise_answer(q, ctx)

        except Exception as e:
            print(f"❌ Lỗi: {e}")
            return f"Lỗi: {str(e)}"

    def _select_best_sources(self, q, results, top_n=2):
        try:
            sources_text = "\n\n".join([
                f"NGUỒN {i + 1} (Score: {r['score']:.3f}, Trang {r.get('page_num', '?')}):\n{r['text'][:500]}"
                for i, r in enumerate(results[:5])
            ])

            prompt = f"""Phân tích nguồn nào TRỰC TIẾP trả lời câu hỏi sau:

CÂU HỎI: {q}

CÁC NGUỒN:
{sources_text}

YÊU CẦU:
- Chọn {top_n} nguồn PHÙ HỢP NHẤT
- Ưu tiên nguồn có thông tin CỤ THỂ, CHÍNH XÁC

TRẢ LỜI JSON:
{{
  "selected": [1, 3]
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
                result_text = result_text.replace("```json", "").replace("```", "").strip()
                selected_data = json.loads(result_text)
                selected_indices = selected_data.get("selected", [1, 2])
                selected = [results[i - 1] for i in selected_indices if 0 < i <= len(results)]
                if len(selected) >= top_n:
                    return selected[:top_n]
        except:
            pass

        return results[:top_n]

    def _gemini_concise_answer(self, q, ctx):
        prompt = f"""Bạn là trợ lý AI, trả lời NGẮN GỌN dựa trên sách giáo khoa.

NGUYÊN TẮC:
1. ĐI THẲNG VÀO VẤN ĐỀ - không dài dòng
2. CHỈ DÙNG THÔNG TIN TỪ 2 NGUỒN DƯỚI ĐÂY
3. TRẢ LỜI TỐI ĐA 150-200 TỪ
4. **BẮT BUỘC GHI TRANG**: Mỗi khi dùng thông tin, ghi **(Trang X)**
5. Nếu không đủ thông tin → "Sách không đề cập chi tiết"

NGUỒN TỪ SÁCH:
{ctx}

CÂU HỎI: {q}

TRẢ LỜI NGẮN GỌN:"""

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024}
        }

        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=data, timeout=60)
                if resp.status_code == 200:
                    return resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                elif resp.status_code == 429:
                    print("⏳ Quota hết, đợi 2s...")
                    time.sleep(2)
            except Exception as e:
                if attempt == 1:
                    return f"Lỗi API: {str(e)}"
                time.sleep(2)

        return "Không thể trả lời"


# ==================== MAIN TEST ====================
def main():
    print("=" * 80)
    print("🚀 BẮT ĐẦU TEST HÀNG LOẠT 50 CÂU HỎI")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Device: {device.upper()}")

    results = {
        "STT": [],
        "Câu hỏi": [],
    }

    # Test từng pipeline
    for pipeline_key, pipeline_cfg in PIPELINES.items():
        print(f"\n{'=' * 80}")
        print(f"📊 TESTING: {pipeline_cfg['name']}")
        print(f"{'=' * 80}")

        # Load embedder
        if "minilm" in pipeline_cfg['model'].lower():
            embedder = SingleEmbedder(model_name=pipeline_cfg['model'], device=device)
        elif "gte" in pipeline_cfg['model'].lower():
            embedder = VietnameseEmbedder(model_key="gte-viet", device=device)
        else:
            embedder = SingleEmbedder(model_name=pipeline_cfg['model'], device=device)

        # Load chatbot
        bot = Chatbot(pipeline_cfg)
        bot.load(embedder)

        # Test 50 câu hỏi
        answers = []
        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"  [{i}/50] {question[:50]}...")
            answer = bot.ask(question)
            answers.append(answer)
            time.sleep(0.5)  # Tránh bị rate limit

        results[f"Trả lời - {pipeline_cfg['name']}"] = answers
        print(f"✅ Hoàn thành {pipeline_cfg['name']}")

    # Tạo DataFrame
    results["STT"] = list(range(1, 51))
    results["Câu hỏi"] = TEST_QUESTIONS

    df = pd.DataFrame(results)

    # Sắp xếp lại thứ tự cột
    cols = ["STT", "Câu hỏi"]
    for pipeline_cfg in PIPELINES.values():
        cols.append(f"Trả lời - {pipeline_cfg['name']}")
    df = df[cols]

    # Xuất Excel
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    df.to_excel(output_file, index=False, engine='openpyxl')

    print(f"\n{'=' * 80}")
    print(f"✅ HOÀN THÀNH!")
    print(f"📁 Kết quả đã lưu: {output_file}")
    print(f"📊 Tổng số câu hỏi: {len(TEST_QUESTIONS)}")
    print(f"📊 Số pipeline: {len(PIPELINES)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()