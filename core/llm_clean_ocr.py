"""
llm_clean_optimal.py - HOÀN HẢO 100%
- Task 1: ĐỌC KEY TỪ .env (AN TOÀN)
- Task 2: GROQ ƯU TIÊN TRƯỚC (tốc độ nhanh)
- Task 3: 10 TRANG/BATCH + XEN KẼ API → KHÔNG BỊ 429
- Fail → chuyển ngay → tối đa 2 lần/API
- Delay 2s → an toàn RPM
- Output đầy đủ → max_tokens 32K
"""

from pathlib import Path
import requests
import time
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv  # pip install python-dotenv

# ================= ĐỌC .env =================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY or not GROQ_API_KEY:
    print("LỖI: Thiếu GEMINI_API_KEY hoặc GROQ_API_KEY trong .env!")
    print("   Tạo file .env:")
    print("   GEMINI_API_KEY=your_gemini_key")
    print("   GROQ_API_KEY=your_groq_key")
    exit(1)
# ==========================================

# ================= CẤU HÌNH =================
PAGES_PER_BATCH = 10                    # 10 trang/batch → ~2k tokens → an toàn
MAX_TOKENS_OUTPUT = 32768               # Output tối đa
MAX_INPUT_TOKENS = 28000                # An toàn context
DELAY_BETWEEN_BATCHES = 2               # Tránh 429 (2s đủ)
MAX_ATTEMPTS_PER_API = 2                # Mỗi API thử tối đa 2 lần
# ==========================================

class OptimalLLMCleaner:
    def __init__(self):
        self.gemini_key = GEMINI_API_KEY
        self.groq_key = GROQ_API_KEY
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"

        self.input_file = Path("output/before_llm.txt")
        self.output_file = Path("output/after_llm.txt")
        self.log_file = Path("output/llm_log.txt")

        Path("output").mkdir(exist_ok=True)
        self.output_file.write_text("", encoding='utf-8')
        self.log_file.write_text("", encoding='utf-8')

        self.stats = {"gemini": 0, "groq": 0, "fallback": 0, "batches": 0}
        self.next_api = "groq"  # ƯU TIÊN GROQ TRƯỚC

    def _log(self, msg: str):
        t = time.strftime("%H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{t}] {msg}\n")

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4 + 100

    def _split_batches(self, pages: List[Dict]) -> List[List[Dict]]:
        return [pages[i:i + PAGES_PER_BATCH] for i in range(0, len(pages), PAGES_PER_BATCH)]

    def _parse_pages(self) -> List[Dict]:
        content = self.input_file.read_text(encoding='utf-8')
        pages = []
        for block in content.split('===PAGE_'):
            if not block.strip(): continue
            parts = block.split('===\n', 1)
            if len(parts) != 2: continue
            try:
                num = int(parts[0].strip())
                text = parts[1].strip()
                pages.append({'page_num': num, 'text': text})
            except: continue
        return sorted(pages, key=lambda x: x['page_num'])

    def process(self):
        print(f"\n{'='*80}")
        print("OPTIMAL DUAL LLM - GROQ ƯU TIÊN - 10 TRANG/BATCH - XEN KẼ API")
        print(f"{'='*80}\n")

        if not self.input_file.exists():
            print("before_llm.txt không tồn tại!")
            return

        pages = self._parse_pages()
        total = len(pages)
        print(f"Total pages: {total}")
        if total == 0: return

        batches = self._split_batches(pages)
        print(f"Split: {len(batches)} batches (10 trang/batch)")

        start = time.time()

        for i, batch in enumerate(batches, 1):
            s, e = batch[0]['page_num'], batch[-1]['page_num']
            combined = "\n".join([f"--- PAGE {p['page_num']} ---\n{p['text']}\n" for p in batch])
            tokens = self._estimate_tokens(combined)

            print(f"\nBATCH {i}/{len(batches)} | PAGES {s}-{e} | ~{tokens:,} tokens")
            print(f"   → Ưu tiên: {self.next_api.upper()}")

            cleaned, ok, api_used = self._clean_with_xen_ke(combined)

            results = self._split_result(cleaned if ok else combined, batch, ok)

            for num, text, status in results:
                api_label = api_used.upper() if status == "cleaned" else "OCR"
                self._write_page(num, text, api_label)
                if status == "cleaned":
                    self.stats[api_used] += 1
                else:
                    self.stats["fallback"] += 1

            self.stats["batches"] += 1

            # XEN KẼ: OK → CHUYỂN API CHO BATCH SAU
            if ok:
                self.next_api = "gemini" if api_used == "groq" else "groq"

            print(f"   Delay {DELAY_BETWEEN_BATCHES}s để API nghỉ...")
            time.sleep(DELAY_BETWEEN_BATCHES)

        total_time = time.time() - start
        print(f"\nHOÀN THÀNH!")
        print(f"   Gemini  : {self.stats['gemini']} pages")
        print(f"   Groq    : {self.stats['groq']} pages")
        print(f"   Fallback: {self.stats['fallback']} pages")
        print(f"   Time    : {total_time:.0f}s (~{total_time/60:.1f} phút)")

    def _clean_with_xen_ke(self, text: str) -> Tuple[str, bool, str]:
        primary = self.next_api
        secondary = "gemini" if primary == "groq" else "groq"
        apis = [
            (primary, self._groq_clean if primary == "groq" else self._gemini_clean),
            (secondary, self._gemini_clean if secondary == "gemini" else self._groq_clean)
        ]

        for api_name, func in apis:
            for attempt in range(1, MAX_ATTEMPTS_PER_API + 1):
                print(f"   Trying {api_name.upper()} (lần {attempt})")
                cleaned, ok, error = func(text)
                if ok:
                    return cleaned, True, api_name
                print(f"     → {api_name.upper()} FAILED: {error}")
                if attempt < MAX_ATTEMPTS_PER_API:
                    time.sleep(2)
            print(f"   {api_name.upper()} hết lượt → chuyển")

        print("   CẢ 2 API FAIL → fallback")
        return text, False, "fallback"

    def _gemini_clean(self, text: str) -> Tuple[str, bool, str]:
        prompt = f"""Sửa lỗi OCR sách giáo khoa Toán. Giữ --- PAGE X ---. Sửa lỗi chính tả, OCR. Giữ công thức.

VĂN BẢN:
{text}

TRẢ VỀ NGAY VĂN BẢN ĐÃ SỬA:"""
        try:
            resp = requests.post(
                f"{self.gemini_url}?key={self.gemini_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": MAX_TOKENS_OUTPUT}
                },
                timeout=90
            )
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text'], True, "OK"
            error = f"HTTP {resp.status_code}"
            if resp.status_code == 429: error = "RATE LIMIT (429)"
            self._log(f"Gemini fail: {error}")
            return "", False, error
        except Exception as e:
            error = f"EXCEPTION: {str(e)[:100]}"
            self._log(f"Gemini fail: {error}")
            return "", False, error

    def _groq_clean(self, text: str) -> Tuple[str, bool, str]:
        prompt = f"""Sửa lỗi OCR sách giáo khoa. Giữ --- PAGE X ---. Sửa lỗi chính tả, OCR. Giữ công thức.

VĂN BẢN:
{text}

TRẢ VỀ NGAY VĂN BẢN ĐÃ SỬA:"""
        try:
            resp = requests.post(
                self.groq_url,
                headers={"Authorization": f"Bearer {self.groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": MAX_TOKENS_OUTPUT
                },
                timeout=90
            )
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'], True, "OK"
            error = f"HTTP {resp.status_code}"
            if resp.status_code == 429: error = "RATE LIMIT (429)"
            elif resp.status_code == 401: error = "INVALID KEY (401)"
            self._log(f"Groq fail: {error}")
            return "", False, error
        except Exception as e:
            error = f"EXCEPTION: {str(e)[:100]}"
            self._log(f"Groq fail: {error}")
            return "", False, error

    def _split_result(self, cleaned: str, batch: List[Dict], success: bool):
        if not success:
            return [(p['page_num'], p['text'], "fallback") for p in batch]
        page_map = {}
        cur_page = None
        cur_lines = []
        for line in cleaned.split('\n'):
            if line.strip().startswith('--- PAGE ') and line.strip().endswith(' ---'):
                if cur_page is not None:
                    page_map[cur_page] = '\n'.join(cur_lines).strip()
                try:
                    cur_page = int(line.strip().split()[2])
                    cur_lines = []
                except: cur_page = None
            elif cur_page is not None:
                cur_lines.append(line)
        if cur_page is not None:
            page_map[cur_page] = '\n'.join(cur_lines).strip()
        results = []
        for p in batch:
            num = p['page_num']
            if num in page_map and page_map[num].strip():
                results.append((num, page_map[num], "cleaned"))
            else:
                results.append((num, p['text'], "fallback"))
        return results

    def _write_page(self, num: int, text: str, api_label: str):
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"PAGE {num} | {api_label}\n")
            f.write(f"{'='*80}\n\n")
            f.write(text.strip() + "\n\n")


def main():
    print("\n" + "="*80)
    print("OPTIMAL DUAL LLM - GROQ ƯU TIÊN - 10 TRANG/BATCH - XEN KẼ API")
    print("="*80)

    try:
        cleaner = OptimalLLMCleaner()
        cleaner.process()
    except Exception as e:
        print(f"Lỗi: {e}")


if __name__ == "__main__":
    main()
