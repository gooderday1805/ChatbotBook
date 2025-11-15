"""
STEP 1: OCR + TEXT PROCESSOR (v3) - Fixed Memory Issues
- Extract TEXT từ PDF (nếu có)
- Extract IMAGES từ PDF và OCR (nếu có)
- Combine cả 2 vào đúng page
- Save to before_llm.txt
- Sử dụng logic resize đã test thành công từ code gốc
"""
import sys
from pathlib import Path
from PIL import Image
from typing import List, Dict
import time
import fitz  # PyMuPDF

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import easyocr
    import torch
except ImportError:
    print("❌ Thiếu thư viện! Cài đặt:")
    print("   pip install easyocr torch pillow PyMuPDF")
    sys.exit(1)


class PDFTextImageProcessor:
    """Extract cả TEXT và IMAGE từ PDF"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text + images từ PDF"""
        doc = fitz.open(pdf_path)
        pages = []

        print(f"📄 Đang xử lý PDF: {pdf_path}")
        print(f"   Tổng số trang: {len(doc)}")

        for page_num in range(len(doc)):
            page = doc[page_num]

            # 1. Extract TEXT
            text = page.get_text("text").strip()

            # 2. Extract IMAGES
            images = []
            image_list = page.get_images()

            if image_list:
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Save image
                        img_ext = base_image["ext"]
                        img_filename = f"page_{page_num + 1}_img_{img_idx + 1}.{img_ext}"
                        img_path = self.images_dir / img_filename  # Lưu vào output/images/

                        with open(img_path, "wb") as f:
                            f.write(image_bytes)

                        images.append({
                            'path': str(img_path),
                            'index': img_idx + 1
                        })
                    except Exception as e:
                        print(f"   ⚠️  Không extract được image {img_idx + 1} từ page {page_num + 1}: {e}")

            pages.append({
                'page_num': page_num + 1,
                'text': text,
                'images': images,
                'has_text': len(text) > 0,
                'has_images': len(images) > 0
            })

            if (page_num + 1) % 10 == 0:
                print(f"   Đã xử lý {page_num + 1}/{len(doc)} trang")

        doc.close()
        print(f"✅ Hoàn thành extract {len(pages)} trang")
        return pages


class OCRProcessor:
    """OCR processor với detailed logging - COPY EXACT LOGIC TỪ CODE GỐC"""

    def __init__(self):
        self.output_file = Path("output/before_llm.txt")
        self.log_file = Path("output/ocr_log.txt")

        Path("output").mkdir(exist_ok=True)
        self.output_file.write_text("", encoding='utf-8')
        self.log_file.write_text("", encoding='utf-8')

        self._init_ocr()

        print(f"📁 Output: {self.output_file}")
        print(f"📋 Log: {self.log_file}")

    def _init_ocr(self):
        """Try GPU first, fallback to CPU - EXACT COPY"""
        try:
            print("\n🔄 Initializing EasyOCR...")

            # Check GPU
            has_gpu = torch.cuda.is_available()

            if has_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"   🎮 GPU detected: {gpu_name}")
                print(f"   🔄 Loading with GPU...")

                try:
                    self.reader = easyocr.Reader(['vi', 'en'], gpu=True, verbose=False)
                    self.device = 'GPU'
                    print(f"   ✅ Running on GPU: {gpu_name}\n")
                    self._log(f"EasyOCR initialized with GPU: {gpu_name}")
                    return
                except Exception as e:
                    print(f"   ⚠️  GPU init failed: {e}")
                    print(f"   🔄 Falling back to CPU...")
                    self._log(f"GPU init failed: {e}")
            else:
                print(f"   ⚠️  No GPU detected")
                print(f"   🔄 Loading with CPU...")

            # Fallback to CPU
            self.reader = easyocr.Reader(['vi', 'en'], gpu=False, verbose=False)
            self.device = 'CPU'
            print(f"   ✅ Running on CPU\n")
            self._log(f"EasyOCR initialized with CPU")

        except Exception as e:
            print(f"❌ Failed to initialize EasyOCR: {e}")
            self._log(f"FATAL: EasyOCR init failed: {e}")
            sys.exit(1)

    def _log(self, message: str):
        """Write to log file"""
        timestamp = time.strftime("%H:%M:%S")
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    def _resize_if_needed(self, path: Path, max_size=1920) -> Path:
        """EXACT COPY từ code gốc - đã test ngon"""
        try:
            img = Image.open(path)
            w, h = img.size

            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                resized = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
                new_path = path.parent / f"resized_{path.name}"
                resized.save(new_path, quality=95)
                img.close()
                resized.close()
                return new_path
            else:
                img.close()
                return path
        except Exception as e:
            self._log(f"WARNING: Resize failed for {path.name}: {e}")
            return path

    def process_pdf(self, pdf_path: str):
        pdf_path = Path(pdf_path)

        print(f"\n{'=' * 80}")
        print(f"📚 {pdf_path.name}")
        print(f"{'=' * 80}\n")

        self._log(f"=" * 80)
        self._log(f"Processing: {pdf_path.name}")
        self._log(f"Device: {self.device}")
        self._log(f"=" * 80)

        # Step 1: Extract TEXT + IMAGES
        print("🔄 STEP 1: Extract text + images...")
        proc = PDFTextImageProcessor(output_dir=Path("output"))
        all_pages = proc.process_pdf(str(pdf_path))
        print(f"✅ {len(all_pages)} pages\n")
        self._log(f"Extracted {len(all_pages)} pages")

        # Step 2: Process all pages
        print(f"🔄 STEP 2: Process all pages (device: {self.device})...")
        print(f"{'─' * 80}")
        start = time.time()

        success_count, error_count = self._process_all(all_pages)

        elapsed = time.time() - start

        print(f"{'─' * 80}")
        print(f"\n{'=' * 80}")
        print("✅ HOÀN THÀNH!")
        print(f"{'=' * 80}")
        print(f"⏱️  Time: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
        print(f"📊 Results:")
        print(f"   ✅ Success: {success_count} pages")
        print(f"   ❌ Errors: {error_count} pages")
        print(f"📁 Output: {self.output_file} ({self.output_file.stat().st_size:,} bytes)")
        print(f"📋 Log: {self.log_file}")
        print(f"\n💡 Next step: Run test_full_file.py to clean the text")

        self._log(f"Processing completed: {success_count} success, {error_count} errors")
        self._log(f"Total time: {elapsed:.0f}s")
        self._log(f"Output size: {self.output_file.stat().st_size:,} bytes")

    def _process_all(self, pages: List[Dict]):
        """
        Process all pages - LOGIC ĐÚNG:
        1. Nếu có TEXT → thêm vào parts
        2. Nếu có IMAGES → OCR từng cái và thêm vào parts
        3. Combine parts → save

        OCR logic copy y hệt code gốc đã chạy ngon
        """
        total_text_chars = 0
        total_ocr_words = 0
        success_count = 0
        error_count = 0

        for idx, page in enumerate(pages, 1):
            page_num = page['page_num']

            try:
                parts = []

                # 1. Thêm TEXT nếu có
                if page['has_text']:
                    parts.append(page['text'])
                    total_text_chars += len(page['text'])

                # 2. OCR IMAGES nếu có - EXACT COPY logic từ code gốc
                ocr_parts = []
                if page['has_images']:
                    for img_idx, img in enumerate(page['images'], 1):
                        try:
                            path = Path(img['path'])
                            # Resize nhỏ hơn cho ảnh từ PyMuPDF (1280 thay vì 1920)
                            # vì PyMuPDF extract ảnh ở resolution cao hơn
                            resized = self._resize_if_needed(path, max_size=1280)

                            # OCR - EXACT COPY
                            img_start = time.time()
                            try:
                                result = self.reader.readtext(str(resized), detail=0)
                            except Exception as ocr_err:
                                # Nếu lỗi memory, thử resize nhỏ hơn nữa
                                if "allocate" in str(ocr_err).lower():
                                    self._log(f"Page {page_num} Image {img_idx}: Memory error, retry 640px")

                                    # Xóa resize cũ
                                    if resized != path:
                                        try:
                                            resized.unlink()
                                        except:
                                            pass

                                    # Resize cực nhỏ
                                    resized = self._resize_if_needed(path, max_size=640)

                                    try:
                                        # Clear GPU cache trước khi retry
                                        if self.device == 'GPU':
                                            torch.cuda.empty_cache()
                                        result = self.reader.readtext(str(resized), detail=0)
                                    except:
                                        result = []  # Skip nếu vẫn fail
                                        self._log(f"Page {page_num} Image {img_idx}: Failed even at 640px")
                                else:
                                    raise

                            img_elapsed = time.time() - img_start

                            text = " ".join(result)

                            if resized != path:
                                try:
                                    resized.unlink()
                                except:
                                    pass

                            # Clear GPU memory sau mỗi ảnh
                            if self.device == 'GPU':
                                try:
                                    torch.cuda.empty_cache()
                                except:
                                    pass

                            ocr_parts.append(text)

                            # Log detail nếu có nhiều images
                            if len(page['images']) > 1:
                                self._log(f"Page {page_num} Image {img_idx}/{len(page['images'])}: "
                                          f"{len(text)} chars, {img_elapsed:.1f}s")

                        except Exception as e:
                            error_msg = str(e)[:100]
                            print(f"   [{idx}/{len(pages)}] Page {page_num} Image {img_idx}: ❌ {error_msg}")
                            self._log(f"ERROR Page {page_num} Image {img_idx}: {error_msg}")

                # Combine OCR parts
                if ocr_parts:
                    ocr_text = "\n\n".join(ocr_parts)
                    parts.append(ocr_text)
                    total_ocr_words += len(ocr_text.split())

                # 3. Combine all parts
                page['final_text'] = "\n\n".join(parts)

                # Save ngay
                self._save_page(page)
                success_count += 1

                # Console log
                status_parts = []
                if page['has_text']:
                    status_parts.append(f"📝 {len(page['text'])} chars")
                if page['has_images']:
                    ocr_word_count = sum(len(p.split()) for p in ocr_parts)
                    status_parts.append(f"🖼️  {len(page['images'])} imgs, {ocr_word_count} words OCR")

                if not status_parts:
                    status_parts.append("⚠️  Empty")

                print(f"   [{idx}/{len(pages)}] Page {page_num}: ✅ {', '.join(status_parts)}")

                # Detail log
                self._log(f"Page {page_num}: SUCCESS - "
                          f"text={len(page.get('text', ''))} chars, "
                          f"images={len(page['images'])}, "
                          f"final={len(page['final_text'])} chars")

                # Clean GPU memory mỗi 10 trang
                if idx % 10 == 0 and self.device == 'GPU':
                    try:
                        torch.cuda.empty_cache()
                        self._log(f"GPU cache cleared at page {page_num}")
                    except:
                        pass

            except Exception as e:
                error_count += 1
                error_msg = str(e)[:100]
                page['final_text'] = ""

                print(f"   [{idx}/{len(pages)}] Page {page_num}: ❌ ERROR - {error_msg}")
                self._log(f"ERROR Page {page_num}: {error_msg}")

                # Save empty page
                self._save_page(page)

        print(f"\n📊 Summary:")
        print(f"   Total pages: {len(pages)}")
        print(f"   Success: {success_count}")
        print(f"   Errors: {error_count}")
        print(f"   PDF text: {total_text_chars:,} chars")
        print(f"   OCR words: {total_ocr_words:,} words")

        self._log(f"Summary: {success_count} success, {error_count} errors, "
                  f"{total_text_chars} text chars, {total_ocr_words} OCR words")

        return success_count, error_count

    def _save_page(self, page: Dict):
        """Save page ngay sau khi xử lý xong"""
        with open(self.output_file, 'a', encoding='utf-8') as f:
            f.write(f"===PAGE_{page['page_num']}===\n")
            f.write(page.get('final_text', ''))
            f.write("\n\n")


def main():
    print("\n" + "=" * 80)
    print("STEP 1: OCR + TEXT PROCESSOR (v3 - Fixed)")
    print("Xử lý cả TEXT và IMAGES từ PDF")
    print("=" * 80)

    pdf = input("\n📂 PDF path (Enter=default): ") or "D:/ChatbotBook/toanlop6tap1.pdf"
    print(f"\n⚡ Starting processor...")
    print(f"   📄 PDF: {pdf}")
    print(f"   🔄 Will extract TEXT + OCR IMAGES")

    processor = OCRProcessor()

    start = time.time()
    processor.process_pdf(pdf)

    print(f"\n🎉 Done in {time.time() - start:.0f}s ({(time.time() - start) / 60:.1f}m)!")


if __name__ == "__main__":
    main()