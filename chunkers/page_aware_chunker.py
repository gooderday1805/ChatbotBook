"""
chunkers/page_aware_chunker.py - Page-aware chunking (Model 2)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkers.base_chunker import BaseChunker
from typing import List, Dict
# from .base_chunker import BaseChunker


class PageAwareChunker(BaseChunker):
    """
    Page-aware chunking strategy
    - Giữ nguyên cấu trúc page
    - Ưu tiên giữ full page nếu có thể
    - Split theo đoạn văn nếu page quá dài
    - Preserve page context
    """

    def __init__(self, max_chars: int = 2000):
        """
        Args:
            max_chars: Max characters per chunk
        """
        super().__init__(max_chars=max_chars)
        self.max_chars = max_chars

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk single page"""
        if not text:
            return []

        page_num = metadata.get('page_num', 0) if metadata else 0

        # Nếu trang ngắn, giữ nguyên
        if len(text) <= self.max_chars:
            return [{
                'text': text,
                'page_num': page_num,
                'chunk_type': 'full_page',
                'chunk_id': 0,
                'total_chunks': 1,
                'chunk_size': len(text)
            }]

        # Nếu trang dài, chia theo đoạn văn
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""
        chunk_id = 0
        para_start = 0

        for i, para in enumerate(paragraphs):
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(test_chunk) <= self.max_chars:
                current_chunk = test_chunk
            else:
                # Save chunk hiện tại
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'page_num': page_num,
                        'chunk_type': 'partial_page',
                        'chunk_id': chunk_id,
                        'paragraph_range': f"para_{para_start}-{i - 1}",
                        'chunk_size': len(current_chunk)
                    })
                    chunk_id += 1
                    para_start = i

                # Bắt đầu chunk mới
                current_chunk = para

        # Save chunk cuối
        if current_chunk:
            chunks.append({
                'text': current_chunk,
                'page_num': page_num,
                'chunk_type': 'partial_page',
                'chunk_id': chunk_id,
                'paragraph_range': f"para_{para_start}-end",
                'chunk_size': len(current_chunk)
            })

        # Update total chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)

        return chunks

    def chunk_document(self, pages: List[Dict]) -> List[Dict]:
        """Chunk entire document"""
        all_chunks = []

        for page in pages:
            page_chunks = self.chunk_text(
                page['text'],
                metadata={'page_num': page['page_num']}
            )
            all_chunks.extend(page_chunks)

        return all_chunks

    def get_page_chunks(self, chunks: List[Dict], page_num: int) -> List[Dict]:
        """Get all chunks from a specific page"""
        return [c for c in chunks if c.get('page_num') == page_num]

    def get_full_page_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Get only full page chunks"""
        return [c for c in chunks if c.get('chunk_type') == 'full_page']


def test_page_aware_chunker():
    """Test page-aware chunker"""
    print("\n" + "=" * 80)
    print("TEST: Page-Aware Chunker")
    print("=" * 80 + "\n")

    chunker = PageAwareChunker(max_chars=2000)

    # Test với trang ngắn
    short_text = "Đây là trang ngắn. Nội dung ít."
    chunks = chunker.chunk_text(short_text, metadata={'page_num': 1})

    print("Test 1: Short page")
    print(f"  Input: {len(short_text)} chars")
    print(f"  Output: {len(chunks)} chunk(s)")
    print(f"  Type: {chunks[0]['chunk_type']}")
    print()

    # Test với trang dài
    long_text = "\n\n".join([f"Đoạn văn số {i}. " + "Nội dung dài. " * 50 for i in range(10)])
    chunks = chunker.chunk_text(long_text, metadata={'page_num': 2})

    print("Test 2: Long page")
    print(f"  Input: {len(long_text)} chars")
    print(f"  Output: {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks, 1):
        print(f"  Chunk {i}: {c['chunk_type']}, {c['chunk_size']} chars, {c['paragraph_range']}")
    print()

    stats = chunker.get_stats(chunks)
    print("Stats:", stats)


if __name__ == "__main__":
    test_page_aware_chunker()