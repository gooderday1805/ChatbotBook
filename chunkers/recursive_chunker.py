"""
chunkers/recursive_chunker.py - Recursive text splitting (Model 1)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkers.base_chunker import BaseChunker  # ✅ Đổi sang import tuyệt đối
from typing import List, Dict

class RecursiveChunker(BaseChunker):
    """
    Recursive chunking strategy
    - Split theo đoạn văn trước
    - Nếu quá dài, split theo câu
    - Overlap giữa các chunks
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Max tokens per chunk (ước tính 1 token ≈ 4 chars)
            overlap: Số tokens overlap giữa chunks
        """
        super().__init__(chunk_size=chunk_size, overlap=overlap)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chars = chunk_size * 4  # Ước tính
        self.overlap_chars = overlap * 4

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk single text"""
        if not text:
            return []

        if len(text) <= self.max_chars:
            return [{
                'text': text,
                'chunk_id': 0,
                'chunk_size': len(text),
                **(metadata or {})
            }]

        # Split theo đoạn văn
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(test_chunk) <= self.max_chars:
                current_chunk = test_chunk
            else:
                # Save chunk hiện tại
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'chunk_id': chunk_id,
                        'chunk_size': len(current_chunk),
                        **(metadata or {})
                    })
                    chunk_id += 1

                # Nếu đoạn quá dài, split theo câu
                if len(para) > self.max_chars:
                    sub_chunks = self._split_long_paragraph(para)
                    for sub in sub_chunks:
                        chunks.append({
                            'text': sub,
                            'chunk_id': chunk_id,
                            'chunk_size': len(sub),
                            **(metadata or {})
                        })
                        chunk_id += 1
                    current_chunk = ""
                else:
                    current_chunk = para

        # Save chunk cuối
        if current_chunk:
            chunks.append({
                'text': current_chunk,
                'chunk_id': chunk_id,
                'chunk_size': len(current_chunk),
                **(metadata or {})
            })

        # Add total chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)

        return chunks

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split long paragraph by sentences"""
        sentences = [s.strip() + '.' for s in paragraph.split('. ') if s.strip()]

        chunks = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) <= self.max_chars:
                current += " " + sent if current else sent
            else:
                if current:
                    chunks.append(current)
                current = sent

        if current:
            chunks.append(current)

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


def test_recursive_chunker():
    """Test recursive chunker"""
    print("\n" + "=" * 80)
    print("TEST: Recursive Chunker")
    print("=" * 80 + "\n")

    chunker = RecursiveChunker(chunk_size=512, overlap=50)

    # Test with sample text
    text = """Tập hợp số tự nhiên là một khái niệm cơ bản trong toán học.

Các số tự nhiên bao gồm: 0, 1, 2, 3, 4, 5, ...

Tính chất của số tự nhiên:
- Có thể cộng, trừ, nhân, chia
- Có tính giao hoán
- Có tính kết hợp

Ví dụ: a + b = b + a (tính giao hoán)"""

    chunks = chunker.chunk_text(text, metadata={'page_num': 1})

    print(f"Input text: {len(text)} chars")
    print(f"Output: {len(chunks)} chunks\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  Size: {chunk['chunk_size']} chars")
        print(f"  Preview: {chunk['text'][:100]}...")
        print()

    stats = chunker.get_stats(chunks)
    print("Stats:", stats)


if __name__ == "__main__":
    test_recursive_chunker()