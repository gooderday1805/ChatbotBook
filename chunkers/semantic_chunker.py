"""
chunkers/semantic_chunker.py - Semantic + Page-aware chunking (Model 3)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunkers.base_chunker import BaseChunker
from typing import List, Dict
import re
# from .base_chunker import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic + Page-aware chunking strategy
    - Phát hiện semantic boundaries (headings, sections)
    - Giữ context page
    - Adaptive chunk size
    """

    def __init__(self, detect_headings: bool = True):
        """
        Args:
            detect_headings: Có detect headings/sections không
        """
        super().__init__(detect_headings=detect_headings)
        self.detect_headings = detect_headings

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk single page with semantic awareness"""
        if not text:
            return []

        page_num = metadata.get('page_num', 0) if metadata else 0

        if not self.detect_headings:
            # Fallback to simple chunking
            return [{
                'text': text,
                'page_num': page_num,
                'chunk_type': 'full_page',
                'chunk_id': 0,
                'total_chunks': 1,
                'section_heading': 'N/A'
            }]

        # Detect sections
        sections = self._detect_sections(text)

        if not sections:
            return [{
                'text': text,
                'page_num': page_num,
                'chunk_type': 'full_page',
                'chunk_id': 0,
                'total_chunks': 1,
                'section_heading': 'N/A'
            }]

        # Convert sections to chunks
        chunks = []
        for i, section in enumerate(sections):
            heading = section['heading'] or f"Section {i + 1}"
            content = section['content']

            # Add heading to content if exists
            full_text = f"{heading}\n\n{content}" if section['heading'] else content

            chunks.append({
                'text': full_text,
                'page_num': page_num,
                'chunk_type': 'semantic_section',
                'section_heading': heading,
                'chunk_id': i,
                'total_chunks': len(sections),
                'chunk_size': len(full_text)
            })

        return chunks

    def _detect_sections(self, text: str) -> List[Dict]:
        """
        Detect semantic sections in text

        Heading patterns:
        1. ALL CAPS (VÍ DỤ)
        2. Số ở đầu (1. Giới thiệu, 1) Bài tập)
        3. Dòng ngắn không có dấu chấm cuối
        """
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        sections = []
        current_section = []
        current_heading = None

        for line in lines:
            is_heading = self._is_heading(line)

            if is_heading and current_section:
                # Save previous section
                sections.append({
                    'heading': current_heading,
                    'content': '\n'.join(current_section)
                })
                current_section = []
                current_heading = line
            elif is_heading:
                current_heading = line
            else:
                current_section.append(line)

        # Save last section
        if current_section:
            sections.append({
                'heading': current_heading,
                'content': '\n'.join(current_section)
            })

        return sections

    def _is_heading(self, line: str) -> bool:
        """Check if line is a heading"""
        # Pattern 1: ALL CAPS
        if line.isupper() and len(line) < 100:
            return True

        # Pattern 2: Starts with number
        if re.match(r'^\d+[\.\)]', line):
            return True

        # Pattern 3: Short line without period at end
        if len(line) < 50 and not line.endswith('.') and not line.endswith(','):
            # Check if it looks like a title (capitalized, etc.)
            words = line.split()
            if words and words[0][0].isupper():
                return True

        return False

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

    def get_sections_by_heading(self, chunks: List[Dict], heading_keyword: str) -> List[Dict]:
        """Get chunks by heading keyword"""
        return [
            c for c in chunks
            if heading_keyword.lower() in c.get('section_heading', '').lower()
        ]


def test_semantic_chunker():
    """Test semantic chunker"""
    print("\n" + "=" * 80)
    print("TEST: Semantic Chunker")
    print("=" * 80 + "\n")

    chunker = SemanticChunker(detect_headings=True)

    # Test with sections
    text = """TẬP HỢP SỐ TỰ NHIÊN

Tập hợp số tự nhiên là một khái niệm cơ bản.

1. Định nghĩa

Các số tự nhiên: 0, 1, 2, 3, 4, 5, ...

2. Tính chất

Số tự nhiên có các tính chất:
- Tính giao hoán
- Tính kết hợp

VÍ DỤ

Ví dụ 1: a + b = b + a"""

    chunks = chunker.chunk_text(text, metadata={'page_num': 1})

    print(f"Input: {len(text)} chars")
    print(f"Detected {len(chunks)} sections:\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Section {i}: {chunk['section_heading']}")
        print(f"  Type: {chunk['chunk_type']}")
        print(f"  Size: {chunk['chunk_size']} chars")
        print(f"  Preview: {chunk['text'][:80]}...")
        print()

    stats = chunker.get_stats(chunks)
    print("Stats:", stats)


if __name__ == "__main__":
    test_semantic_chunker()