from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

class BaseChunker(ABC):
    """Base class cho tất cả chunking strategies"""

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text into smaller pieces

        Args:
            text: Text to chunk
            metadata: Optional metadata (page_num, etc.)

        Returns:
            List of chunks with metadata
        """
        pass

    @abstractmethod
    def chunk_document(self, pages: List[Dict]) -> List[Dict]:
        """
        Chunk entire document (multiple pages)

        Args:
            pages: List of pages with 'page_num' and 'text'

        Returns:
            List of all chunks with metadata
        """
        pass

    def load_from_file(self, file_path: str) -> List[Dict]:
        """Load and parse document from file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding='utf-8')

        # Parse pages
        pages = []
        for block in content.split('\n' + '=' * 80):
            if not block.strip():
                continue

            lines = block.strip().split('\n')
            if not lines:
                continue

            page_num = None
            text_start = 0

            for i, line in enumerate(lines):
                if line.startswith('PAGE '):
                    try:
                        page_num = int(line.split('PAGE ')[1].split('|')[0].strip())
                        text_start = i + 2
                        break
                    except:
                        pass

            if page_num is None:
                continue

            text = '\n'.join(lines[text_start:]).strip()
            if text:
                pages.append({'page_num': page_num, 'text': text})

        return pages

    def get_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {}

        chunk_sizes = [len(c['text']) for c in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_size': min(chunk_sizes),
            'max_size': max(chunk_sizes),
            'total_chars': sum(chunk_sizes)
        }