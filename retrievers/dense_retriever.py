"""
dense_retriever.py - FULL VERSION
- GPU/CPU support
- Save/Load FAISS + chunks
- Safe fit() với kiểm tra lỗi
- Tìm kiếm top-k + score
- Dùng trong app_chatbot_modular.py
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any
import os


class DenseRetriever:
    def __init__(self, embedder):
        """
        embedder: SingleEmbedder hoặc VietnameseEmbedder
        """
        self.embedder = embedder
        self.index = None
        self.chunks = None
        self.dim = None

    def fit(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Fit FAISS index từ chunks + embeddings
        """
        if len(chunks) == 0 or len(embeddings) == 0:
            raise ValueError("Chunks hoặc embeddings rỗng!")

        if len(chunks) != len(embeddings):
            raise ValueError(f"Số lượng chunks ({len(chunks)}) != embeddings ({len(embeddings)})")

        embeddings = np.array(embeddings)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"Embeddings shape sai: {embeddings.shape}")

        self.dim = embeddings.shape[1]
        self.chunks = chunks

        # Tạo FAISS Index (Inner Product = cosine nếu đã normalize)
        self.index = faiss.IndexFlatIP(self.dim)

        # Normalize embeddings nếu cần (bge-m3, gte-viet cần)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-6):
            embeddings = embeddings / norms

        self.index.add(embeddings.astype('float32'))
        print(f"FAISS Index: {self.index.ntotal} vectors, dim={self.dim}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm top-k chunks
        """
        if not self.index or not self.chunks:
            return []

        # Encode query
        q_emb = self.embedder.encode_chunks([{"text": query}], text_key='text')
        q_emb = np.array(q_emb).astype('float32')

        # Normalize query
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        # Search
        scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(score)
            results.append(chunk)

        return results

    def save(self, save_dir: str):
        """
        Lưu FAISS index + chunks
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # Lưu FAISS
        faiss.write_index(self.index, os.path.join(save_dir, "index.faiss"))

        # Lưu chunks
        with open(os.path.join(save_dir, "chunks.pkl"), 'wb') as f:
            pickle.dump(self.chunks, f)

        print(f"Retriever saved: {save_dir}")

    @classmethod
    def load(cls, load_dir: str, embedder):
        """
        Load retriever từ thư mục
        """
        index_path = os.path.join(load_dir, "index.faiss")
        chunks_path = os.path.join(load_dir, "chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Không tìm thấy file: {load_dir}")

        # Load FAISS
        index = faiss.read_index(index_path)

        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)

        # Tạo object
        retriever = cls(embedder)
        retriever.index = index
        retriever.chunks = chunks
        retriever.dim = index.d

        print(f"Retriever loaded: {load_dir} | {len(chunks)} chunks")
        return retriever

    @classmethod
    def exists(cls, path: str) -> bool:
        """Kiểm tra đã lưu chưa"""
        return (os.path.exists(os.path.join(path, "index.faiss")) and
                os.path.exists(os.path.join(path, "chunks.pkl")))