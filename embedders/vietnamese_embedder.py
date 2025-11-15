"""
embedders/vietnamese_embedder.py

Vietnamese-optimized embedding strategy
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class VietnameseEmbedder:
    """Vietnamese-optimized embedding"""

    MODELS = {
        'gte-viet': "Alibaba-NLP/gte-multilingual-base",
        'bge-m3': "BAAI/bge-m3",
        'minilm': "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    }

    def __init__(
        self,
        model_key: str = "gte-viet",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Args:
            model_key: 'gte-viet', 'bge-m3', or 'minilm'
            device: "cpu" or "cuda"
            batch_size: Batch size
        """
        model_name = self.MODELS.get(model_key, self.MODELS['gte-viet'])

        print(f"Loading Vietnamese model: {model_name}...")
        print("   (This may take a while on first load...)")

        # Simple load - để transformers tự quyết định
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True  # ← THÊM DÒNG NÀY
        )

        self.model_name = model_name
        self.model_key = model_key
        self.device = device
        self.batch_size = batch_size

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded!")
        print(f"   Model: {model_key}")
        print(f"   Dimension: {self.embedding_dim}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode texts"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )

        return embeddings

    def encode_chunks(self, chunks: List[dict], text_key: str = 'text') -> np.ndarray:
        """Encode chunks"""
        texts = [chunk[text_key] for chunk in chunks]
        return self.encode(texts)

    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings"""
        np.save(filepath, embeddings)
        print(f"💾 Saved: {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings"""
        return np.load(filepath)

    def get_info(self) -> dict:
        """Get info"""
        return {
            'model_key': self.model_key,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'optimized_for': 'Vietnamese'
        }