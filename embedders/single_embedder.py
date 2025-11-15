"""
embedders/single_embedder.py

Single model embedding strategy
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class SingleEmbedder:
    """Single model embedding strategy"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Args:
            model_name: Model from HuggingFace
            device: "cpu" or "cuda"
            batch_size: Batch size for encoding
        """
        print(f"Loading model: {model_name}...")
        print("   (This may take a while on first load...)")

        # Simple load - để transformers tự quyết định dùng safetensors hay pytorch_model
        self.model = SentenceTransformer(model_name, device=device)

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✅ Model loaded!")
        print(f"   Dimension: {self.embedding_dim}")
        print(f"   Device: {device}")

    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Encode text(s) to embeddings"""
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
        print(f"💾 Saved embeddings: {filepath}")

    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings"""
        embeddings = np.load(filepath)
        print(f"📂 Loaded embeddings: {filepath}")
        print(f"   Shape: {embeddings.shape}")
        return embeddings

    def get_info(self) -> dict:
        """Get embedder info"""
        return {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'batch_size': self.batch_size
        }