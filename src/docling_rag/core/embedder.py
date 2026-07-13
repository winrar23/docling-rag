import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps SentenceTransformer for generating L2-normalized embeddings.
    Model is loaded once at __init__.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Args:
            texts: list of strings to embed
            batch_size: number of texts to encode in each batch (default 32)
        Returns:
            np.ndarray shape (N, 384), L2-normalized float32 vectors
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # L2 normalization → dot product == cosine similarity
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32, copy=False)
