# storage/file_storage.py
import json
from pathlib import Path

import numpy as np

from core.chunker import Chunk


def _chunk_to_meta(chunk: Chunk) -> dict:
    return {
        "text": chunk.text,
        "source_file": chunk.source_file,
        "chunk_id": chunk.chunk_id,
        "page_number": chunk.page_number,
        "element_type": chunk.element_type,
    }


class FileStorage:
    """
    NumPy-backed storage: embeddings.npy (N x dim) + metadata.json.
    Implements the StorageBackend protocol.
    """

    EMB_FILE = "embeddings.npy"
    META_FILE = "metadata.json"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self._dir = Path(data_dir)

    def _emb_path(self) -> Path:
        return self._dir / self.EMB_FILE

    def _meta_path(self) -> Path:
        return self._dir / self.META_FILE

    def save(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._emb_path(), embeddings)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump([_chunk_to_meta(c) for c in chunks], f, ensure_ascii=False, indent=2)

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        try:
            existing_emb, existing_meta = self.load()
            new_emb = np.vstack([existing_emb, embeddings])
            new_meta = existing_meta + [_chunk_to_meta(c) for c in chunks]
        except FileNotFoundError:
            new_emb = embeddings
            new_meta = [_chunk_to_meta(c) for c in chunks]

        self._dir.mkdir(parents=True, exist_ok=True)
        np.save(self._emb_path(), new_emb)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(new_meta, f, ensure_ascii=False, indent=2)

    def load(self) -> tuple[np.ndarray, list[dict]]:
        if not self._emb_path().exists():
            raise FileNotFoundError(f"Storage not found: {self._emb_path()}")
        embeddings = np.load(self._emb_path())
        with open(self._meta_path(), encoding="utf-8") as f:
            metadata = json.load(f)
        return embeddings, metadata

    def delete_by_source(self, source_file: str) -> None:
        embeddings, metadata = self.load()
        keep = [i for i, m in enumerate(metadata) if m["source_file"] != source_file]
        if not keep:
            self._emb_path().unlink(missing_ok=True)
            self._meta_path().unlink(missing_ok=True)
            return
        new_emb = embeddings[keep]
        new_meta = [metadata[i] for i in keep]
        np.save(self._emb_path(), new_emb)
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(new_meta, f, ensure_ascii=False, indent=2)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """
        Linear cosine similarity search.
        query_embedding must be L2-normalized.
        Since stored embeddings are normalized, cosine_sim = dot(query, vec).
        """
        embeddings, metadata = self.load()
        scores: np.ndarray = embeddings @ query_embedding  # (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(metadata[i], float(scores[i])) for i in top_indices]
