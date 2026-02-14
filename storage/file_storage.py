# storage/file_storage.py
import json
import os
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

    def _atomic_save(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Write both files atomically using temp-file-then-rename."""
        self._dir.mkdir(parents=True, exist_ok=True)
        emb_tmp = self._dir / "embeddings.tmp.npy"
        meta_tmp = self._dir / "metadata.tmp.json"
        try:
            np.save(emb_tmp, embeddings)
            with open(meta_tmp, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            os.replace(emb_tmp, self._emb_path())
            os.replace(meta_tmp, self._meta_path())
        except Exception:
            emb_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)
            raise

    def save(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )
        self._atomic_save(embeddings, [_chunk_to_meta(c) for c in chunks])

    def append(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings"
            )
        try:
            existing_emb, existing_meta = self.load()
            new_emb = np.vstack([existing_emb, embeddings])
            new_meta = existing_meta + [_chunk_to_meta(c) for c in chunks]
        except FileNotFoundError:
            new_emb = embeddings
            new_meta = [_chunk_to_meta(c) for c in chunks]

        self._atomic_save(new_emb, new_meta)

    def load(self) -> tuple[np.ndarray, list[dict]]:
        if not self._emb_path().exists():
            raise FileNotFoundError(f"Storage not found: {self._emb_path()}")
        if not self._meta_path().exists():
            raise FileNotFoundError(
                f"Storage is corrupted: embeddings.npy exists but metadata.json is missing in {self._dir}"
            )
        embeddings = np.load(self._emb_path())
        with open(self._meta_path(), encoding="utf-8") as f:
            metadata = json.load(f)
        return embeddings, metadata

    def delete_by_source(self, source_file: str) -> None:
        try:
            embeddings, metadata = self.load()
        except FileNotFoundError:
            return  # nothing to delete

        keep = [i for i, m in enumerate(metadata) if m["source_file"] != source_file]
        if len(keep) == len(metadata):
            return  # source_file not found — no-op, avoid unnecessary rewrite
        if not keep:
            self._emb_path().unlink(missing_ok=True)
            self._meta_path().unlink(missing_ok=True)
            return
        self._atomic_save(embeddings[keep], [metadata[i] for i in keep])

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """
        Linear cosine similarity search.
        query_embedding must be L2-normalized.
        Since stored embeddings are normalized, cosine_sim = dot(query, vec).
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        embeddings, metadata = self.load()
        if embeddings.shape[1] != query_embedding.shape[0]:
            raise ValueError(
                f"Dimension mismatch: stored={embeddings.shape[1]}, query={query_embedding.shape[0]}"
            )
        scores: np.ndarray = embeddings @ query_embedding  # (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(metadata[i], float(scores[i])) for i in top_indices]
