"""
faiss_store.py
--------------
Stores chunk embeddings using FAISS and performs semantic search
(retrieval). The index can also be saved to / loaded from disk.
"""
import json
import logging
import os
import pickle
from typing import List

import faiss
import numpy as np

from college_rag.embeddings.embedder import Embedder
from college_rag.exceptions import EmptyIndexError, IndexNotBuiltError
from college_rag.models import Chunk, SearchResult

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "index.faiss"
_CHUNKS_FILENAME = "chunks.pkl"
_META_FILENAME = "meta.json"
_META_VERSION = 1


class FaissVectorStore:
    """A FAISS-backed in-memory + on-disk vector store for chunk embeddings."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: faiss.Index = None
        self.chunks: List[Chunk] = []

    @property
    def is_built(self) -> bool:
        return self.index is not None

    def build(self, chunks: List[Chunk]) -> None:
        """Builds a new FAISS index from a list of chunks.

        Raises:
            EmptyIndexError: If the chunks list is empty.
        """
        if not chunks:
            raise EmptyIndexError("Cannot build an index with an empty chunks list")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)

        dim = embeddings.shape[1]
        # Normalized embeddings + Inner Product = cosine similarity search
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.chunks = list(chunks)
        logger.info("Index built: %d chunks, dim=%d", len(self.chunks), dim)

    def add(self, chunks: List[Chunk]) -> None:
        """Adds more chunks to an existing index (incremental update)."""
        if not chunks:
            return
        if not self.is_built:
            self.build(chunks)
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info("Added %d chunks to the index. Total: %d", len(chunks), len(self.chunks))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Searches for the top_k chunks most similar to the query.

        Raises:
            IndexNotBuiltError: If search is attempted before build()/load().
        """
        if not self.is_built:
            raise IndexNotBuiltError("Call build() or load() first.")
        if not query or not query.strip():
            return []

        top_k = max(1, min(top_k, len(self.chunks)))

        q_emb = self.embedder.encode([query])
        scores, idxs = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            # If floating point rounding pushes a normalized-vector score
            # slightly outside [-1, 1], clamp it.
            clamped_score = float(np.clip(score, -1.0, 1.0))
            results.append(SearchResult(chunk=self.chunks[idx], score=clamped_score))
        return results

    def save(self, folder: str) -> None:
        """Saves the index + chunks + metadata to disk."""
        if not self.is_built:
            raise IndexNotBuiltError("Cannot save an empty index. Call build() first.")

        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, _INDEX_FILENAME))
        with open(os.path.join(folder, _CHUNKS_FILENAME), "wb") as f:
            pickle.dump(self.chunks, f)

        meta = {
            "version": _META_VERSION,
            "embedding_model": self.embedder.model_name,
            "total_chunks": len(self.chunks),
        }
        with open(os.path.join(folder, _META_FILENAME), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, folder: str) -> None:
        """Loads a previously-saved index from disk.

        Raises:
            FileNotFoundError: If required files are missing from `folder`.
        """
        index_path = os.path.join(folder, _INDEX_FILENAME)
        chunks_path = os.path.join(folder, _CHUNKS_FILENAME)
        meta_path = os.path.join(folder, _META_FILENAME)

        for p in (index_path, chunks_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"Index file not found: {p}")

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            saved_model = meta.get("embedding_model")
            if saved_model and saved_model != self.embedder.model_name:
                logger.warning(
                    "Index at '%s' was built with embedding model '%s', but the "
                    "current embedder uses '%s'. A dimension mismatch may occur.",
                    folder, saved_model, self.embedder.model_name,
                )

        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info("Index loaded: %d chunks (%s)", len(self.chunks), folder)
