"""
retriever.py
------------
A convenience layer over FaissVectorStore. Handles a default top_k, and
optionally computes sentence-level "highlights" within each retrieved
chunk — the specific sentence(s) inside a chunk most relevant to the
query. This is available via the Python API (`SearchResult.highlight`);
the CLI/UI do not surface it by default and instead show the full chunk.
"""
from dataclasses import replace
from typing import List, Optional

import numpy as np

from college_rag.chunking.semantic_chunker import split_sentences
from college_rag.embeddings.embedder import Embedder
from college_rag.models import SearchResult
from college_rag.vectorstore.faiss_store import FaissVectorStore


class Retriever:
    def __init__(
        self,
        vectorstore: FaissVectorStore,
        embedder: Optional[Embedder] = None,
        default_top_k: int = 5,
        highlight_sentences: int = 2,
    ):
        self.vectorstore = vectorstore
        # An embedder is needed to compute highlights — if none is given,
        # reuse the same embedder the vectorstore already uses (to stay
        # consistent within a single embedding space).
        self.embedder = embedder or vectorstore.embedder
        self.default_top_k = default_top_k
        self.highlight_sentences = max(1, highlight_sentences)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        k = top_k if top_k is not None else self.default_top_k
        results = self.vectorstore.search(query, top_k=k)
        if not results or not query.strip():
            return results
        return [self._with_highlight(query, r) for r in results]

    # ------------------------------------------------------------------ #
    # Sentence-level highlight extraction
    # ------------------------------------------------------------------ #
    def _with_highlight(self, query: str, result: SearchResult) -> SearchResult:
        highlight = self._extract_highlight(query, result.chunk.text)
        return replace(result, highlight=highlight)

    def _extract_highlight(self, query: str, chunk_text: str) -> str:
        """Returns the `highlight_sentences` sentences within chunk_text most
        relevant to the query, in their original reading order.

        If the chunk has fewer sentences than highlight_sentences, the
        whole chunk is returned (nothing to trim).
        """
        sentences = split_sentences(chunk_text)
        if len(sentences) <= self.highlight_sentences:
            return chunk_text

        query_vec = self.embedder.encode([query])[0]
        sentence_vecs = self.embedder.encode(sentences)
        # Embeddings are normalized, so dot product == cosine similarity.
        sims = sentence_vecs @ query_vec

        top_n = min(self.highlight_sentences, len(sentences))
        top_indices = np.argsort(sims)[::-1][:top_n]
        # Re-sort to preserve original reading order in the output.
        ordered_indices = sorted(top_indices.tolist())
        return " ".join(sentences[i] for i in ordered_indices)
