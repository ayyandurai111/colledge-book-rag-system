"""
semantic_chunker.py
--------------------
Semantic Chunking: splits text into chunks based on sentence-embedding
similarity — where a topic switch occurs — rather than fixed-size
splitting.

How it works:
1. Split each TextBlock into sentences
2. Compute an embedding for every sentence
3. Compute cosine similarity between consecutive sentences
4. Start a new chunk wherever similarity drops (a topic switch)
5. Respect min/max chunk size limits
"""
import re
from itertools import count
from typing import List, Optional

import numpy as np

from college_rag.embeddings.embedder import Embedder
from college_rag.models import Chunk, TextBlock

# Handles both Tamil (।) and English (. ! ?) sentence-ending punctuation
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?।])\s+|\n{2,}')


def split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


class SemanticChunker:
    """Converts TextBlocks into Chunks based on semantic similarity."""

    def __init__(
        self,
        embedder: Embedder,
        min_chunk_chars: int = 200,
        max_chunk_chars: int = 1200,
        breakpoint_percentile: float = 75.0,
    ):
        if max_chunk_chars <= min_chunk_chars:
            raise ValueError("max_chunk_chars must be greater than min_chunk_chars")
        self.embedder = embedder
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.breakpoint_percentile = breakpoint_percentile

    def chunk_block(self, block: TextBlock, id_counter: Optional[count] = None) -> List[Chunk]:
        """Splits a single TextBlock into semantic chunks."""
        id_counter = id_counter if id_counter is not None else count()
        sentences = split_sentences(block.text)

        if len(sentences) <= 1:
            return [Chunk(
                text=block.text, source_file=block.source_file,
                heading=block.heading, page_number=block.page_number,
                chunk_id=next(id_counter),
            )]

        embeddings = self.embedder.encode(sentences)

        # Cosine similarity between consecutive sentences. Embeddings are
        # normalized, so dot product == cosine similarity.
        sims = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # Find the point in the similarity distribution where a "large gap"
        # (topic switch) occurs, via a percentile threshold.
        threshold = float(np.percentile(sims, 100 - self.breakpoint_percentile)) if sims else -1.0

        raw_chunks: List[str] = []
        current = [sentences[0]]
        current_len = len(sentences[0])

        for i in range(1, len(sentences)):
            sim_to_prev = sims[i - 1] if (i - 1) < len(sims) else 1.0
            sent = sentences[i]
            would_be_len = current_len + len(sent)

            topic_switch = sim_to_prev < threshold and current_len >= self.min_chunk_chars
            too_long = would_be_len > self.max_chunk_chars

            if topic_switch or too_long:
                raw_chunks.append(" ".join(current))
                current = [sent]
                current_len = len(sent)
            else:
                current.append(sent)
                current_len += len(sent)

        if current:
            raw_chunks.append(" ".join(current))

        return [
            Chunk(
                text=text, source_file=block.source_file,
                heading=block.heading, page_number=block.page_number,
                chunk_id=next(id_counter),
            )
            for text in raw_chunks if text.strip()
        ]

    def chunk_blocks(self, blocks: List[TextBlock]) -> List[Chunk]:
        """Semantically chunks every block for a whole book.

        `chunk_id` is assigned using a single shared counter across all
        blocks, so every chunk gets a unique ID.
        """
        id_counter = count()
        all_chunks: List[Chunk] = []
        for block in blocks:
            all_chunks.extend(self.chunk_block(block, id_counter=id_counter))
        return all_chunks
