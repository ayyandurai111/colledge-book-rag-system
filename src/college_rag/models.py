"""
models.py
---------
Shared dataclasses used across the whole package. These act as the
"contract" between the ingestion, chunking, vectorstore, and retrieval
layers.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TextBlock:
    """A contiguous piece of text extracted from a document, associated
    with a single heading."""
    text: str
    source_file: str
    page_number: Optional[int] = None
    heading: str = ""

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("TextBlock.text must not be empty")


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit produced after semantic chunking."""
    text: str
    source_file: str
    heading: str = ""
    page_number: Optional[int] = None
    chunk_id: Optional[int] = None

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Chunk.text must not be empty")

    def location_label(self) -> str:
        """A human-readable location string for UI/citation purposes."""
        parts = [self.source_file]
        if self.heading:
            parts.append(self.heading)
        label = " › ".join(parts)
        if self.page_number:
            label += f" (page {self.page_number})"
        return label


@dataclass(frozen=True)
class SearchResult:
    """A single retrieval result for a query.

    `highlight`: the specific sentence(s) within this chunk that are most
    relevant to the query — not the whole chunk. Empty by default; callers
    that don't need sentence-level highlighting can ignore it and use
    `chunk.text` directly (this is what the CLI/UI do by default).
    """
    chunk: Chunk
    score: float
    highlight: str = ""

    def __post_init__(self):
        if not (-1.0001 <= self.score <= 1.0001):
            raise ValueError(f"score must be within cosine similarity range [-1, 1]: {self.score}")

    def best_text(self) -> str:
        """The best text to display — the highlight if present, otherwise the full chunk."""
        return self.highlight if self.highlight else self.chunk.text


@dataclass
class IndexStats:
    """Metadata about the current state of an index."""
    total_chunks: int = 0
    total_source_files: int = 0
    source_files: list = field(default_factory=list)
