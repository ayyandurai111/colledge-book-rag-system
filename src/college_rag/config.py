"""
config.py
---------
Centralized configuration for the whole package. Every value is
overridable via environment variables so behavior can be tuned in
production without touching code.
"""
import os
from dataclasses import dataclass


def _get_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # Embedding model — multilingual, supports both Tamil and English
    embedding_model_name: str = os.environ.get(
        "COLLEGE_RAG_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Semantic chunking parameters
    min_chunk_chars: int = _get_int("COLLEGE_RAG_MIN_CHUNK_CHARS", 200)
    max_chunk_chars: int = _get_int("COLLEGE_RAG_MAX_CHUNK_CHARS", 1200)
    breakpoint_percentile: float = _get_float("COLLEGE_RAG_BREAKPOINT_PERCENTILE", 75.0)

    # Retrieval
    default_top_k: int = _get_int("COLLEGE_RAG_TOP_K", 5)
    # How many sentences to select for SearchResult.highlight (Python-API-only feature;
    # not surfaced by the CLI/UI, which always show the full chunk).
    highlight_sentences: int = _get_int("COLLEGE_RAG_HIGHLIGHT_SENTENCES", 2)

    def validate(self) -> None:
        """Checks that config values are logically consistent."""
        if self.min_chunk_chars < 1:
            raise ValueError("min_chunk_chars must be greater than 0")
        if self.max_chunk_chars <= self.min_chunk_chars:
            raise ValueError("max_chunk_chars must be greater than min_chunk_chars")
        if not (0 <= self.breakpoint_percentile <= 100):
            raise ValueError("breakpoint_percentile must be within 0-100")
        if self.default_top_k < 1:
            raise ValueError("default_top_k must be at least 1")
        if self.highlight_sentences < 1:
            raise ValueError("highlight_sentences must be at least 1")


DEFAULT_CONFIG = Config()
