"""
Context Expander — "Auto-Merging" retrieval
============================================
Chunks are stored small (400-600 tokens) so *similarity search* stays sharp.
But an LLM answering a question usually needs the *whole section*, not one
window of it. This module takes a single similarity-search hit and expands
it back into the complete section it came from:

  Step 1  Similarity search already gave us the top chunk (caller's job).
  Step 2  Read that chunk's heading_path + chunk_index.
  Step 3  Pull EVERY chunk for that source straight from ChromaDB
          (ChromaStore.get_by_source), ordered by chunk_index.
  Step 4  Walk outward from the hit while heading_path stays identical.
          The walk stops the instant heading_path changes — that change
          IS the next-topic boundary. No re-parsing, no guessing: the
          metadata ChromaDB already stored (written once, at ingest time,
          by the chunker) is the single source of truth for where a
          section starts and ends.
  Step 5  Merge the contiguous run into one block of text, stripping the
          duplicate text created by the ingest-time sentence overlap
          (Fix: overlap_tokens) and the repeated heading-path prefix each
          window carries, so the merged section reads as one clean piece
          for the LLM instead of a stitched-together transcript.
"""
import re
from src.vectorstore.chroma_store import ChromaStore
from src.utils.tokenizer import count_tokens
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Safety cap: a heading_path (e.g. an entire UNIT with no subsections) could
# theoretically span the whole textbook. Stop merging past this many tokens
# so one expansion can never blow up the LLM context / cost / latency.
MAX_EXPANDED_TOKENS = 3000


def expand_chunk_to_section(chunk: dict, store: ChromaStore = None) -> dict:
    """
    Expand one retrieved chunk into the full section it belongs to.

    Args:
        chunk: A single result from retriever.retrieve() — must include
               'source', 'heading_path', and 'chunk_index'.
        store: Optional existing ChromaStore (created if None).

    Returns:
        {
          "source", "chapter", "section", "subsection", "heading_path",
          "chunk_type", "score" (of the original hit),
          "pages": [int, ...],           # every page the merged section spans
          "merged_chunk_count": int,     # how many stored chunks were merged
          "token_count": int,            # of the merged text
          "text": str                    # full, de-duplicated section text
        }
        Falls back to the original chunk (wrapped in the same shape) if
        expansion isn't possible (e.g. missing chunk_index).
    """
    if "chunk_index" not in chunk or not chunk.get("source"):
        logger.warning("Chunk missing chunk_index/source — cannot expand, returning as-is")
        return _passthrough(chunk)

    store = store or ChromaStore()
    doc_chunks = store.get_by_source(chunk["source"])
    if not doc_chunks:
        return _passthrough(chunk)

    # Step 3/4: locate the hit inside the ordered document, then walk
    # outward while heading_path is unchanged. Position in the SORTED LIST
    # (not the raw chunk_index value) is what matters — quality-gate drops
    # during ingestion can leave gaps in chunk_index, and walking by list
    # adjacency is robust to that.
    anchor_pos = _find_anchor(doc_chunks, chunk)
    if anchor_pos is None:
        return _passthrough(chunk)

    target_heading = doc_chunks[anchor_pos]["heading_path"]

    start = anchor_pos
    while start - 1 >= 0 and doc_chunks[start - 1]["heading_path"] == target_heading:
        start -= 1

    end = anchor_pos
    while end + 1 < len(doc_chunks) and doc_chunks[end + 1]["heading_path"] == target_heading:
        end += 1

    section_chunks = doc_chunks[start:end + 1]

    merged_text = _merge_without_overlap(section_chunks, target_heading)

    # Enforce the size cap by trimming trailing merged chunks, not by
    # truncating mid-sentence.
    if count_tokens(merged_text) > MAX_EXPANDED_TOKENS:
        merged_text = _cap_tokens(merged_text, MAX_EXPANDED_TOKENS)
        logger.info(
            f"Section '{target_heading}' exceeded {MAX_EXPANDED_TOKENS} tokens "
            f"after merge — trimmed to fit the cap"
        )

    anchor = doc_chunks[anchor_pos]
    return {
        "source":             anchor["source"],
        "chapter":            anchor["chapter"],
        "section":            anchor["section"],
        "subsection":         anchor["subsection"],
        "heading_path":       anchor["heading_path"],
        "chunk_type":         anchor["chunk_type"],
        "score":              chunk.get("score"),
        "pages":              sorted({c["page_number"] for c in section_chunks}),
        "merged_chunk_count": len(section_chunks),
        "token_count":        count_tokens(merged_text),
        "text":               merged_text,
    }


def expand_hits(chunks: list[dict], store: ChromaStore = None) -> list[dict]:
    """
    Expand a list of similarity-search hits into full sections.

    Multiple hits often land in the same section — those are de-duplicated
    (by source + heading_path) so the caller doesn't get the same expanded
    section twice, keeping the highest-scoring hit's rank position.
    """
    store = store or ChromaStore()
    seen: set[tuple] = set()
    expanded = []

    for chunk in chunks:
        key = (chunk.get("source"), chunk.get("heading_path"))
        if key in seen:
            continue
        seen.add(key)
        expanded.append(expand_chunk_to_section(chunk, store=store))

    return expanded


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _find_anchor(doc_chunks: list[dict], chunk: dict) -> int | None:
    """Find the retrieved chunk's position within the ordered document list."""
    for i, c in enumerate(doc_chunks):
        if c["chunk_index"] == chunk["chunk_index"]:
            return i
    # Fallback: chunk_index not found (e.g. store was re-ingested since the
    # search happened) — match on text as a last resort.
    for i, c in enumerate(doc_chunks):
        if c["text"] == chunk.get("text"):
            return i
    return None


def _strip_heading_prefix(text: str, heading_path: str) -> str:
    """
    The chunker prepends "heading_path: " to every window's text
    (see chunker._add_heading_prefix). When merging N windows from the
    same section, that prefix must appear once, not N times.
    """
    prefix = f"{heading_path}: "
    if heading_path and text.startswith(prefix):
        return text[len(prefix):]
    return text


def _longest_word_overlap(a_words: list[str], b_words: list[str], max_check: int) -> int:
    """
    Find how many trailing words of `a_words` exactly match the leading
    words of `b_words` (the sentence-overlap the chunker intentionally
    introduced between adjacent windows). Returns the overlap length in
    words, 0 if none found.
    """
    limit = min(max_check, len(a_words), len(b_words))
    for size in range(limit, 0, -1):
        if a_words[-size:] == b_words[:size]:
            return size
    return 0


def _merge_without_overlap(section_chunks: list[dict], heading_path: str) -> str:
    """
    Join a contiguous run of chunks into one clean block:
      - strip the repeated heading-path prefix from every window but the first
      - drop the duplicated words created by the chunker's sentence overlap
    """
    texts = [_strip_heading_prefix(c["text"], heading_path) for c in section_chunks]
    if not texts:
        return ""

    merged_words = texts[0].split()
    # Overlap can't exceed the configured overlap window in practice, but we
    # search generously (up to 60 words) since overlap is sentence-bounded,
    # not a fixed token cut.
    for text in texts[1:]:
        next_words = text.split()
        overlap = _longest_word_overlap(merged_words, next_words, max_check=60)
        merged_words.extend(next_words[overlap:])

    merged = " ".join(merged_words)
    # Collapse any accidental double spaces left by the join.
    merged = re.sub(r"\s+", " ", merged).strip()

    if heading_path:
        merged = f"{heading_path}: {merged}"
    return merged


def _cap_tokens(text: str, max_tokens: int) -> str:
    from src.utils.tokenizer import truncate_to_limit
    return truncate_to_limit(text, max_tokens)


def _passthrough(chunk: dict) -> dict:
    """Wrap a single (unexpanded) chunk in the same output shape as a merge."""
    return {
        "source":             chunk.get("source", ""),
        "chapter":            chunk.get("chapter", ""),
        "section":            chunk.get("section", ""),
        "subsection":         chunk.get("subsection", ""),
        "heading_path":       chunk.get("heading_path", ""),
        "chunk_type":         chunk.get("chunk_type", "text"),
        "score":              chunk.get("score"),
        "pages":              [chunk["page_number"]] if chunk.get("page_number") else [],
        "merged_chunk_count": 1,
        "token_count":        chunk.get("token_count", count_tokens(chunk.get("text", ""))),
        "text":               chunk.get("text", ""),
    }
