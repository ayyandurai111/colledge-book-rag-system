import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.context_expander import (
    expand_chunk_to_section, expand_hits, MAX_EXPANDED_TOKENS
)


def _chunk(idx, heading_path, text, page=1, dim=10, source="test.pdf"):
    """Build a chunk in the shape chunker.py produces (heading-prefixed text)."""
    return {
        "chunk_id":     f"{source}_c{idx:03d}",
        "text":         f"{heading_path}: {text}" if heading_path else text,
        "embedding":    [float(idx) / 100] * dim,
        "source":       source,
        "chapter":      "Chapter 1",
        "section":      heading_path,
        "subsection":   "",
        "page_number":  page,
        "heading_path": heading_path,
        "chunk_type":   "text",
        "token_count":  20,
        "chunk_index":  idx,
    }


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_path=str(tmp_path / "chroma"))


class TestExpandChunkToSection:

    def test_single_window_section_passthrough(self, store):
        """A section with just one window still returns a valid merge shape."""
        chunks = [_chunk(0, "Section A", "Only one window here.")]
        store.add(chunks)
        hit = store.search([0.0] * 10, top_k=1)[0]

        result = expand_chunk_to_section(hit, store=store)
        assert result["merged_chunk_count"] == 1
        assert "Only one window here." in result["text"]

    def test_merges_contiguous_same_section_windows(self, store):
        """Three windows of the same section merge into one text block."""
        chunks = [
            _chunk(0, "Section A", "First part of the section."),
            _chunk(1, "Section A", "Second part of the section."),
            _chunk(2, "Section A", "Third part of the section."),
        ]
        store.add(chunks)
        hit = store.search([0.0] * 10, top_k=1, filters={"chunk_index": 1})[0]

        result = expand_chunk_to_section(hit, store=store)
        assert result["merged_chunk_count"] == 3
        assert "First part" in result["text"]
        assert "Second part" in result["text"]
        assert "Third part" in result["text"]

    def test_stops_at_next_topic(self, store):
        """Expansion must not pull in chunks from a different heading_path."""
        chunks = [
            _chunk(0, "Section A", "Content of section A."),
            _chunk(1, "Section A", "More content of section A."),
            _chunk(2, "Section B", "Content of section B — a new topic."),
        ]
        store.add(chunks)
        hit = store.search([0.0] * 10, top_k=1, filters={"chunk_index": 0})[0]

        result = expand_chunk_to_section(hit, store=store)
        assert result["merged_chunk_count"] == 2
        assert "section A" in result["text"]
        assert "section B" not in result["text"]

    def test_removes_duplicated_overlap_words(self, store):
        """Sentence overlap between adjacent windows must not be duplicated."""
        chunks = [
            _chunk(0, "Section A", "The cat sat on the mat quietly today."),
            _chunk(1, "Section A", "quietly today. Then it slept soundly."),
        ]
        store.add(chunks)
        hit = store.search([0.0] * 10, top_k=1, filters={"chunk_index": 0})[0]

        result = expand_chunk_to_section(hit, store=store)
        assert result["text"].count("quietly today") == 1

    def test_heading_prefix_appears_once(self, store):
        """The 'heading_path: ' prefix must not repeat per merged window."""
        chunks = [
            _chunk(0, "Section A", "First sentence."),
            _chunk(1, "Section A", "Second sentence."),
        ]
        store.add(chunks)
        hit = store.search([0.0] * 10, top_k=1, filters={"chunk_index": 0})[0]

        result = expand_chunk_to_section(hit, store=store)
        assert result["text"].count("Section A:") == 1

    def test_pages_collected_across_merged_chunks(self, store):
        chunks = [
            _chunk(0, "Section A", "Page one content.", page=5),
            _chunk(1, "Section A", "Page two content.", page=6),
        ]
        store.add(chunks)
        hit = store.search([0.0] * 10, top_k=1, filters={"chunk_index": 0})[0]

        result = expand_chunk_to_section(hit, store=store)
        assert result["pages"] == [5, 6]

    def test_size_cap_is_respected(self, store):
        """A pathologically long section is trimmed to the token cap."""
        big_text = " ".join(f"word{i}" for i in range(4000))
        chunks = [_chunk(0, "Huge Section", big_text)]
        store.add(chunks)
        hit = store.search([0.0] * 10, top_k=1)[0]

        result = expand_chunk_to_section(hit, store=store)
        assert result["token_count"] <= MAX_EXPANDED_TOKENS

    def test_missing_chunk_index_falls_back_to_passthrough(self, store):
        chunk = _chunk(0, "Section A", "Some text.")
        del chunk["chunk_index"]  # simulate an older/incomplete chunk
        # embed manually to avoid ChromaStore.add() requiring chunk_index
        chunk["chunk_index"] = 0
        store.add([chunk])
        hit = store.search([0.0] * 10, top_k=1)[0]
        hit.pop("chunk_index")

        result = expand_chunk_to_section(hit, store=store)
        assert result["merged_chunk_count"] == 1


class TestExpandHits:

    def test_deduplicates_hits_from_same_section(self, store):
        """Two similarity hits landing in the same section expand only once."""
        chunks = [
            _chunk(0, "Section A", "First window of section A."),
            _chunk(1, "Section A", "Second window of section A."),
        ]
        store.add(chunks)
        hits = store.search([0.0] * 10, top_k=2)

        expanded = expand_hits(hits, store=store)
        assert len(expanded) == 1
        assert expanded[0]["merged_chunk_count"] == 2

    def test_expands_hits_from_different_sections_separately(self, store):
        chunks = [
            _chunk(0, "Section A", "Content A."),
            _chunk(1, "Section B", "Content B."),
        ]
        store.add(chunks)
        hits = store.search([0.0] * 10, top_k=2)

        expanded = expand_hits(hits, store=store)
        assert len(expanded) == 2
