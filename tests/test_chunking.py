import pytest

from college_rag.chunking.semantic_chunker import SemanticChunker, split_sentences
from college_rag.models import TextBlock


class TestSplitSentences:
    def test_splits_on_period(self):
        assert split_sentences("Hello world. This is a test.") == [
            "Hello world.", "This is a test.",
        ]

    def test_handles_tamil_sentence_terminator(self):
        result = split_sentences("இது ஒரு வாக்கியம்। இது இரண்டாவது வாக்கியம்।")
        assert len(result) == 2

    def test_empty_string_returns_empty_list(self):
        assert split_sentences("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert split_sentences("    \n\n   ") == []

    def test_single_sentence_no_terminator(self):
        assert split_sentences("just some words with no ending") == [
            "just some words with no ending"
        ]


class TestSemanticChunker:
    def test_rejects_invalid_size_bounds(self, fake_embedder):
        with pytest.raises(ValueError):
            SemanticChunker(fake_embedder, min_chunk_chars=500, max_chunk_chars=100)

    def test_single_sentence_block_returns_one_chunk(self, fake_embedder):
        chunker = SemanticChunker(fake_embedder, min_chunk_chars=10, max_chunk_chars=1000)
        block = TextBlock(text="Just one sentence here", source_file="x.pdf")
        chunks = chunker.chunk_block(block)
        assert len(chunks) == 1
        assert chunks[0].text == "Just one sentence here"

    def test_splits_at_topic_boundary(self, fake_embedder):
        # First two sentences are about mechanics, last two about thermodynamics.
        # A good semantic chunker should NOT merge these into a single chunk.
        text = (
            "Newton's First Law states that an object remains at rest. "
            "Force equals mass times acceleration in Newton's Second Law. "
            "Entropy always increases in an isolated thermodynamic system. "
            "Heat flows from hot objects to cold objects naturally."
        )
        block = TextBlock(text=text, source_file="physics.docx", heading="Ch1")
        chunker = SemanticChunker(
            fake_embedder, min_chunk_chars=10, max_chunk_chars=1000, breakpoint_percentile=50,
        )
        chunks = chunker.chunk_block(block)

        assert len(chunks) >= 2
        assert "newton" in chunks[0].text.lower() or "force" in chunks[0].text.lower()
        assert "entropy" in chunks[-1].text.lower() or "heat" in chunks[-1].text.lower()

    def test_respects_max_chunk_chars(self, fake_embedder):
        # Many similar (mechanics) sentences that would otherwise merge into one
        # chunk based on similarity alone — max size must force a split.
        sentence = "Newton's force and mass and acceleration are related concepts. "
        text = sentence * 20
        block = TextBlock(text=text, source_file="x.pdf")
        chunker = SemanticChunker(fake_embedder, min_chunk_chars=10, max_chunk_chars=200)
        chunks = chunker.chunk_block(block)

        assert len(chunks) > 1
        for c in chunks:
            # Allow the single-sentence overflow case, but chunks built from
            # more than one sentence must not wildly exceed the max.
            assert len(c.text) <= 200 + len(sentence)

    def test_chunk_ids_are_unique_across_blocks(self, fake_embedder):
        chunker = SemanticChunker(fake_embedder, min_chunk_chars=10, max_chunk_chars=1000)
        blocks = [
            TextBlock(text="Newton force mass acceleration inertia motion.", source_file="a.pdf"),
            TextBlock(text="Entropy heat thermodynamics disorder temperature.", source_file="b.pdf"),
        ]
        chunks = chunker.chunk_blocks(blocks)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk_id values must be unique"

    def test_preserves_source_metadata(self, fake_embedder):
        chunker = SemanticChunker(fake_embedder, min_chunk_chars=10, max_chunk_chars=1000)
        block = TextBlock(
            text="Some sentence here about cells and life.",
            source_file="bio.pdf", page_number=3, heading="Ch2",
        )
        chunks = chunker.chunk_block(block)
        assert chunks[0].source_file == "bio.pdf"
        assert chunks[0].page_number == 3
        assert chunks[0].heading == "Ch2"

    def test_empty_blocks_list_returns_empty(self, fake_embedder):
        chunker = SemanticChunker(fake_embedder, min_chunk_chars=10, max_chunk_chars=1000)
        assert chunker.chunk_blocks([]) == []
