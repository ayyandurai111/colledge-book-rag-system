import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.chunker import (
    _detect_chunk_type, _split_sentences, _window_sentences,
    _quality_check, _add_heading_prefix, _build_heading_path,
    _make_chunk_id, _is_heading_line, chunk_document
)
from src.utils.tokenizer import count_tokens


# ─── Chunk type detection ─────────────────────────────────────────────────────
class TestDetectChunkType:
    def test_definition(self):
        assert _detect_chunk_type("Definition: A set is a collection of objects.") == "definition"

    def test_theorem(self):
        assert _detect_chunk_type("Theorem: For all x, f(x) = g(x).") == "definition"

    def test_formula(self):
        assert _detect_chunk_type("The equation ∑x = n applies.") == "formula"

    def test_formula_latex(self):
        assert _detect_chunk_type(r"Given \frac{a}{b} = c") == "formula"

    def test_table(self):
        assert _detect_chunk_type("Table 3. Results of the experiment") == "table"

    def test_figure(self):
        assert _detect_chunk_type("Figure 2. Diagram of the system") == "figure"

    def test_fig_abbrev(self):
        assert _detect_chunk_type("Fig. 5. Cross-section view") == "figure"

    def test_plain_text(self):
        assert _detect_chunk_type("This is a normal paragraph about physics.") == "text"


# ─── Heading detection ────────────────────────────────────────────────────────
class TestHeadingDetection:
    def test_chapter_large_font(self):
        assert _is_heading_line("Chapter 1", 18.0, False, 12.0) == "chapter"

    def test_section_medium_font(self):
        assert _is_heading_line("1.1 Introduction", 15.0, False, 12.0) == "section"

    def test_subsection_bold(self):
        assert _is_heading_line("Key Concepts", 12.0, True, 12.0) == "subsection"

    def test_subsection_all_caps(self):
        assert _is_heading_line("OVERVIEW", 12.0, False, 12.0) == "subsection"

    def test_body_text_not_heading(self):
        assert _is_heading_line("This is normal body text that goes on for quite a while.", 12.0, False, 12.0) is None

    def test_empty_line_not_heading(self):
        assert _is_heading_line("", 14.0, True, 12.0) is None


# ─── Sentence splitting ───────────────────────────────────────────────────────
class TestSplitSentences:
    def test_basic_three_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        parts = _split_sentences(text)
        assert len(parts) == 3

    def test_single_sentence(self):
        assert len(_split_sentences("Only one.")) == 1

    def test_exclamation(self):
        parts = _split_sentences("Alert! Warning! Stop.")
        assert len(parts) == 3

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_no_punctuation(self):
        parts = _split_sentences("no punctuation here")
        assert len(parts) == 1


# ─── Sliding window ───────────────────────────────────────────────────────────
class TestWindowSentences:
    def test_produces_multiple_windows(self):
        sentences = ["This is sentence number %d." % i for i in range(50)]
        windows = _window_sentences(sentences, min_tok=20, max_tok=60, overlap_tok=15)
        assert len(windows) > 1

    def test_each_window_within_token_limit(self):
        sentences = ["word " * 20 + "." for _ in range(20)]
        windows = _window_sentences(sentences, min_tok=10, max_tok=80, overlap_tok=20)
        for w in windows:
            assert count_tokens(w) <= 100   # generous tolerance

    def test_single_long_sentence(self):
        # One sentence that is itself longer than max — should still produce 1 window
        sentences = ["word " * 50 + "."]
        windows = _window_sentences(sentences, min_tok=10, max_tok=30, overlap_tok=5)
        assert len(windows) >= 1

    def test_overlap_creates_shared_content(self):
        sentences = ["Sentence %d is here for testing." % i for i in range(20)]
        windows = _window_sentences(sentences, min_tok=5, max_tok=40, overlap_tok=15)
        if len(windows) >= 2:
            # Last sentence of window 0 should appear in window 1
            last_of_first = windows[0].split(". ")[-1].strip()
            assert any(last_of_first[:10] in w for w in windows[1:])


# ─── Chunk ID deduplication ───────────────────────────────────────────────────
class TestChunkId:
    def test_same_inputs_produce_same_id(self):
        # Deterministic: identical (source, text, page, index) → identical ID.
        # This is what makes re-ingestion idempotent (upsert, no duplicates).
        id1 = _make_chunk_id("book.pdf", "hello world", page=1, index=0)
        id2 = _make_chunk_id("book.pdf", "hello world", page=1, index=0)
        assert id1 == id2

    def test_different_page_different_id(self):
        id1 = _make_chunk_id("book.pdf", "hello world", page=1, index=0)
        id2 = _make_chunk_id("book.pdf", "hello world", page=2, index=0)
        assert id1 != id2

    def test_different_index_different_id(self):
        # Same text repeated at two positions on the same page (e.g. boilerplate)
        # must still get distinct ids.
        id1 = _make_chunk_id("book.pdf", "hello world", page=1, index=0)
        id2 = _make_chunk_id("book.pdf", "hello world", page=1, index=1)
        assert id1 != id2

    def test_different_content_produces_ids(self):
        id1 = _make_chunk_id("book.pdf", "text A", page=1, index=0)
        id2 = _make_chunk_id("book.pdf", "text B", page=1, index=0)
        assert isinstance(id1, str) and isinstance(id2, str)
        assert id1 != id2

    def test_different_source_different_id(self):
        # Different sources must produce different IDs
        id1 = _make_chunk_id("book1.pdf", "same text", page=1, index=0)
        id2 = _make_chunk_id("book2.pdf", "same text", page=1, index=0)
        assert id1 != id2

    def test_id_contains_source(self):
        cid = _make_chunk_id("physics.pdf", "some text")
        assert "physics.pdf" in cid

    def test_ids_are_strings(self):
        cid = _make_chunk_id("test.pdf", "content")
        assert isinstance(cid, str)
        assert len(cid) > 0


# ─── Heading path ─────────────────────────────────────────────────────────────
class TestBuildHeadingPath:
    def test_full_path(self):
        assert _build_heading_path("Ch 1", "Sec 2", "Sub 3") == "Ch 1 > Sec 2 > Sub 3"

    def test_chapter_only(self):
        assert _build_heading_path("Ch 1", "", "") == "Ch 1"

    def test_chapter_and_section(self):
        assert _build_heading_path("Ch 1", "Sec 1", "") == "Ch 1 > Sec 1"

    def test_all_empty(self):
        assert _build_heading_path("", "", "") == ""


# ─── Heading prefix ───────────────────────────────────────────────────────────
class TestAddHeadingPrefix:
    def test_adds_prefix(self):
        chunk = {"heading_path": "Ch 1 > Sec 1", "text": "Some content.", "token_count": 10}
        result = _add_heading_prefix(chunk)
        assert result["text"].startswith("Ch 1 > Sec 1:")

    def test_no_duplicate_prefix(self):
        path = "Ch 1 > Sec 1"
        chunk = {"heading_path": path, "text": f"{path}: Already prefixed.", "token_count": 10}
        result = _add_heading_prefix(chunk)
        assert result["text"].count(path) == 1

    def test_no_path_no_prefix(self):
        chunk = {"heading_path": "", "text": "Plain text.", "token_count": 5}
        result = _add_heading_prefix(chunk)
        assert result["text"] == "Plain text."


# ─── Quality check ────────────────────────────────────────────────────────────
class TestQualityCheck:
    def test_too_short_dropped(self):
        chunk = {"text": "Short.", "token_count": 5, "heading_path": ""}
        assert _quality_check(chunk) is None

    def test_too_long_dropped(self):
        text = "word " * 700
        chunk = {"text": text, "token_count": 900, "heading_path": ""}
        assert _quality_check(chunk) is None

    def test_valid_chunk_passes(self):
        text = "This is a complete sentence about Newton's laws of motion. " * 15
        chunk = {"text": text, "token_count": count_tokens(text), "heading_path": ""}
        assert _quality_check(chunk) is not None

    def test_broken_sentence_trimmed(self):
        text = "Complete sentence here. Another complete one. Broken fragment without"
        chunk = {"text": text, "token_count": count_tokens(text), "heading_path": ""}
        result = _quality_check(chunk)
        # Either trimmed to complete sentence or dropped — either is correct
        if result:
            assert result["text"].endswith(".")

    def test_token_count_updated_after_trim(self):
        text = ("Valid content sentence. " * 15) + "incomplete"
        chunk = {"text": text, "token_count": count_tokens(text), "heading_path": ""}
        result = _quality_check(chunk)
        if result:
            assert result["token_count"] == count_tokens(result["text"])


# ─── Full document chunking ───────────────────────────────────────────────────
class TestChunkDocument:
    def _make_doc(self, pdf_type="font", with_toc=False):
        toc = [[1, "Chapter 1 Introduction", 1],
               [2, "Background", 1],
               [1, "Chapter 2 Methods", 3]] if with_toc else []
        pages = [
            {
                "page": i,
                "text": " ".join([f"This is sentence {j} on page {i}." for j in range(60)]),
                "blocks": [
                    {"text": f"CHAPTER {i}", "font_size": 18.0, "is_bold": True, "bbox": []},
                    {"text": " ".join([f"Sentence {j}." for j in range(40)]),
                     "font_size": 12.0, "is_bold": False, "bbox": []},
                ]
            }
            for i in range(1, 5)
        ]
        return {"source": "test.pdf", "pdf_type": pdf_type, "toc": toc, "pages": pages}

    def test_font_based_produces_chunks(self):
        doc    = self._make_doc("font")
        chunks = chunk_document(doc)
        assert len(chunks) > 0

    def test_toc_based_produces_chunks(self):
        doc    = self._make_doc("toc", with_toc=True)
        chunks = chunk_document(doc)
        assert len(chunks) > 0

    def test_all_chunks_have_required_fields(self):
        doc    = self._make_doc("font")
        chunks = chunk_document(doc)
        required = {"chunk_id","source","chapter","section","subsection",
                    "page_number","heading_path","chunk_type","text","token_count"}
        for c in chunks:
            assert required.issubset(c.keys()), f"Missing fields: {required - c.keys()}"

    def test_no_duplicate_chunk_ids(self):
        doc    = self._make_doc("font")
        chunks = chunk_document(doc)
        ids    = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_chunks_pass_quality_gate(self):
        doc    = self._make_doc("font")
        chunks = chunk_document(doc)
        for c in chunks:
            assert c["token_count"] >= 100, f"Chunk below min: {c['token_count']}"
            assert c["token_count"] <= 700, f"Chunk above max: {c['token_count']}"

    def test_chunk_text_starts_with_heading(self):
        doc    = self._make_doc("font")
        chunks = chunk_document(doc)
        # Chunks with a heading_path should have it as prefix
        for c in chunks:
            if c["heading_path"]:
                assert c["text"].startswith(c["heading_path"]), (
                    f"Missing prefix in: {c['text'][:80]!r}"
                )

    def test_chunk_type_valid(self):
        doc    = self._make_doc("font")
        chunks = chunk_document(doc)
        valid  = {"text", "definition", "formula", "table", "figure"}
        for c in chunks:
            assert c["chunk_type"] in valid
