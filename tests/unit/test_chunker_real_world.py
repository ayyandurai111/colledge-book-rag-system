"""
Regression tests covering real-world textbook patterns discovered while
testing against an actual college study-material PDF (Kamaraj College,
Principles of Management — flat-font headings, repeated page banners,
quote-led definition subsections). These patterns broke the chunker in
ways the earlier synthetic fixtures didn't catch.
"""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.chunker import (
    chunk_document, _is_heading_line, _detect_running_headers,
    _detect_chunk_type, _UNIT_CHAPTER_RE
)


# ─── UNIT/CHAPTER pattern detection (font-size-independent headings) ────────
class TestUnitChapterPattern:
    def test_unit_roman_numeral(self):
        assert _UNIT_CHAPTER_RE.match("UNIT I")
        assert _UNIT_CHAPTER_RE.match("UNIT III")
        assert _UNIT_CHAPTER_RE.match("UNIT V")

    def test_unit_with_dash(self):
        assert _UNIT_CHAPTER_RE.match("UNIT – II")
        assert _UNIT_CHAPTER_RE.match("UNIT - 2")

    def test_chapter_arabic_numeral(self):
        assert _UNIT_CHAPTER_RE.match("CHAPTER 3")
        assert _UNIT_CHAPTER_RE.match("Chapter 12")

    def test_module_pattern(self):
        assert _UNIT_CHAPTER_RE.match("MODULE IV")

    def test_does_not_match_body_text(self):
        assert not _UNIT_CHAPTER_RE.match("This unit covers planning.")

    def test_heading_detection_returns_chapter_at_body_font_size(self):
        # Real bug: "UNIT I" rendered at the SAME font size as body text
        # (common in Word-exported study materials) must still be detected
        # as a chapter heading via the structural pattern, not font size.
        level = _is_heading_line("UNIT I", font_size=12.0, is_bold=True, body_size=12.0)
        assert level == "chapter"


# ─── Quote-led prose must not be misread as a subsection heading ───────────
class TestQuoteProseGuard:
    def test_bolded_quote_lead_not_a_heading(self):
        # "According to Peter Drucker, "Management is..."" is bolded body
        # text (a quote attribution), not a heading — must not be captured.
        text = 'According to Peter Drucker, "Management is a multi-purpose organ."'
        level = _is_heading_line(text, font_size=12.0, is_bold=True, body_size=12.0)
        assert level is None

    def test_short_bold_label_is_still_a_heading(self):
        # Genuine short bold subheadings ("Definition", "Meaning") must
        # still be detected.
        level = _is_heading_line("Definition", font_size=12.0, is_bold=True, body_size=12.0)
        assert level == "subsection"


# ─── Running header/footer detection ────────────────────────────────────────
class TestRunningHeaderDetection:
    def _make_pages(self, n=10, banner="STUDY MATERIAL FOR BBA"):
        return [
            {
                "page": i,
                "blocks": [
                    {"text": banner, "font_size": 11.0, "is_bold": True,
                     "bbox": (200.0, 40.0, 380.0, 52.0)},   # fixed position every page
                    {"text": f"Unique content for page {i} discussing topic {i}.",
                     "font_size": 12.0, "is_bold": False,
                     "bbox": (72.0, 100.0 + i * 5, 500.0, 120.0 + i * 5)},
                ]
            }
            for i in range(1, n + 1)
        ]

    def test_detects_short_repeated_banner(self):
        pages = self._make_pages()
        headers = _detect_running_headers(pages)
        assert "STUDY MATERIAL FOR BBA" in headers

    def test_does_not_flag_unique_content(self):
        pages = self._make_pages()
        headers = _detect_running_headers(pages)
        assert not any("UNIQUE CONTENT" in h for h in headers)

    def test_long_repeated_paragraph_not_flagged(self):
        # Regression: a long body paragraph that happens to repeat
        # verbatim across pages (e.g. test fixtures, boilerplate)
        # must NOT be treated as a header — headers are short banners.
        long_text = "This is a long repeated paragraph. " * 20  # >100 chars
        pages = [
            {
                "page": i,
                "blocks": [{"text": long_text, "font_size": 12.0, "is_bold": False, "bbox": []}]
            }
            for i in range(1, 6)
        ]
        headers = _detect_running_headers(pages)
        assert long_text.strip().upper() not in headers

    def test_low_repeat_count_not_flagged(self):
        # A line appearing on only 1-2 pages out of many shouldn't be
        # treated as a running header.
        pages = self._make_pages(n=10)
        pages[0]["blocks"].append(
            {"text": "ONE OFF NOTE", "font_size": 10.0, "is_bold": False, "bbox": []}
        )
        headers = _detect_running_headers(pages)
        assert "ONE OFF NOTE" not in headers

    def test_same_text_varying_position_not_flagged_as_banner(self):
        # Regression: a short label that repeats verbatim across pages
        # (e.g. a "Definition" subsection heading appearing in every
        # unit) but at a DIFFERENT y-position each time — because it
        # follows a different amount of preceding content — must NOT
        # be treated as a running header. Real banners sit at a fixed
        # position; recurring structural headings don't.
        pages = []
        for i in range(1, 6):
            pages.append({
                "page": i,
                "blocks": [
                    {"text": "Definition", "font_size": 12.0, "is_bold": True,
                     "bbox": (72.0, 100.0 + i * 40, 200.0, 115.0 + i * 40)},
                ]
            })
        headers = _detect_running_headers(pages)
        assert "DEFINITION" not in headers

    def test_same_text_fixed_position_flagged_as_banner(self):
        # Contrast case: identical text AND identical position on every
        # page is the genuine signature of a running header/footer.
        pages = []
        for i in range(1, 6):
            pages.append({
                "page": i,
                "blocks": [
                    {"text": "CONFIDENTIAL DRAFT", "font_size": 9.0, "is_bold": False,
                     "bbox": (250.0, 770.0, 350.0, 780.0)},  # same footer position
                ]
            })
        headers = _detect_running_headers(pages)
        assert "CONFIDENTIAL DRAFT" in headers


# ─── Heading-aware chunk type classification ────────────────────────────────
class TestHeadingAwareChunkType:
    def test_quote_definition_classified_via_heading_label(self):
        # The prose itself doesn't say "Definition:" — only the heading
        # label does. Real textbooks structure content this way.
        text = 'According to Harold Koontz, "Management is the art of getting things done."'
        result = _detect_chunk_type(text, heading_label="Definition")
        assert result == "definition"

    def test_theorem_heading_label(self):
        text = "For all x in the domain, f(x) equals g(x) under these conditions."
        result = _detect_chunk_type(text, heading_label="Theorem 2.1")
        assert result == "definition"

    def test_no_heading_label_falls_back_to_text(self):
        text = "This is ordinary explanatory prose with no special markers."
        result = _detect_chunk_type(text, heading_label="Overview")
        assert result == "text"

    def test_prose_signal_still_takes_priority(self):
        # If the body text itself clearly states "Definition:", that
        # should win regardless of the heading label.
        text = "Definition: A formal statement of meaning."
        result = _detect_chunk_type(text, heading_label="Background")
        assert result == "definition"

    def test_formula_heading_label(self):
        text = "This expresses the relationship between force, mass and acceleration."
        result = _detect_chunk_type(text, heading_label="Formula")
        assert result == "formula"


# ─── End-to-end: flat-font textbook with UNIT structure ─────────────────────
class TestFlatFontTextbookEndToEnd:
    def _make_realistic_doc(self):
        """
        Simulates the real-world failure pattern: UNIT headings at the
        SAME font size as body text (only distinguished by being bold
        and matching the UNIT pattern), plus a repeated page banner at
        a FIXED position, plus a short quote-led "Definition" subsection
        that repeats across units but at a VARYING position (since it
        follows a different amount of preceding content each time) —
        exactly like the real PDF this pattern was discovered in.
        """
        banner = "STUDY MATERIAL FOR BBA"
        pages = []
        for i in range(1, 4):
            blocks = [
                {"text": banner, "font_size": 11.0, "is_bold": True,
                 "bbox": (200.0, 40.0, 380.0, 52.0)},  # fixed banner position
                {"text": f"UNIT {'I' * i}", "font_size": 12.0, "is_bold": True,
                 "bbox": (72.0, 90.0, 200.0, 105.0)},
                {"text": "Definition", "font_size": 12.0, "is_bold": True,
                 "bbox": (72.0, 130.0 + i * 20, 200.0, 145.0 + i * 20)},  # varies
                {"text": ('According to a noted scholar, "Management is the art of '
                          'getting things done through people in formally organised '
                          'groups, applying coordination and administration to achieve '
                          'goals across the whole organisation." ') * 3,
                 "font_size": 12.0, "is_bold": False, "bbox": []},
            ]
            pages.append({"page": i, "blocks": blocks})
        return {"source": "real_world.pdf", "pdf_type": "font", "toc": [], "pages": pages}

    def test_chapters_correctly_separated(self):
        doc    = self._make_realistic_doc()
        chunks = chunk_document(doc)
        chapters = {c["chapter"] for c in chunks}
        assert "UNIT I" in chapters
        assert "UNIT II" in chapters
        assert "UNIT III" in chapters

    def test_banner_not_leaked_into_chunk_text(self):
        doc    = self._make_realistic_doc()
        chunks = chunk_document(doc)
        for c in chunks:
            assert "STUDY MATERIAL FOR BBA" not in c["text"]

    def test_definition_subsection_classified_correctly(self):
        doc    = self._make_realistic_doc()
        chunks = chunk_document(doc)
        def_chunks = [c for c in chunks if c["subsection"] == "Definition"]
        assert len(def_chunks) > 0
        for c in def_chunks:
            assert c["chunk_type"] == "definition"
