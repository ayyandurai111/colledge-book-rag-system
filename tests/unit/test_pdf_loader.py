import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import fitz
from src.ingestion.pdf_loader import load_pdf, load_pdfs_from_dir, _extract_blocks


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _make_pdf(tmp_path, filename="test.pdf", pages=1, with_toc=False):
    path = tmp_path / filename
    doc  = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72),  f"Chapter {i+1}", fontsize=18)
        page.insert_text((72, 120), f"Section {i+1}.1", fontsize=14)
        page.insert_text((72, 160),
            "This is body text for the page. " * 10, fontsize=12)
    if with_toc:
        doc.set_toc([[1, "Chapter 1", 1], [2, "Section 1.1", 1]])
    doc.save(str(path))
    doc.close()
    return str(path)


# ─── Error handling ───────────────────────────────────────────────────────────
class TestLoadPdfErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_pdf("no_such_file.pdf")

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        with pytest.raises(ValueError):
            load_pdf(str(f))


# ─── Return structure ─────────────────────────────────────────────────────────
class TestLoadPdfStructure:
    def test_returns_dict_with_required_keys(self, tmp_path):
        path   = _make_pdf(tmp_path)
        result = load_pdf(path)
        assert set(result.keys()) >= {"source", "pdf_type", "toc", "pages"}

    def test_source_is_filename(self, tmp_path):
        path   = _make_pdf(tmp_path, filename="mybook.pdf")
        result = load_pdf(path)
        assert result["source"] == "mybook.pdf"

    def test_pages_is_list(self, tmp_path):
        path   = _make_pdf(tmp_path, pages=3)
        result = load_pdf(path)
        assert isinstance(result["pages"], list)
        assert len(result["pages"]) == 3

    def test_page_has_required_fields(self, tmp_path):
        path   = _make_pdf(tmp_path)
        result = load_pdf(path)
        page   = result["pages"][0]
        assert "page" in page
        assert "text" in page
        assert "blocks" in page

    def test_page_numbers_start_at_one(self, tmp_path):
        path   = _make_pdf(tmp_path, pages=2)
        result = load_pdf(path)
        nums   = [p["page"] for p in result["pages"]]
        assert nums[0] == 1

    def test_pages_have_text(self, tmp_path):
        path   = _make_pdf(tmp_path)
        result = load_pdf(path)
        for page in result["pages"]:
            assert len(page["text"].strip()) > 0


# ─── PDF type detection ───────────────────────────────────────────────────────
class TestPdfTypeDetection:
    def test_font_type_when_no_toc(self, tmp_path):
        path   = _make_pdf(tmp_path, with_toc=False)
        result = load_pdf(path)
        assert result["pdf_type"] == "font"
        assert result["toc"] == []

    def test_toc_type_when_bookmarks_exist(self, tmp_path):
        path   = _make_pdf(tmp_path, with_toc=True)
        result = load_pdf(path)
        assert result["pdf_type"] == "toc"
        assert len(result["toc"]) >= 1


# ─── Block extraction ─────────────────────────────────────────────────────────
class TestExtractBlocks:
    def test_blocks_have_required_fields(self, tmp_path):
        path   = _make_pdf(tmp_path)
        result = load_pdf(path)
        for block in result["pages"][0]["blocks"]:
            assert "text"      in block
            assert "font_size" in block
            assert "is_bold"   in block
            assert "bbox"      in block

    def test_font_size_positive(self, tmp_path):
        path   = _make_pdf(tmp_path)
        result = load_pdf(path)
        for block in result["pages"][0]["blocks"]:
            assert block["font_size"] > 0

    def test_is_bold_is_bool(self, tmp_path):
        path   = _make_pdf(tmp_path)
        result = load_pdf(path)
        for block in result["pages"][0]["blocks"]:
            assert isinstance(block["is_bold"], bool)

    def test_different_font_sizes_detected(self, tmp_path):
        path   = _make_pdf(tmp_path)
        result = load_pdf(path)
        sizes  = {b["font_size"] for b in result["pages"][0]["blocks"]}
        assert len(sizes) > 1   # heading (18pt) vs body (12pt) vs section (14pt)


# ─── Directory loading ────────────────────────────────────────────────────────
class TestLoadPdfsFromDir:
    def test_loads_multiple_pdfs(self, tmp_path):
        _make_pdf(tmp_path, "book1.pdf")
        _make_pdf(tmp_path, "book2.pdf")
        docs = load_pdfs_from_dir(str(tmp_path))
        assert len(docs) == 2

    def test_empty_directory(self, tmp_path):
        docs = load_pdfs_from_dir(str(tmp_path))
        assert docs == []

    def test_ignores_non_pdfs(self, tmp_path):
        _make_pdf(tmp_path, "book.pdf")
        (tmp_path / "notes.txt").write_text("ignore me")
        docs = load_pdfs_from_dir(str(tmp_path))
        assert len(docs) == 1
