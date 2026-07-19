import os

import pytest

from college_rag.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from college_rag.ingestion.docx_extractor import DocxExtractor
from college_rag.ingestion.factory import extract
from college_rag.ingestion.pdf_extractor import PDFExtractor


class TestDocxExtractor:
    def test_extracts_headings_and_text(self, sample_docx_path):
        blocks = DocxExtractor().extract(sample_docx_path)
        assert len(blocks) == 2
        assert blocks[0].heading == "Chapter 1: Mechanics"
        assert "Newton's First Law" in blocks[0].text
        assert blocks[1].heading == "Chapter 2: Thermodynamics"
        assert "entropy" in blocks[1].text.lower()

    def test_source_file_name_recorded(self, sample_docx_path):
        blocks = DocxExtractor().extract(sample_docx_path)
        assert blocks[0].source_file == os.path.basename(sample_docx_path)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            DocxExtractor().extract("/tmp/does_not_exist_12345.docx")

    def test_corrupt_file_raises_empty_document_error(self, tmp_path):
        bad_file = tmp_path / "corrupt.docx"
        bad_file.write_text("this is not a real docx file")
        with pytest.raises(EmptyDocumentError):
            DocxExtractor().extract(str(bad_file))


class TestPDFExtractor:
    def test_extracts_headings_and_text(self, sample_pdf_path):
        blocks = PDFExtractor().extract(sample_pdf_path)
        assert len(blocks) >= 2
        headings = [b.heading for b in blocks]
        assert any("BIOLOGY" in h for h in headings)
        assert any("GENETICS" in h for h in headings)

    def test_page_numbers_recorded(self, sample_pdf_path):
        blocks = PDFExtractor().extract(sample_pdf_path)
        assert all(b.page_number is not None for b in blocks)
        assert blocks[0].page_number == 1

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            PDFExtractor().extract("/tmp/does_not_exist_12345.pdf")

    def test_corrupt_file_raises_empty_document_error(self, tmp_path):
        bad_file = tmp_path / "corrupt.pdf"
        bad_file.write_bytes(b"%PDF-1.4 not a real pdf structure")
        with pytest.raises(EmptyDocumentError):
            PDFExtractor().extract(str(bad_file))


class TestFactory:
    def test_routes_docx_correctly(self, sample_docx_path):
        blocks = extract(sample_docx_path)
        assert len(blocks) > 0

    def test_routes_pdf_correctly(self, sample_pdf_path):
        blocks = extract(sample_pdf_path)
        assert len(blocks) > 0

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("some text")
        with pytest.raises(UnsupportedFileTypeError):
            extract(str(f))

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract("/tmp/totally_missing_file_98765.pdf")
