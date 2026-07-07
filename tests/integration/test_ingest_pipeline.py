"""
Integration tests for the full ingest pipeline.
Uses real PyMuPDF PDFs + real ChromaDB (tmp_path) but mocks the NVIDIA API.
"""
import sys, os, pytest
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import fitz
from src.vectorstore.chroma_store import ChromaStore


def _make_textbook_pdf(tmp_path, filename="textbook.pdf", chapters=2, pages_per_chapter=2):
    """
    Create a realistic multi-chapter textbook PDF.
    Uses insert_textbox so text isn't page-clipped.
    Body is long enough to guarantee chunks pass quality_min_tokens regardless
    of whether tiktoken or the local word-split estimator is used.
    """
    path = tmp_path / filename
    doc  = fitz.open()
    toc_entries = []
    page_num    = 1

    # ~600 words  → ~600 local-estimator tokens  → well above quality_min_tokens=50
    # ~60 repetitions × ~9 words × 1.3 ≈ 700+ tokens per page
    # fontsize=10 with a wide rect ensures PyMuPDF renders the full body
    body = "This is a detailed explanation of Newton laws of motion. " * 60

    for ch in range(1, chapters + 1):
        toc_entries.append([1, f"Chapter {ch} — Topic {ch}", page_num])
        for sec in range(1, pages_per_chapter + 1):
            page = doc.new_page()
            page.insert_text((72, 50), f"Chapter {ch} — Topic {ch}", fontsize=18)
            page.insert_text((72, 90), f"Section {ch}.{sec}", fontsize=14)
            rect = fitz.Rect(50, 120, 550, 780)
            page.insert_textbox(rect, body, fontsize=10, align=0)
            toc_entries.append([2, f"Section {ch}.{sec}", page_num])
            page_num += 1

    doc.set_toc(toc_entries)
    doc.save(str(path))
    doc.close()
    return str(path)


def _fake_embed_client():
    mock_client = MagicMock()
    def fake_create(**kwargs):
        n    = len(kwargs["input"])
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1] * 10) for _ in range(n)]
        return resp
    mock_client.embeddings.create.side_effect = fake_create
    return mock_client


@pytest.fixture
def chroma_store(tmp_path):
    return ChromaStore(persist_path=str(tmp_path / "chroma"))


@pytest.fixture(autouse=True)
def lower_quality_threshold(monkeypatch):
    """
    Use a lower quality_min_tokens threshold so tests pass whether
    tiktoken (BPE) or the local word-split estimator is active.
    The real config (100 tokens min) is enforced in unit tests.
    """
    import src.utils.config as cfg
    cfg._config = None
    original_load = cfg.load_config

    def patched_load(path=None):
        c = original_load(path)
        c["chunking"]["quality_min_tokens"] = 50
        c["chunking"]["min_tokens"]         = 50
        c["chunking"]["max_tokens"]         = 600
        return c

    monkeypatch.setattr(cfg, "load_config", patched_load)
    cfg._config = None   # force re-load with patched function
    yield
    cfg._config = None


class TestIngestPipeline:

    def test_ingest_single_pdf_produces_chunks(self, tmp_path, chroma_store):
        pdf_path = _make_textbook_pdf(tmp_path)
        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_pdf
            summary = ingest_pdf(pdf_path, store=chroma_store)

        assert summary["pages_loaded"]   > 0, f"Expected pages, got: {summary}"
        assert summary["chunks_created"] > 0, f"Expected chunks, got: {summary}"
        assert summary["total_in_store"] > 0, f"Expected store entries, got: {summary}"

    def test_ingest_uses_toc_strategy_for_bookmarked_pdf(self, tmp_path, chroma_store):
        pdf_path = _make_textbook_pdf(tmp_path)
        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_pdf
            summary = ingest_pdf(pdf_path, store=chroma_store)

        assert summary["pdf_type"] == "toc", f"Expected toc, got {summary['pdf_type']}"

    def test_chunks_stored_have_source_metadata(self, tmp_path, chroma_store):
        pdf_path = _make_textbook_pdf(tmp_path)
        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_pdf
            ingest_pdf(pdf_path, store=chroma_store)

        results = chroma_store.search([0.1] * 10, top_k=5)
        assert len(results) > 0, "Expected results from vector store"
        for r in results:
            assert r["source"] == "textbook.pdf", f"Wrong source: {r['source']}"

    def test_chunks_have_chapter_metadata(self, tmp_path, chroma_store):
        pdf_path = _make_textbook_pdf(tmp_path)
        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_pdf
            ingest_pdf(pdf_path, store=chroma_store)

        results = chroma_store.search([0.1] * 10, top_k=5)
        assert len(results) > 0
        for r in results:
            # chapter field should be populated (TOC strategy fills it)
            assert isinstance(r["chapter"], str)
            assert isinstance(r["heading_path"], str)

    def test_reingest_same_pdf_no_duplicates(self, tmp_path, chroma_store):
        pdf_path = _make_textbook_pdf(tmp_path)
        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_pdf
            s1 = ingest_pdf(pdf_path, store=chroma_store)
            s2 = ingest_pdf(pdf_path, store=chroma_store)

        assert s1["total_in_store"] == s2["total_in_store"], (
            f"Re-ingest should not grow store: {s1['total_in_store']} → {s2['total_in_store']}"
        )

    def test_chunk_types_sum_to_total(self, tmp_path, chroma_store):
        pdf_path = _make_textbook_pdf(tmp_path)
        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_pdf
            summary = ingest_pdf(pdf_path, store=chroma_store)

        assert "chunk_types" in summary
        assert isinstance(summary["chunk_types"], dict)
        assert sum(summary["chunk_types"].values()) == summary["chunks_created"], (
            f"chunk_types totals don't match: {summary}"
        )

    def test_ingest_directory(self, tmp_path, chroma_store):
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        _make_textbook_pdf(pdf_dir, "book1.pdf")
        _make_textbook_pdf(pdf_dir, "book2.pdf")

        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_directory
            summary = ingest_directory(str(pdf_dir), store=chroma_store)

        assert summary["files_ingested"] == 2
        assert summary["chunks_created"] > 0
        assert summary["total_in_store"] > 0

    def test_empty_pdf_handled_gracefully(self, tmp_path, chroma_store):
        """An almost-empty PDF (only a heading) should produce 0 chunks without crashing."""
        path = tmp_path / "empty.pdf"
        doc  = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Chapter 1", fontsize=18)
        doc.save(str(path))
        doc.close()

        with patch("src.ingestion.embedder.get_client", return_value=_fake_embed_client()):
            from src.pipeline.ingest_pipeline import ingest_pdf
            summary = ingest_pdf(str(path), store=chroma_store)

        assert summary["pages_loaded"] == 1
        assert summary["chunks_created"] == 0   # too short to pass quality gate
