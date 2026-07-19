import pytest

from college_rag.config import Config
from college_rag.exceptions import EmptyDocumentError
from college_rag.pipeline import RAGPipeline


@pytest.fixture
def pipeline(fake_embedder):
    config = Config(min_chunk_chars=10, max_chunk_chars=500, breakpoint_percentile=60,
                     default_top_k=3)
    return RAGPipeline(config=config, embedder=fake_embedder)


class TestRAGPipelineEndToEnd:
    def test_build_index_from_pdf_and_docx(self, pipeline, sample_docx_path, sample_pdf_path):
        stats = pipeline.build_index_from_files([sample_docx_path, sample_pdf_path])
        assert stats.total_chunks > 0
        assert stats.total_source_files == 2

    def test_query_retrieves_relevant_chunk(self, pipeline, sample_docx_path, sample_pdf_path):
        pipeline.build_index_from_files([sample_docx_path, sample_pdf_path])
        results = pipeline.query("What is entropy in thermodynamics?")

        assert len(results) > 0
        assert "entropy" in results[0].chunk.text.lower()

    def test_query_biology_topic_from_pdf(self, pipeline, sample_docx_path, sample_pdf_path):
        pipeline.build_index_from_files([sample_docx_path, sample_pdf_path])
        results = pipeline.query("Tell me about DNA and genes")
        assert any("dna" in r.chunk.text.lower() or "gene" in r.chunk.text.lower() for r in results)

    def test_query_respects_top_k(self, pipeline, sample_docx_path, sample_pdf_path):
        pipeline.build_index_from_files([sample_docx_path, sample_pdf_path])
        results = pipeline.query("physics and biology", top_k=1)
        assert len(results) == 1

    def test_build_with_no_valid_files_raises(self, pipeline, tmp_path):
        bad_file = tmp_path / "empty.docx"
        bad_file.write_text("not a real docx")
        with pytest.raises(EmptyDocumentError):
            pipeline.build_index_from_files([str(bad_file)])

    def test_partial_failure_still_builds_from_good_files(self, pipeline, sample_docx_path, tmp_path):
        bad_file = tmp_path / "corrupt.docx"
        bad_file.write_text("not a real docx")
        stats = pipeline.build_index_from_files([sample_docx_path, str(bad_file)])
        assert stats.total_chunks > 0
        assert stats.total_source_files == 1

    def test_save_and_reload_index_preserves_query_results(self, pipeline, sample_docx_path, tmp_path):
        pipeline.build_index_from_files([sample_docx_path])
        save_dir = str(tmp_path / "idx")
        pipeline.save_index(save_dir)

        new_pipeline = RAGPipeline(config=pipeline.config, embedder=pipeline.embedder)
        new_pipeline.load_index(save_dir)

        results = new_pipeline.query("Newton's laws of motion")
        assert len(results) > 0

    def test_stats_lists_source_files(self, pipeline, sample_docx_path, sample_pdf_path):
        pipeline.build_index_from_files([sample_docx_path, sample_pdf_path])
        stats = pipeline.stats()
        assert len(stats.source_files) == 2

    def test_query_on_empty_index_returns_empty(self, fake_embedder, sample_docx_path, tmp_path):
        # A freshly-loaded index with a query that matches nothing relevant
        # should not error — it should simply return whatever is closest,
        # or an empty list if the store itself is empty.
        config = Config(min_chunk_chars=10, max_chunk_chars=500, default_top_k=3)
        p = RAGPipeline(config=config, embedder=fake_embedder)
        p.build_index_from_files([sample_docx_path])
        results = p.query("   ")  # blank query
        assert results == []

