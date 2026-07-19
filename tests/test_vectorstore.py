import pytest

from college_rag.exceptions import EmptyIndexError, IndexNotBuiltError
from college_rag.models import Chunk
from college_rag.vectorstore.faiss_store import FaissVectorStore


def _sample_chunks():
    return [
        Chunk(text="Newton force mass acceleration inertia motion.",
              source_file="physics.pdf", heading="Mechanics", page_number=1, chunk_id=0),
        Chunk(text="Entropy heat thermodynamics disorder temperature.",
              source_file="physics.pdf", heading="Thermo", page_number=2, chunk_id=1),
        Chunk(text="Cell organism life division biology basics.",
              source_file="bio.pdf", heading="Biology", page_number=1, chunk_id=2),
    ]


class TestFaissVectorStore:
    def test_search_before_build_raises(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        with pytest.raises(IndexNotBuiltError):
            store.search("hello")

    def test_build_with_empty_chunks_raises(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        with pytest.raises(EmptyIndexError):
            store.build([])

    def test_build_and_search_returns_most_relevant(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())

        results = store.search("What is entropy and heat?", top_k=1)
        assert len(results) == 1
        assert "entropy" in results[0].chunk.text.lower()

    def test_search_top_k_capped_to_available_chunks(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        # Requesting more than exist should not error, just return what's available
        results = store.search("Newton force", top_k=100)
        assert len(results) == 3

    def test_search_empty_query_returns_empty_list(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        assert store.search("   ", top_k=5) == []

    def test_scores_are_within_valid_range(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        results = store.search("cell biology life", top_k=3)
        for r in results:
            assert -1.0 <= r.score <= 1.0

    def test_save_and_load_roundtrip(self, fake_embedder, tmp_path):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        save_dir = str(tmp_path / "my_index")
        store.save(save_dir)

        loaded_store = FaissVectorStore(fake_embedder)
        loaded_store.load(save_dir)

        assert len(loaded_store.chunks) == len(store.chunks)
        results = loaded_store.search("entropy heat", top_k=1)
        assert "entropy" in results[0].chunk.text.lower()

    def test_save_without_build_raises(self, fake_embedder, tmp_path):
        store = FaissVectorStore(fake_embedder)
        with pytest.raises(IndexNotBuiltError):
            store.save(str(tmp_path / "idx"))

    def test_load_missing_folder_raises(self, fake_embedder, tmp_path):
        store = FaissVectorStore(fake_embedder)
        with pytest.raises(FileNotFoundError):
            store.load(str(tmp_path / "does_not_exist"))

    def test_add_incremental_chunks(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        chunks = _sample_chunks()
        store.build(chunks[:2])
        assert len(store.chunks) == 2

        store.add(chunks[2:])
        assert len(store.chunks) == 3
        results = store.search("cell biology life", top_k=1)
        assert "cell" in results[0].chunk.text.lower()

    def test_add_to_unbuilt_store_builds_it(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.add(_sample_chunks())
        assert store.is_built
        assert len(store.chunks) == 3
