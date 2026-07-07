import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.vectorstore.chroma_store import ChromaStore


def _make_chunks(n=5, dim=10):
    return [
        {
            "chunk_id":    f"test.pdf_chunk_{i:03d}",
            "text":        f"Chapter 1 > Section {i}: Sentence about topic {i}.",
            "embedding":   [float(i) / 100] * dim,
            "source":      "test.pdf",
            "chapter":     "Chapter 1",
            "section":     f"Section {i}",
            "subsection":  "",
            "page_number": i + 1,
            "heading_path": f"Chapter 1 > Section {i}",
            "chunk_type":  "text",
            "token_count": 20,
        }
        for i in range(n)
    ]


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_path=str(tmp_path / "chroma"))


class TestChromaStore:
    def test_initial_count_is_zero(self, store):
        assert store.count() == 0

    def test_add_increases_count(self, store):
        store.add(_make_chunks(3))
        assert store.count() == 3

    def test_upsert_no_duplicates(self, store):
        chunks = _make_chunks(3)
        store.add(chunks)
        store.add(chunks)   # same IDs — should upsert not duplicate
        assert store.count() == 3

    def test_search_returns_results(self, store):
        store.add(_make_chunks(5))
        query_vec = [0.02] * 10
        results   = store.search(query_vec, top_k=3)
        assert len(results) == 3

    def test_search_result_has_all_metadata_fields(self, store):
        store.add(_make_chunks(3))
        results = store.search([0.01] * 10, top_k=1)
        chunk   = results[0]
        for field in ["text","source","chapter","section","subsection",
                      "page_number","heading_path","chunk_type","token_count","score"]:
            assert field in chunk, f"Missing field: {field}"

    def test_search_score_between_0_and_1(self, store):
        store.add(_make_chunks(5))
        results = store.search([0.01] * 10, top_k=5)
        for r in results:
            assert 0.0 <= r["score"] <= 1.0

    def test_search_with_source_filter(self, store):
        chunks = _make_chunks(3)
        chunks[0]["source"] = "other.pdf"
        store.add(chunks)
        results = store.search([0.01]*10, top_k=5, filters={"source": "test.pdf"})
        for r in results:
            assert r["source"] == "test.pdf"

    def test_search_by_type_definition(self, store):
        chunks = _make_chunks(4)
        chunks[0]["chunk_type"] = "definition"
        chunks[1]["chunk_type"] = "definition"
        store.add(chunks)
        results = store.search_by_type([0.01]*10, chunk_type="definition", top_k=5)
        for r in results:
            assert r["chunk_type"] == "definition"

    def test_reset_empties_store(self, store):
        store.add(_make_chunks(5))
        assert store.count() == 5
        store.reset()
        assert store.count() == 0

    def test_add_empty_list_is_noop(self, store):
        store.add([])
        assert store.count() == 0

    def test_metadata_stored_and_retrieved_correctly(self, store):
        chunk = _make_chunks(1)[0]
        chunk["chapter"]     = "Chapter 7 Thermodynamics"
        chunk["section"]     = "7.2 Entropy"
        chunk["page_number"] = 99
        store.add([chunk])
        results = store.search([0.0]*10, top_k=1)
        r = results[0]
        assert r["chapter"]     == "Chapter 7 Thermodynamics"
        assert r["section"]     == "7.2 Entropy"
        assert r["page_number"] == 99
