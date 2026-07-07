import sys, os, pytest
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


FULL_CHUNK = {
    "text":         "Chapter 3 > Newton's Laws > Second Law: F = ma explains force.",
    "source":       "physics.pdf",
    "chapter":      "Chapter 3",
    "section":      "Newton's Laws",
    "subsection":   "Second Law",
    "page_number":  42,
    "heading_path": "Chapter 3 > Newton's Laws > Second Law",
    "chunk_type":   "formula",
    "token_count":  120,
    "score":        0.91
}


class TestRetrieve:
    def _mock_store(self, count=100, results=None):
        store = MagicMock()
        store.count.return_value = count
        store.search.return_value = results if results is not None else [FULL_CHUNK]
        store.search_by_type.return_value = results if results is not None else [FULL_CHUNK]
        return store

    def test_empty_store_returns_empty(self):
        store = self._mock_store(count=0)
        with patch("src.retrieval.retriever.get_store", return_value=store), \
             patch("src.retrieval.retriever.embed_query", return_value=[0.1]*10):
            from src.retrieval.retriever import retrieve
            assert retrieve("any question") == []

    def test_returns_chunks_with_full_metadata(self):
        store = self._mock_store()
        with patch("src.retrieval.retriever.get_store", return_value=store), \
             patch("src.retrieval.retriever.embed_query", return_value=[0.1]*10):
            import src.retrieval.retriever as r
            original_store = r._store
            r._store = store
            try:
                results = r.retrieve("What is Newton's Second Law?")
            finally:
                r._store = original_store
            assert len(results) == 1
            chunk = results[0]
            for field in ["chapter","section","subsection","heading_path","chunk_type","token_count"]:
                assert field in chunk, f"Missing field: {field}"

    def test_score_threshold_filters_low_scores(self):
        low_score_chunk = {**FULL_CHUNK, "score": 0.1}
        store = self._mock_store(results=[low_score_chunk])
        with patch("src.retrieval.retriever.get_store", return_value=store), \
             patch("src.retrieval.retriever.embed_query", return_value=[0.1]*10):
            import src.retrieval.retriever as r
            r._store = store
            # score_threshold in config is 0.3 — score 0.1 should be filtered
            results = r.retrieve("test question")
            assert all(c["score"] >= 0.3 for c in results)

    def test_retrieve_calls_search_with_filters(self):
        store = self._mock_store()
        with patch("src.retrieval.retriever.get_store", return_value=store), \
             patch("src.retrieval.retriever.embed_query", return_value=[0.1]*10):
            import src.retrieval.retriever as r
            r._store = store
            r.retrieve("question", filters={"source": "physics.pdf"})
            store.search.assert_called_once()
            _, kwargs = store.search.call_args
            assert kwargs.get("filters") == {"source": "physics.pdf"}

    def test_retrieve_definitions_calls_search_by_type(self):
        store = self._mock_store()
        with patch("src.retrieval.retriever.get_store", return_value=store), \
             patch("src.retrieval.retriever.embed_query", return_value=[0.1]*10):
            import src.retrieval.retriever as r
            r._store = store
            r.retrieve_definitions("What is entropy?")
            store.search_by_type.assert_called_once_with(
                [0.1]*10, chunk_type="definition", top_k=5
            )

    def test_retrieve_formulas_calls_search_by_type(self):
        store = self._mock_store()
        with patch("src.retrieval.retriever.get_store", return_value=store), \
             patch("src.retrieval.retriever.embed_query", return_value=[0.1]*10):
            import src.retrieval.retriever as r
            r._store = store
            r.retrieve_formulas("Force formula")
            store.search_by_type.assert_called_once_with(
                [0.1]*10, chunk_type="formula", top_k=5
            )
