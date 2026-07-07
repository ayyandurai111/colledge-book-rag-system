import sys, os, pytest
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.ingestion.embedder import MAX_EMBED_TOKENS
from src.utils.tokenizer import count_tokens


class TestEmbedder:
    def _mock_response(self, n=1, dim=10):
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1] * dim) for _ in range(n)]
        return resp

    def test_max_embed_tokens_is_512(self):
        assert MAX_EMBED_TOKENS == 512

    def test_embed_texts_returns_list_of_vectors(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._mock_response(n=2)
        with patch("src.ingestion.embedder.get_client", return_value=mock_client):
            from src.ingestion.embedder import embed_texts
            result = embed_texts(["text one", "text two"])
            assert len(result) == 2
            assert all(isinstance(v, list) for v in result)

    def test_long_text_is_truncated_before_api_call(self):
        long_text = "word " * 600   # ~600 tokens, over the 512 limit
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._mock_response(n=1)

        with patch("src.ingestion.embedder.get_client", return_value=mock_client):
            from src.ingestion.embedder import embed_texts
            embed_texts([long_text])
            call_args = mock_client.embeddings.create.call_args
            sent_texts = call_args.kwargs.get("input") or call_args.args[0] if call_args.args else call_args.kwargs["input"]
            for t in sent_texts:
                assert count_tokens(t) <= MAX_EMBED_TOKENS

    def test_embed_chunks_adds_embedding_field(self):
        chunks = [
            {"chunk_id": "c1", "text": "Short text.", "token_count": 5},
            {"chunk_id": "c2", "text": "Another text.", "token_count": 5},
        ]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._mock_response(n=2)
        with patch("src.ingestion.embedder.get_client", return_value=mock_client):
            from src.ingestion import embedder
            result = embedder.embed_chunks(chunks)
            for c in result:
                assert "embedding" in c
                assert isinstance(c["embedding"], list)

    def test_embed_query_returns_single_vector(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._mock_response(n=1, dim=10)
        with patch("src.ingestion.embedder.get_client", return_value=mock_client):
            from src.ingestion.embedder import embed_query
            result = embed_query("What is Newton's law?")
            assert isinstance(result, list)
            assert len(result) == 10

    def test_embed_query_uses_query_input_type(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._mock_response(n=1)
        with patch("src.ingestion.embedder.get_client", return_value=mock_client):
            from src.ingestion.embedder import embed_query
            embed_query("test query")
            call_kwargs = mock_client.embeddings.create.call_args.kwargs
            assert call_kwargs["extra_body"]["input_type"] == "query"

    def test_embed_texts_uses_passage_input_type_by_default(self):
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = self._mock_response(n=1)
        with patch("src.ingestion.embedder.get_client", return_value=mock_client):
            from src.ingestion.embedder import embed_texts
            embed_texts(["some passage"])
            call_kwargs = mock_client.embeddings.create.call_args.kwargs
            assert call_kwargs["extra_body"]["input_type"] == "passage"

    def test_batching_splits_large_inputs(self):
        """50+ texts should be split into multiple batches."""
        mock_client = MagicMock()
        def side_effect(**kwargs):
            n = len(kwargs["input"])
            return self._mock_response(n=n)
        mock_client.embeddings.create.side_effect = side_effect

        with patch("src.ingestion.embedder.get_client", return_value=mock_client):
            from src.ingestion.embedder import embed_texts
            texts = [f"Sentence {i}." for i in range(120)]
            result = embed_texts(texts)
            assert len(result) == 120
            assert mock_client.embeddings.create.call_count >= 3  # 120/50 = 3 batches
