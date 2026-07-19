import numpy as np
import pytest

from college_rag.embeddings.embedder import Embedder
from college_rag.models import Chunk
from college_rag.retrieval.retriever import Retriever
from college_rag.vectorstore.faiss_store import FaissVectorStore


def _sample_chunks():
    return [
        Chunk(text="Newton force mass acceleration.", source_file="a.pdf", chunk_id=0),
        Chunk(text="Entropy heat thermodynamics.", source_file="a.pdf", chunk_id=1),
        Chunk(text="Cell organism life division.", source_file="a.pdf", chunk_id=2),
    ]


class TestRetrieverTopK:
    def test_retriever_uses_default_top_k(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        retriever = Retriever(store, embedder=fake_embedder, default_top_k=2)
        results = retriever.retrieve("Newton force")
        assert len(results) == 2

    def test_retriever_explicit_top_k_overrides_default(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        retriever = Retriever(store, embedder=fake_embedder, default_top_k=2)
        results = retriever.retrieve("Newton force", top_k=1)
        assert len(results) == 1

    def test_falls_back_to_vectorstore_embedder_if_none_given(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        retriever = Retriever(store)  # no embedder passed explicitly
        assert retriever.embedder is fake_embedder

    def test_blank_query_returns_empty_list(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        retriever = Retriever(store, embedder=fake_embedder)
        assert retriever.retrieve("   ") == []


class _DistinctVectorModel:
    """A fully-controllable fake model that returns a precise, orthogonal
    vector for each known text — used to deterministically test
    sentence-level highlight selection logic."""

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vecs = [np.array(self.mapping[t], dtype="float32") for t in texts]
        return np.array(vecs, dtype="float32")


class TestHighlightExtraction:
    def test_chunk_with_few_sentences_returns_full_text(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        retriever = Retriever(store, embedder=fake_embedder, highlight_sentences=5)
        # single-sentence chunk -> highlight == full chunk text (nothing to trim)
        highlight = retriever._extract_highlight("Newton force", "Newton force mass acceleration.")
        assert highlight == "Newton force mass acceleration."

    def test_picks_the_single_most_relevant_sentence(self, fake_embedder):
        chunk_text = "Sentence about newton force. Sentence about biology cells. Sentence about entropy heat."
        sentence_a, sentence_b, sentence_c = chunk_text.split(". ")[0] + ".", \
            chunk_text.split(". ")[1] + ".", chunk_text.split(". ")[2]

        mapping = {
            "physics force query": [1.0, 0.0, 0.0],
            sentence_a: [1.0, 0.0, 0.0],
            sentence_b: [0.0, 1.0, 0.0],
            sentence_c: [0.0, 0.0, 1.0],
        }
        model = _DistinctVectorModel(mapping)
        embedder = Embedder(model_name="distinct-fake", model=model)
        store = FaissVectorStore(embedder)  # not built — _extract_highlight doesn't need a built index

        retriever = Retriever(store, embedder=embedder, highlight_sentences=1)
        highlight = retriever._extract_highlight("physics force query", chunk_text)

        assert highlight == sentence_a
        assert sentence_b not in highlight
        assert sentence_c not in highlight

    def test_picks_top_n_sentences_in_original_reading_order(self, fake_embedder):
        chunk_text = "Sentence about newton force. Sentence about biology cells. Sentence about entropy heat."
        sentence_a, sentence_b, sentence_c = chunk_text.split(". ")[0] + ".", \
            chunk_text.split(". ")[1] + ".", chunk_text.split(". ")[2]

        # Query is closest to sentence_c, second-closest to sentence_a — but the
        # returned highlight must preserve ORIGINAL order (a, then c), not
        # relevance-rank order (c, then a).
        mapping = {
            "query": [0.9, 0.0, 1.0],
            sentence_a: [1.0, 0.0, 0.0],
            sentence_b: [0.0, 1.0, 0.0],
            sentence_c: [0.0, 0.0, 1.0],
        }
        model = _DistinctVectorModel(mapping)
        embedder = Embedder(model_name="distinct-fake", model=model)
        store = FaissVectorStore(embedder)  # not built — _extract_highlight doesn't need a built index

        retriever = Retriever(store, embedder=embedder, highlight_sentences=2)
        highlight = retriever._extract_highlight("query", chunk_text)

        assert highlight == f"{sentence_a} {sentence_c}"
        assert sentence_b not in highlight

    def test_highlight_sentences_clamped_to_minimum_one(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        retriever = Retriever(store, embedder=fake_embedder, highlight_sentences=0)
        assert retriever.highlight_sentences == 1

    def test_retrieve_attaches_highlight_to_every_result(self, fake_embedder):
        store = FaissVectorStore(fake_embedder)
        store.build(_sample_chunks())
        retriever = Retriever(store, embedder=fake_embedder, default_top_k=3)
        results = retriever.retrieve("entropy heat thermodynamics")
        assert all(r.highlight for r in results)

    def test_search_result_best_text_prefers_highlight(self):
        from college_rag.models import Chunk, SearchResult
        chunk = Chunk(text="Full chunk text here.", source_file="x.pdf")
        r = SearchResult(chunk=chunk, score=0.9, highlight="Highlighted sentence.")
        assert r.best_text() == "Highlighted sentence."

    def test_search_result_best_text_falls_back_to_chunk(self):
        from college_rag.models import Chunk, SearchResult
        chunk = Chunk(text="Full chunk text here.", source_file="x.pdf")
        r = SearchResult(chunk=chunk, score=0.9)  # no highlight
        assert r.best_text() == "Full chunk text here."
