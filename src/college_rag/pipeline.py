"""
pipeline.py
-----------
`RAGPipeline` is the main entry point of this library. It wires together
ingestion, chunking, embedding, indexing, and retrieval into a simple API
usable from both the CLI-less Streamlit UI and plain Python code.

This is a **pure retrieval** (the "R" in RAG only) system — there is no
LLM/Claude dependency. `query()` returns the most relevant book chunks
directly.

Usage:
    pipeline = RAGPipeline()
    pipeline.build_index_from_files(["book1.pdf", "book2.docx"])
    results = pipeline.query("What is Newton's First Law?")
    for r in results:
        print(r.score, r.chunk.location_label(), r.chunk.text)
"""
import logging
from typing import List, Optional

from college_rag.chunking.semantic_chunker import SemanticChunker
from college_rag.config import Config, DEFAULT_CONFIG
from college_rag.embeddings.embedder import Embedder
from college_rag.exceptions import EmptyDocumentError
from college_rag.ingestion.factory import extract
from college_rag.models import Chunk, IndexStats, SearchResult, TextBlock
from college_rag.retrieval.retriever import Retriever
from college_rag.vectorstore.faiss_store import FaissVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, config: Optional[Config] = None, embedder: Optional[Embedder] = None):
        self.config = config or DEFAULT_CONFIG
        self.config.validate()

        self.embedder = embedder or Embedder(model_name=self.config.embedding_model_name)
        self.chunker = SemanticChunker(
            embedder=self.embedder,
            min_chunk_chars=self.config.min_chunk_chars,
            max_chunk_chars=self.config.max_chunk_chars,
            breakpoint_percentile=self.config.breakpoint_percentile,
        )
        self.vectorstore = FaissVectorStore(embedder=self.embedder)
        self.retriever = Retriever(
            self.vectorstore,
            embedder=self.embedder,
            default_top_k=self.config.default_top_k,
            highlight_sentences=self.config.highlight_sentences,
        )

    # ------------------------------------------------------------------ #
    # Ingestion + Indexing
    # ------------------------------------------------------------------ #
    def ingest_files(self, paths: List[str]) -> List[TextBlock]:
        """Extracts text blocks from a list of PDF/DOCX file paths.

        If a file fails, processing continues with the remaining files —
        a warning is logged rather than aborting the whole batch.
        """
        all_blocks: List[TextBlock] = []
        for path in paths:
            try:
                blocks = extract(path)
                all_blocks.extend(blocks)
                logger.info("'%s': extracted %d text blocks", path, len(blocks))
            except EmptyDocumentError as e:
                logger.warning("Skipping '%s': %s", path, e)
        return all_blocks

    def chunk(self, blocks: List[TextBlock]) -> List[Chunk]:
        return self.chunker.chunk_blocks(blocks)

    def build_index_from_files(self, paths: List[str]) -> IndexStats:
        blocks = self.ingest_files(paths)
        if not blocks:
            raise EmptyDocumentError("Could not extract text from any of the given files")

        chunks = self.chunk(blocks)
        self.vectorstore.build(chunks)
        return self.stats()

    def save_index(self, folder: str) -> None:
        self.vectorstore.save(folder)

    def load_index(self, folder: str) -> None:
        self.vectorstore.load(folder)

    def stats(self) -> IndexStats:
        source_files = sorted({c.source_file for c in self.vectorstore.chunks})
        return IndexStats(
            total_chunks=len(self.vectorstore.chunks),
            total_source_files=len(source_files),
            source_files=source_files,
        )

    # ------------------------------------------------------------------ #
    # Query (pure retrieval — no LLM)
    # ------------------------------------------------------------------ #
    def query(self, question: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """Returns the chunks most relevant to the question (with similarity
        scores). No LLM is called — this is semantic retrieval only.
        """
        return self.retriever.retrieve(question, top_k=top_k)
