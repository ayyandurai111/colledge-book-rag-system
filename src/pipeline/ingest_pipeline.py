from src.ingestion.pdf_loader import load_pdf, load_pdfs_from_dir
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_chunks
from src.ingestion.indexer import index_chunks
from src.vectorstore.chroma_store import ChromaStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_pdf(file_path: str, store: ChromaStore = None) -> dict:
    """
    Full textbook ingestion pipeline: PDF → ChromaDB

    Steps:
      1. Load PDF with rich block extraction (font size, bold flags, TOC)
      2. Auto-detect PDF type (toc | font)
      3. Smart section-based chunking with quality checks
      4. Embed chunks with NVIDIA NV-Embed-v1
      5. Index into ChromaDB with full metadata

    Returns:
        Summary dict
    """
    logger.info(f"=== Textbook ingestion: {file_path} ===")

    doc    = load_pdf(file_path)
    chunks = chunk_document(doc)

    if not chunks:
        logger.warning(f"No valid chunks produced from {file_path}")
        return {
            "file": file_path, "pdf_type": doc["pdf_type"],
            "pages_loaded": len(doc["pages"]),
            "chunks_created": 0, "chunk_types": {}, "total_in_store": 0
        }

    chunks = embed_chunks(chunks)
    store  = index_chunks(chunks, store=store)

    # Count chunk types
    type_counts = {}
    for c in chunks:
        t = c.get("chunk_type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    summary = {
        "file":          file_path,
        "pdf_type":      doc["pdf_type"],
        "pages_loaded":  len(doc["pages"]),
        "chunks_created": len(chunks),
        "chunk_types":   type_counts,
        "total_in_store": store.count()
    }

    logger.info(f"=== Ingestion complete: {summary} ===")
    return summary


def ingest_directory(dir_path: str, store: ChromaStore = None) -> dict:
    """
    Ingest all PDFs in a directory.
    Each PDF is processed independently with its own detected strategy.
    """
    logger.info(f"=== Batch textbook ingestion from: {dir_path} ===")

    if store is None:
        store = ChromaStore()

    docs = load_pdfs_from_dir(dir_path)
    total_chunks = 0
    all_type_counts: dict[str, int] = {}

    for doc in docs:
        chunks = chunk_document(doc)
        if not chunks:
            continue
        chunks = embed_chunks(chunks)
        index_chunks(chunks, store=store)
        total_chunks += len(chunks)
        for c in chunks:
            t = c.get("chunk_type", "text")
            all_type_counts[t] = all_type_counts.get(t, 0) + 1

    summary = {
        "directory":     dir_path,
        "files_ingested": len(docs),
        "chunks_created": total_chunks,
        "chunk_types":   all_type_counts,
        "total_in_store": store.count()
    }

    logger.info(f"=== Batch ingestion complete: {summary} ===")
    return summary
