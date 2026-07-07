from src.vectorstore.chroma_store import ChromaStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


def index_chunks(chunks: list[dict], store: ChromaStore = None) -> ChromaStore:
    """
    Save embedded textbook chunks into the vector store.

    Args:
        chunks: List of chunk dicts — must include 'embedding' field
        store:  Optional existing ChromaStore (created if None)

    Returns:
        ChromaStore instance
    """
    if store is None:
        store = ChromaStore()

    missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing:
        raise ValueError(f"Chunks missing embeddings: {missing[:5]}")

    # Separate by chunk_type for logging insight
    type_counts: dict[str, int] = {}
    for c in chunks:
        t = c.get("chunk_type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    store.add(chunks)
    logger.info(
        f"Indexed {len(chunks)} chunks. "
        f"Types: {type_counts}. "
        f"Total in store: {store.count()}"
    )
    return store
