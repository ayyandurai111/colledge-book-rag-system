from src.ingestion.embedder import embed_query
from src.vectorstore.chroma_store import ChromaStore
from src.retrieval.context_expander import expand_hits
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_store = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store


def retrieve(query: str, top_k: int = None,
             filters: dict = None) -> list[dict]:
    """
    Embed the query and retrieve top-K relevant textbook chunks.

    Args:
        query:   User's question
        top_k:   Number of results (defaults to config)
        filters: Optional ChromaDB metadata filters
                 e.g. {"source": "calculus.pdf"}
                      {"chapter": "Chapter 3"}
                      {"chunk_type": "definition"}

    Returns:
        List of dicts with full metadata:
        { text, source, chapter, section, subsection,
          page_number, heading_path, chunk_type, token_count, score }
    """
    config          = get_config()
    top_k           = top_k or config["retrieval"]["top_k"]
    score_threshold = config["retrieval"].get("score_threshold", 0.0)

    logger.info(f"Retrieving top-{top_k} for: '{query[:80]}'")

    store = get_store()
    if store.count() == 0:
        logger.warning("Vector store is empty. Ingest textbooks first.")
        return []

    query_vector = embed_query(query)
    results      = store.search(query_vector, top_k=top_k, filters=filters)
    filtered     = [r for r in results if r["score"] >= score_threshold]

    logger.info(f"Returned {len(filtered)} chunks above threshold {score_threshold}")
    return filtered


def retrieve_expanded(query: str, top_k: int = None,
                       filters: dict = None) -> list[dict]:
    """
    Same as retrieve(), but each hit is expanded to its full section
    (see src.retrieval.context_expander) instead of returning the raw
    400-600 token window. Use this when the caller (e.g. an LLM answer
    step) needs complete context rather than a ranked snippet list.

    Returns list of:
      { text, source, chapter, section, subsection, heading_path,
        chunk_type, score, pages, merged_chunk_count, token_count }
    """
    hits = retrieve(query, top_k=top_k, filters=filters)
    if not hits:
        return []

    expanded = expand_hits(hits, store=get_store())
    logger.info(f"Expanded {len(hits)} hits into {len(expanded)} full section(s)")
    return expanded


def retrieve_definitions(query: str, top_k: int = 5) -> list[dict]:
    """Priority retrieval for definition chunks only."""
    store        = get_store()
    query_vector = embed_query(query)
    return store.search_by_type(query_vector, chunk_type="definition", top_k=top_k)


def retrieve_formulas(query: str, top_k: int = 5) -> list[dict]:
    """Priority retrieval for formula/equation chunks only."""
    store        = get_store()
    query_vector = embed_query(query)
    return store.search_by_type(query_vector, chunk_type="formula", top_k=top_k)
