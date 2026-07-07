from src.retrieval.retriever import (
    retrieve, retrieve_definitions, retrieve_formulas, retrieve_expanded
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def query(question: str, top_k: int = None,
          filters: dict = None, content_type: str = None,
          expand_context: bool = False) -> dict:
    """
    Textbook RAG query pipeline (retrieval-only, no LLM).

    Args:
        question:       User's natural language question
        top_k:          Number of chunks to return
        filters:        Metadata filters e.g. {"source": "calculus.pdf"}
        content_type:   Optional priority type — "definition", "formula",
                        "table", "figure", or None (search all)
        expand_context: When True, each hit is expanded from its raw
                        400-600 token window into the FULL section it
                        belongs to (merged from ChromaDB, overlap removed,
                        stops at the next heading_path). Use this when the
                        chunks are headed to an LLM and partial context
                        would hurt the answer.

    Returns:
        {
          "question":     str,
          "content_type": str | None,
          "filters":      dict | None,
          "expand_context": bool,
          "chunks": [
            # expand_context=False:
            { text, source, chapter, section, subsection,
              page_number, heading_path, chunk_type, token_count, score }
            # expand_context=True:
            { text, source, chapter, section, subsection, heading_path,
              chunk_type, score, pages, merged_chunk_count, token_count }
          ],
          "total_chunks": int
        }
    """
    logger.info(f"RAG query | type={content_type} | filters={filters} | "
                f"expand={expand_context} | q='{question[:80]}'")

    if content_type == "definition":
        chunks = retrieve_definitions(question, top_k=top_k or 5)
    elif content_type == "formula":
        chunks = retrieve_formulas(question, top_k=top_k or 5)
    elif expand_context:
        chunks = retrieve_expanded(question, top_k=top_k, filters=filters)
    else:
        chunks = retrieve(question, top_k=top_k, filters=filters)

    return {
        "question":       question,
        "content_type":   content_type,
        "filters":        filters,
        "expand_context": expand_context,
        "chunks":         chunks,
        "total_chunks":   len(chunks)
    }
