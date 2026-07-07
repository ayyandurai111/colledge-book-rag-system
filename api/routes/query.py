import os
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import APIKeyHeader
from api.schemas.request import QueryRequest
from api.schemas.response import QueryResponse, ChunkResult
from src.pipeline.rag_pipeline import query as run_query
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str | None = Depends(_key_header)):
    api_key = os.getenv("RAG_API_KEY", "")
    if api_key and key != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.post("/query", response_model=QueryResponse,
             summary="Query the textbook RAG system",
             dependencies=[Depends(verify_api_key)])
async def query_endpoint(request: QueryRequest):
    """
    Embed the question and retrieve the most relevant textbook chunks.

    Optional filters:
    - **content_type**: prioritise definitions, formulas, tables, or figures
    - **source**: restrict to a specific textbook PDF filename
    - **chapter**: restrict to a specific chapter title
    - **expand_context**: return each hit's full section (merged from
      ChromaDB, overlap removed) instead of a raw 400-600 token window —
      use this when the chunks are headed to an LLM for answer generation
    """
    try:
        filters = {}
        if request.source:  filters["source"]  = request.source
        if request.chapter: filters["chapter"] = request.chapter

        result = run_query(
            question       = request.question,
            top_k          = request.top_k,
            filters        = filters or None,
            content_type   = request.content_type,
            expand_context = request.expand_context
        )
        return QueryResponse(
            question       = result["question"],
            content_type   = result["content_type"],
            filters        = result["filters"],
            expand_context = result["expand_context"],
            chunks         = [ChunkResult(**c) for c in result["chunks"]],
            total_chunks   = result["total_chunks"]
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))
