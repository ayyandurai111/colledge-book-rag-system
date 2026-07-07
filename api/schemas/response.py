from pydantic import BaseModel
from typing import List, Optional, Dict


class ChunkResult(BaseModel):
    text:         str
    source:       str
    chapter:      str
    section:      str
    subsection:   str
    heading_path: str
    chunk_type:   str
    token_count:  int
    score:        Optional[float] = None

    # Present when expand_context=False (one raw 400-600 token window)
    page_number:  Optional[int] = None

    # Present when expand_context=True (a full merged section)
    pages:               Optional[List[int]] = None
    merged_chunk_count:  Optional[int]        = None


class QueryResponse(BaseModel):
    question:       str
    content_type:   Optional[str]  = None
    filters:        Optional[Dict[str, str]] = None
    expand_context: bool = False
    chunks:         List[ChunkResult]
    total_chunks:   int


class IngestJobResponse(BaseModel):
    message:  str
    job_id:   str
    filename: str
    status:   str


class JobStatusResponse(BaseModel):
    job_id:   str
    filename: str
    status:   str
    created:  str
    updated:  str
    result:   Optional[Dict]  = None
    error:    Optional[str]   = None


class HealthResponse(BaseModel):
    status:          str
    total_vectors:   int
    embedding_model: str
    max_embed_tokens: int
