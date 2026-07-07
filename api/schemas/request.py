from pydantic import BaseModel, Field
from typing import Optional, Literal


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Chunks to retrieve")
    content_type: Optional[Literal["definition", "formula", "table", "figure"]] = Field(
        None, description="Filter by chunk type for priority retrieval"
    )
    source: Optional[str] = Field(None, description="Filter by textbook filename")
    chapter: Optional[str] = Field(None, description="Filter by chapter title")
    expand_context: bool = Field(
        False,
        description=(
            "Return each hit's full section instead of its raw 400-600 "
            "token window — merged from ChromaDB, overlap removed, "
            "bounded by the next heading. Use for LLM-bound context."
        )
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is Newton's Second Law?",
                "top_k": 5,
                "content_type": "definition",
                "source": "physics.pdf",
                "expand_context": False
            }
        }
    }
