import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import query, ingest
from src.vectorstore.chroma_store import ChromaStore
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.ingestion.embedder import MAX_EMBED_TOKENS

logger = get_logger(__name__)

app = FastAPI(
    title="Textbook RAG API",
    description=(
        "Retrieval-Augmented Generation for college textbooks. "
        "Uses NVIDIA NV-Embed-v1 + ChromaDB with smart section-aware chunking."
    ),
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(query.router,  tags=["Retrieval"])


@app.get("/health", tags=["Health"])
async def health():
    config = get_config()
    store  = ChromaStore()
    return {
        "status":           "ok",
        "total_vectors":    store.count(),
        "embedding_model":  config["embedding"]["model"],
        "max_embed_tokens": MAX_EMBED_TOKENS,
    }


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Textbook RAG API v2. Visit /docs for Swagger UI."}
