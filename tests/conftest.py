"""Shared pytest configuration and fixtures."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("NVIDIA_API_KEY", "test-dummy-key")
os.environ.setdefault("RAG_API_KEY", "")

import pytest

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons between tests to avoid state leakage."""
    import src.utils.config as cfg
    import src.ingestion.embedder as emb
    import src.retrieval.retriever as ret

    cfg._config = None
    emb._client = None
    ret._store  = None

    yield

    cfg._config = None
    emb._client = None
    ret._store  = None
