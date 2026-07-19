"""
embedder.py
-----------
A thin wrapper that lazily loads a sentence-transformers model and turns
text into vectors. Lazy loading avoids downloading/loading the (large)
model just from importing the package (e.g. during unit tests).

For testing: pass `Embedder(model=<fake object with .encode()>)` to
dependency-inject a fake model with no network dependency.
"""
import logging
from typing import List, Optional

import numpy as np

from college_rag.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder:
    """A wrapper around a sentence-transformers model.

    If `model` is provided (dependency injection, used by tests), it is
    used as-is — otherwise `model_name` is lazily loaded in production.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 model: Optional[object] = None):
        self.model_name = model_name
        self._model = model  # injected model (tests) or None (lazy load in production)

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Run `pip install -r requirements.txt`."
                ) from e
            try:
                logger.info("Loading embedding model: %s", self.model_name)
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {e}"
                ) from e
        return self._model

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """Converts a list of texts into normalized embedding vectors.

        Using normalized embeddings means dot-product == cosine similarity,
        so FAISS IndexFlatIP can be used directly.
        """
        if not texts:
            raise EmbeddingError("encode() was called with an empty list")
        try:
            embeddings = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=show_progress_bar,
            )
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}") from e

        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            # Some fake/mocked models may return a single flat vector for a single text
            embeddings = embeddings.reshape(1, -1)
        return embeddings
