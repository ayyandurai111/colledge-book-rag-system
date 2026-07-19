import numpy as np
import pytest

from college_rag.exceptions import EmbeddingError


class TestEmbedder:
    def test_encode_returns_2d_array(self, fake_embedder):
        result = fake_embedder.encode(["hello world", "another sentence"])
        assert result.ndim == 2
        assert result.shape[0] == 2

    def test_encode_empty_list_raises(self, fake_embedder):
        with pytest.raises(EmbeddingError):
            fake_embedder.encode([])

    def test_injected_model_is_used_without_loading(self, fake_embedder):
        # Accessing .model should just return the injected fake, not attempt
        # any network call / real sentence-transformers import.
        assert fake_embedder.model is not None

    def test_normalized_vectors_have_unit_norm(self, fake_embedder):
        result = fake_embedder.encode(["Newton force mass acceleration"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-4 or norm == 0.0
