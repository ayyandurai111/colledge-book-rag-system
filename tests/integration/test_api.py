"""
Integration tests for FastAPI endpoints.
Mocks NVIDIA embedding API and uses a real in-memory ChromaDB.
"""
import sys, os, pytest, io
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import fitz
from fastapi.testclient import TestClient


def _make_pdf_bytes(chapters=1):
    doc = fitz.open()
    for i in range(chapters):
        page = doc.new_page()
        page.insert_text((72, 72),  f"Chapter {i+1}", fontsize=18)
        page.insert_text((72, 120), "Body content sentence. " * 30, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def patch_chroma(tmp_path):
    """Redirect ChromaDB to a tmp directory for every test."""
    from src.vectorstore.chroma_store import ChromaStore
    real_init = ChromaStore.__init__
    def patched_init(self, persist_path=None):
        real_init(self, persist_path=str(tmp_path / "chroma"))
    with patch.object(ChromaStore, "__init__", patched_init):
        yield


@pytest.fixture(autouse=True)
def patch_embed():
    """Mock NVIDIA embedding API globally."""
    def fake_create(**kwargs):
        n    = len(kwargs.get("input", []))
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1]*10) for _ in range(n)]
        return resp
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create
    with patch("src.ingestion.embedder.get_client", return_value=mock_client), \
         patch("src.ingestion.embedder._client", mock_client):
        yield


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "total_vectors"    in data
        assert "embedding_model"  in data
        assert "max_embed_tokens" in data

    def test_root_returns_message(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "message" in resp.json()


class TestIngestEndpoint:
    def test_ingest_valid_pdf_returns_job_id(self, client):
        pdf_bytes = _make_pdf_bytes()
        resp = client.post(
            "/ingest",
            files={"file": ("textbook.pdf", pdf_bytes, "application/pdf")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id"   in data
        assert "filename" in data
        assert data["filename"] == "textbook.pdf"

    def test_ingest_non_pdf_rejected(self, client):
        resp = client.post(
            "/ingest",
            files={"file": ("notes.txt", b"hello world", "text/plain")}
        )
        assert resp.status_code == 400

    def test_ingest_oversized_file_rejected(self, client):
        big_content = b"x" * (101 * 1024 * 1024)  # 101 MB
        resp = client.post(
            "/ingest",
            files={"file": ("big.pdf", big_content, "application/pdf")}
        )
        assert resp.status_code == 413

    def test_ingest_status_endpoint_exists(self, client):
        pdf_bytes = _make_pdf_bytes()
        post_resp = client.post(
            "/ingest",
            files={"file": ("book.pdf", pdf_bytes, "application/pdf")}
        )
        job_id   = post_resp.json()["job_id"]
        status   = client.get(f"/ingest/status/{job_id}")
        assert status.status_code == 200
        assert status.json()["job_id"] == job_id

    def test_ingest_status_404_for_unknown_job(self, client):
        resp = client.get("/ingest/status/nonexistent-job-id")
        assert resp.status_code == 404


class TestQueryEndpoint:
    def test_query_returns_response_structure(self, client):
        resp = client.post("/query", json={"question": "What is entropy?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "question"     in data
        assert "chunks"       in data
        assert "total_chunks" in data

    def test_query_with_content_type_filter(self, client):
        resp = client.post("/query", json={
            "question":     "What is a derivative?",
            "content_type": "definition"
        })
        assert resp.status_code == 200

    def test_query_with_source_filter(self, client):
        resp = client.post("/query", json={
            "question": "Explain Newton's law",
            "source":   "physics.pdf"
        })
        assert resp.status_code == 200

    def test_query_too_short_rejected(self, client):
        resp = client.post("/query", json={"question": "Hi"})
        assert resp.status_code == 422   # Pydantic min_length=3

    def test_query_invalid_content_type_rejected(self, client):
        resp = client.post("/query", json={
            "question":     "some question here",
            "content_type": "invalid_type"
        })
        assert resp.status_code == 422

    def test_query_top_k_out_of_range_rejected(self, client):
        resp = client.post("/query", json={
            "question": "some question here",
            "top_k":    50
        })
        assert resp.status_code == 422


class TestApiKeyAuth:
    def test_no_auth_when_key_not_set(self, client):
        """When RAG_API_KEY env var is empty, all requests should pass."""
        resp = client.post("/query", json={"question": "What is force?"})
        assert resp.status_code == 200

    def test_wrong_api_key_rejected_when_key_set(self, tmp_path):
        with patch.dict(os.environ, {"RAG_API_KEY": "secret123"}):
            # Re-import app to pick up new env
            import importlib
            import api.routes.query as qr
            import api.routes.ingest as ir
            importlib.reload(qr)
            importlib.reload(ir)

            from api.main import app
            c    = TestClient(app)
            resp = c.post(
                "/query",
                json={"question": "What is entropy?"},
                headers={"X-API-Key": "wrongkey"}
            )
            assert resp.status_code == 401
