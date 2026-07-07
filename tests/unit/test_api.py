"""
Unit-level API tests — fast, no real PDF processing.
Mocks embed client + ChromaDB so no network calls.
"""
import sys, os, io, pytest
from unittest.mock import patch, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import fitz
from fastapi.testclient import TestClient


def _make_tiny_pdf() -> bytes:
    doc  = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Test content for ingestion.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def patch_chroma(tmp_path):
    from src.vectorstore.chroma_store import ChromaStore
    real_init = ChromaStore.__init__
    def fake_init(self, persist_path=None):
        real_init(self, persist_path=str(tmp_path / "chroma"))
    with patch.object(ChromaStore, "__init__", fake_init):
        yield


@pytest.fixture(autouse=True)
def patch_embed():
    def fake_create(**kwargs):
        n    = len(kwargs.get("input", []))
        resp = MagicMock()
        resp.data = [MagicMock(embedding=[0.1]*10) for _ in range(n)]
        return resp
    mock_client = MagicMock()
    mock_client.embeddings.create.side_effect = fake_create
    with patch("src.ingestion.embedder.get_client", return_value=mock_client), \
         patch("src.ingestion.embedder._client", mock_client):
        yield mock_client


@pytest.fixture
def client():
    os.environ.pop("RAG_API_KEY", None)
    from api.main import app
    return TestClient(app)


# ─── Health / root ────────────────────────────────────────────────────────────
def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "total_vectors"    in data
    assert "embedding_model"  in data
    assert "max_embed_tokens" in data


def test_root_returns_message(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


# ─── Query ────────────────────────────────────────────────────────────────────
def test_query_valid(client):
    resp = client.post("/query", json={"question": "What is entropy?"})
    assert resp.status_code == 200


def test_query_returns_chunk_metadata(client):
    resp = client.post("/query", json={"question": "What is entropy?"})
    data = resp.json()
    assert "question"     in data
    assert "chunks"       in data
    assert "total_chunks" in data


def test_query_too_short_question(client):
    resp = client.post("/query", json={"question": "Hi"})
    assert resp.status_code == 422


def test_query_missing_question(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_with_content_type(client):
    resp = client.post("/query", json={
        "question": "What is a derivative?",
        "content_type": "definition"
    })
    assert resp.status_code == 200


def test_query_top_k_out_of_range(client):
    resp = client.post("/query", json={"question": "some question here", "top_k": 50})
    assert resp.status_code == 422


# ─── Ingest ───────────────────────────────────────────────────────────────────
def test_ingest_non_pdf_rejected(client):
    resp = client.post(
        "/ingest",
        files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 400


def test_ingest_pdf_accepted(client):
    pdf_bytes = _make_tiny_pdf()
    with patch("api.routes.ingest._run_ingest"):   # don't actually process
        resp = client.post(
            "/ingest",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id"   in data
    assert "filename" in data


def test_ingest_status_not_found(client):
    resp = client.get("/ingest/status/nonexistent-id")
    assert resp.status_code == 404


def test_ingest_file_too_large(client, monkeypatch):
    monkeypatch.setenv("MAX_FILE_MB", "1")
    large = b"x" * (2 * 1024 * 1024)   # 2 MB > 1 MB limit
    resp  = client.post(
        "/ingest",
        files={"file": ("big.pdf", large, "application/pdf")}
    )
    assert resp.status_code == 413


# ─── API key auth ─────────────────────────────────────────────────────────────
def test_api_key_required_when_set(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "secret123")
    from api.main import app
    c    = TestClient(app)
    resp = c.post("/query",
                  json={"question": "test question here"},
                  headers={"X-API-Key": "wrongkey"})
    assert resp.status_code == 401


def test_api_key_passes_when_correct(monkeypatch, patch_embed, patch_chroma):
    monkeypatch.setenv("RAG_API_KEY", "secret123")
    from api.main import app
    c    = TestClient(app)
    resp = c.post("/query",
                  json={"question": "test question here"},
                  headers={"X-API-Key": "secret123"})
    assert resp.status_code == 200
