"""
Ingest route — Fix 5 (async background tasks) + Fix 6 (file size limit + auth stub).
POST /ingest        → upload PDF, returns job_id immediately
GET  /ingest/status/{job_id} → poll job status
"""
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.security import APIKeyHeader
from src.pipeline.ingest_pipeline import ingest_pdf
from src.utils.job_store import get_job_store, JobStatus
from src.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

UPLOAD_DIR   = "data/raw"
def _max_bytes() -> int:
    return int(os.getenv("MAX_FILE_MB", "100")) * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Optional API key auth (Fix 6) ───────────────────────────────────────────
_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(key: str | None = Depends(_key_header)):
    api_key = os.getenv("RAG_API_KEY", "")
    if api_key and key != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ─── Background worker ───────────────────────────────────────────────────────
def _run_ingest(job_id: str, file_path: str):
    store = get_job_store()
    store.set_processing(job_id)
    try:
        summary = ingest_pdf(file_path)
        store.set_done(job_id, summary)
        logger.info(f"Job {job_id} done: {summary}")
    except Exception as e:
        store.set_failed(job_id, str(e))
        logger.error(f"Job {job_id} failed: {e}")


# ─── Endpoints ───────────────────────────────────────────────────────────────
@router.post("/ingest", summary="Upload a textbook PDF (async)",
             dependencies=[Depends(verify_api_key)])
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a textbook PDF. Processing runs in the background.
    Returns a job_id — poll GET /ingest/status/{job_id} for progress.

    Limits: see MAX_FILE_MB env var (default 100 MB). PDF only.
    """
    # Fix 6a: file type check
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    # Fix 6a: file size check (read into memory just to check size)
    content = await file.read()
    if len(content) > _max_bytes():
        raise HTTPException(413, f"File too large. Max: {os.getenv('MAX_FILE_MB', '100')} MB")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(content)
    logger.info(f"Saved {file.filename} ({len(content)//1024} KB)")

    job = get_job_store().create(file.filename)
    background_tasks.add_task(_run_ingest, job.job_id, save_path)

    return {
        "message":  "Ingestion started",
        "job_id":   job.job_id,
        "filename": file.filename,
        "status":   JobStatus.PENDING
    }


@router.get("/ingest/status/{job_id}", summary="Poll ingestion job status",
            dependencies=[Depends(verify_api_key)])
async def ingest_status(job_id: str):
    """Poll the status of a background ingestion job."""
    job = get_job_store().get(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return job.to_dict()
