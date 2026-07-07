"""
Simple in-memory job store for async ingestion status tracking (Fix 5).
In production, replace with Redis or a DB-backed store.
"""
import uuid
import threading
from enum import Enum
from datetime import datetime, timezone


class JobStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class Job:
    def __init__(self, filename: str):
        self.job_id    = str(uuid.uuid4())
        self.filename  = filename
        self.status    = JobStatus.PENDING
        self.created   = datetime.now(timezone.utc).isoformat()
        self.updated   = self.created
        self.result    = None   # summary dict on success
        self.error     = None   # error message on failure

    def to_dict(self) -> dict:
        return {
            "job_id":   self.job_id,
            "filename": self.filename,
            "status":   self.status,
            "created":  self.created,
            "updated":  self.updated,
            "result":   self.result,
            "error":    self.error,
        }


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(filename)
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def set_processing(self, job_id: str):
        self._update(job_id, status=JobStatus.PROCESSING)

    def set_done(self, job_id: str, result: dict):
        self._update(job_id, status=JobStatus.DONE, result=result)

    def set_failed(self, job_id: str, error: str):
        self._update(job_id, status=JobStatus.FAILED, error=error)

    def _update(self, job_id: str, **kwargs):
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)
                job.updated = datetime.now(timezone.utc).isoformat()


# Singleton
_store = JobStore()

def get_job_store() -> JobStore:
    return _store
