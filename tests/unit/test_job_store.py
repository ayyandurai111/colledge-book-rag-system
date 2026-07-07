import sys, os, time, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.utils.job_store import JobStore, JobStatus


class TestJobStore:
    def setup_method(self):
        self.store = JobStore()

    def test_create_returns_job(self):
        job = self.store.create("book.pdf")
        assert job.job_id is not None
        assert job.filename == "book.pdf"
        assert job.status == JobStatus.PENDING

    def test_get_existing_job(self):
        job      = self.store.create("test.pdf")
        fetched  = self.store.get(job.job_id)
        assert fetched is not None
        assert fetched.job_id == job.job_id

    def test_get_missing_job(self):
        assert self.store.get("nonexistent-id") is None

    def test_set_processing(self):
        job = self.store.create("test.pdf")
        self.store.set_processing(job.job_id)
        assert self.store.get(job.job_id).status == JobStatus.PROCESSING

    def test_set_done(self):
        job    = self.store.create("test.pdf")
        result = {"chunks_created": 42, "total_in_store": 42}
        self.store.set_done(job.job_id, result)
        fetched = self.store.get(job.job_id)
        assert fetched.status == JobStatus.DONE
        assert fetched.result == result

    def test_set_failed(self):
        job = self.store.create("test.pdf")
        self.store.set_failed(job.job_id, "Embedding API error")
        fetched = self.store.get(job.job_id)
        assert fetched.status == JobStatus.FAILED
        assert "Embedding API error" in fetched.error

    def test_to_dict_has_required_keys(self):
        job = self.store.create("test.pdf")
        d   = job.to_dict()
        assert set(d.keys()) >= {"job_id", "filename", "status", "created", "updated"}

    def test_unique_ids_per_job(self):
        ids = {self.store.create("book.pdf").job_id for _ in range(10)}
        assert len(ids) == 10

    def test_thread_safe_creation(self):
        """Multiple threads creating jobs concurrently should not crash."""
        results = []
        def create_job():
            results.append(self.store.create("concurrent.pdf").job_id)

        threads = [threading.Thread(target=create_job) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(results) == 20
        assert len(set(results)) == 20   # all unique

    def test_updated_timestamp_changes(self):
        job = self.store.create("test.pdf")
        created_ts = job.updated
        time.sleep(0.01)
        self.store.set_processing(job.job_id)
        assert self.store.get(job.job_id).updated >= created_ts
