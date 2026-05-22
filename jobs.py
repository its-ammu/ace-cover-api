"""Job lifecycle: dataclass, thread-safe store, single-worker consumer loop.

Design notes:
- Single worker thread matches the snapshot's sequential execution model and
  avoids concurrent access to the shared ``AceStepHandler`` state (monkey-patch).
- ``JobStore`` is entirely in-memory; process restart loses all job records.
- Completed outputs remain on disk and are served directly by the download route.
"""

from __future__ import annotations

import dataclasses
import queue
import threading
import traceback
import uuid
from enum import Enum
from typing import Any, Callable, Optional

from loguru import logger


class JobStatus(str, Enum):
    """Valid states in the Job state machine."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclasses.dataclass
class Job:
    """A single cover-generation task.

    Attributes:
        job_id: Unique identifier (UUIDv4).
        params: Serializable dict of generation parameters.
        instrumental_path: Absolute path to the uploaded instrumental stem.
        bass_path: Optional absolute path to the uploaded bass stem.
        status: Current lifecycle status.
        output_path: Set when status is DONE; path to the generated FLAC.
        error: Set when status is FAILED; human-readable error message.
    """

    job_id: str
    params: dict[str, Any]
    instrumental_path: str
    bass_path: Optional[str]
    status: JobStatus = JobStatus.QUEUED
    output_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this job.

        Returns:
            Dict with ``job_id``, ``status``, ``output_path``, and ``error``.
        """
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "output_path": self.output_path,
            "error": self.error,
        }


class JobStore:
    """Thread-safe in-memory store and submission queue for jobs.

    Attributes:
        _store: Mapping from job_id to Job.
        _queue: Unbounded FIFO queue consumed by the worker thread.
        _lock: Guards reads/writes to ``_store``.
    """

    def __init__(self) -> None:
        """Initialize an empty store and queue."""
        self._store: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()

    def create(
        self,
        params: dict[str, Any],
        instrumental_path: str,
        bass_path: Optional[str],
    ) -> Job:
        """Create a new job, persist it, and enqueue its id.

        Args:
            params: Validated generation parameter dict.
            instrumental_path: Path to the saved instrumental upload.
            bass_path: Optional path to the saved bass upload.

        Returns:
            The newly created Job with status QUEUED.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            params=params,
            instrumental_path=instrumental_path,
            bass_path=bass_path,
        )
        with self._lock:
            self._store[job_id] = job
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        """Retrieve a job by id.

        Args:
            job_id: The UUID string to look up.

        Returns:
            The Job if found, otherwise ``None``.
        """
        with self._lock:
            return self._store.get(job_id)

    def _update(self, job_id: str, **kwargs: Any) -> None:
        """Update fields on a stored job (worker-internal).

        Args:
            job_id: Target job id.
            **kwargs: Attribute names and new values to apply.
        """
        with self._lock:
            job = self._store.get(job_id)
            if job is None:
                return
            for key, value in kwargs.items():
                setattr(job, key, value)


def start_worker(store: JobStore, runner: Callable[[Job], str]) -> threading.Thread:
    """Spawn the single background worker thread that processes jobs.

    The worker blocks on ``store._queue``, marks jobs RUNNING, calls
    ``runner(job)`` which must return the output FLAC path, then marks
    the job DONE (or FAILED on exception).

    Args:
        store: The shared JobStore to consume from.
        runner: Callable ``(job) -> output_path`` that performs the actual
            generation.  Must be thread-safe (only one call at a time here).

    Returns:
        The started daemon thread.
    """

    def _worker_loop() -> None:
        logger.info("[worker] Cover API worker started")
        while True:
            job_id = store._queue.get()
            if job_id is None:
                logger.info("[worker] Received sentinel — shutting down")
                break
            store._update(job_id, status=JobStatus.RUNNING)
            job = store.get(job_id)
            if job is None:
                logger.warning(f"[worker] Job {job_id} disappeared from store — skipping")
                continue
            try:
                logger.info(f"[worker] Starting job {job_id}")
                out_path = runner(job)
                store._update(job_id, status=JobStatus.DONE, output_path=out_path)
                logger.info(f"[worker] Job {job_id} done: {out_path}")
            except Exception:
                err = traceback.format_exc()
                logger.error(f"[worker] Job {job_id} failed:\n{err}")
                store._update(job_id, status=JobStatus.FAILED, error=err)

    thread = threading.Thread(target=_worker_loop, daemon=True, name="cover-api-worker")
    thread.start()
    return thread
