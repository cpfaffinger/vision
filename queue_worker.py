"""
queue.py – Async in-process job queue for face processing
==========================================================

Architecture
------------
  ┌──────────┐   submit    ┌─────────────┐   worker    ┌────────────┐
  │  Client  │ ─────────►  │  JobQueue   │ ─────────►  │  DeepFace  │
  │  (HTTP)  │             │  (asyncio)  │             │  inference │
  └──────────┘             └─────────────┘             └─────┬──────┘
        ▲                         │                          │
        │   202 Accepted          │ persist result           ▼
        │   job_id returned       ▼                   ┌────────────┐
        │                   ┌──────────┐              │  SQLite DB │
        └───────────────────│ Callback │◄─────────────│  job table │
            POST result     │ (aiohttp)│              └────────────┘
            + Basic Auth    └──────────┘

Job lifecycle
-------------
  PENDING → RUNNING → DONE
                    ↘ FAILED
                    ↘ TIMEOUT

Callback payload (POST to callback_url)
----------------------------------------
{
  "job_id": "...",
  "status": "DONE",           // or "FAILED" / "TIMEOUT"
  "job_type": "embed",        // "embed" | "cluster"
  "submitted_at": 1712345678,
  "completed_at": 1712345690,
  "duration_seconds": 12.1,
  "result": { ... }           // same shape as synchronous endpoint
  "error": null               // populated on FAILED
}
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

import aiohttp

log = logging.getLogger("vision-api.queue")

# ─── Config (read once at import time) ────────────────────────────────────────
QUEUE_CONCURRENCY        = int(os.getenv("QUEUE_CONCURRENCY", "1"))
QUEUE_MAX_SIZE           = int(os.getenv("QUEUE_MAX_SIZE", "50"))
QUEUE_JOB_TIMEOUT        = int(os.getenv("QUEUE_JOB_TIMEOUT_SECONDS", "120"))
CALLBACK_TIMEOUT         = int(os.getenv("CALLBACK_TIMEOUT_SECONDS", "15"))
CALLBACK_MAX_RETRIES     = int(os.getenv("CALLBACK_MAX_RETRIES", "3"))
DEFAULT_CALLBACK_URL     = os.getenv("DEFAULT_CALLBACK_URL", "")
DEFAULT_CALLBACK_USER    = os.getenv("DEFAULT_CALLBACK_AUTH_USER", "")
DEFAULT_CALLBACK_PASS    = os.getenv("DEFAULT_CALLBACK_AUTH_PASS", "")
JOB_RESULT_TTL           = int(os.getenv("JOB_RESULT_TTL_SECONDS", "43200"))


# ─── Job model ────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING  = "PENDING"
    RUNNING  = "RUNNING"
    DONE     = "DONE"
    FAILED   = "FAILED"
    TIMEOUT  = "TIMEOUT"


class Job:
    __slots__ = (
        "job_id", "job_type", "callback_url",
        "callback_user", "callback_pass",
        "submitted_at", "started_at", "completed_at",
        "status", "result", "error",
        "_coro_factory",
    )

    def __init__(
        self,
        job_id:        str,
        job_type:      str,
        coro_factory:  Callable[[], Coroutine],
        callback_url:  str,
        callback_user: str = "",
        callback_pass: str = "",
    ):
        self.job_id        = job_id
        self.job_type      = job_type
        self._coro_factory = coro_factory
        self.callback_url  = callback_url
        self.callback_user = callback_user
        self.callback_pass = callback_pass
        self.submitted_at  = time.time()
        self.started_at:   Optional[float] = None
        self.completed_at: Optional[float] = None
        self.status        = JobStatus.PENDING
        self.result:       Optional[dict]  = None
        self.error:        Optional[str]   = None

    def to_dict(self, include_result: bool = True) -> dict:
        d = {
            "job_id":           self.job_id,
            "job_type":         self.job_type,
            "status":           self.status,
            "callback_url":     self.callback_url,
            "submitted_at":     self.submitted_at,
            "started_at":       self.started_at,
            "completed_at":     self.completed_at,
            "duration_seconds": (
                round(self.completed_at - self.submitted_at, 2)
                if self.completed_at else None
            ),
            "error": self.error,
        }
        if include_result:
            d["result"] = self.result
        return d


# ─── In-memory store (also persisted to DB on finish) ─────────────────────────
_jobs: dict[str, Job] = {}


# ─── Queue singleton ──────────────────────────────────────────────────────────

class JobQueue:
    _instance: Optional["JobQueue"] = None

    def __init__(self):
        self._queue:     asyncio.Queue[Job]   = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self._semaphore: asyncio.Semaphore    = asyncio.Semaphore(QUEUE_CONCURRENCY)
        self._workers:   list[asyncio.Task]   = []
        self._running:   int                  = 0

    @classmethod
    def get(cls) -> "JobQueue":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, n_workers: Optional[int] = None):
        n = n_workers or QUEUE_CONCURRENCY
        for _ in range(n):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)
        log.info("Job queue started with %d worker(s)  max_size=%d", n, QUEUE_MAX_SIZE)

    def stop(self):
        for task in self._workers:
            task.cancel()
        self._workers.clear()
        log.info("Job queue stopped")

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def running_count(self) -> int:
        return self._running

    def is_full(self) -> bool:
        return self._queue.full()

    async def submit(self, job: Job) -> bool:
        """Enqueue a job. Returns False if the queue is full."""
        if self._queue.full():
            return False
        _jobs[job.job_id] = job
        await self._queue.put(job)
        log.info("Job queued  job_id=%s  type=%s  queue_depth=%d",
                 job.job_id, job.job_type, self._queue.qsize())
        return True

    async def _worker(self):
        while True:
            job = await self._queue.get()
            async with self._semaphore:
                await self._run(job)
            self._queue.task_done()

    async def _run(self, job: Job):
        job.status     = JobStatus.RUNNING
        job.started_at = time.time()
        self._running += 1
        log.info("Job started  job_id=%s", job.job_id)

        try:
            result = await asyncio.wait_for(
                job._coro_factory(),
                timeout=QUEUE_JOB_TIMEOUT,
            )
            job.status = JobStatus.DONE
            job.result = result

        except asyncio.TimeoutError:
            job.status = JobStatus.TIMEOUT
            job.error  = f"Job exceeded timeout of {QUEUE_JOB_TIMEOUT}s"
            log.warning("Job timeout  job_id=%s", job.job_id)

        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error  = f"{type(exc).__name__}: {exc}"
            log.exception("Job failed  job_id=%s", job.job_id)

        finally:
            self._running -= 1
            job.completed_at = time.time()
            _persist_job(job)

        # fire-and-forget callback (don't await — don't block the worker)
        if job.callback_url:
            asyncio.create_task(_dispatch_callback(job))
        else:
            log.debug("No callback configured for job_id=%s", job.job_id)


# ─── Callback dispatcher ──────────────────────────────────────────────────────

async def _dispatch_callback(job: Job, attempt: int = 1):
    payload = job.to_dict(include_result=True)

    headers = {"Content-Type": "application/json", "X-Vision-Job-Id": job.job_id}

    # Build Basic Auth header if credentials are provided
    auth = None
    if job.callback_user:
        credentials = base64.b64encode(
            f"{job.callback_user}:{job.callback_pass}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {credentials}"

    timeout = aiohttp.ClientTimeout(total=CALLBACK_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                job.callback_url,
                json=payload,
                headers=headers,
                ssl=None,  # auto-detect: TLS for https://, plain for http://
            ) as resp:
                if resp.status < 300:
                    log.info(
                        "Callback OK  job_id=%s  url=%s  status=%d",
                        job.job_id, job.callback_url, resp.status,
                    )
                    return
                body = await resp.text()
                log.warning(
                    "Callback HTTP error  job_id=%s  status=%d  attempt=%d  body=%.200s",
                    job.job_id, resp.status, attempt, body,
                )

    except Exception as exc:
        log.warning(
            "Callback request failed  job_id=%s  attempt=%d  error=%s",
            job.job_id, attempt, exc,
        )

    # Retry with exponential backoff
    if attempt < CALLBACK_MAX_RETRIES:
        delay = 2 ** attempt  # 2s, 4s, 8s …
        log.info("Retrying callback in %ds  job_id=%s  attempt=%d",
                 delay, job.job_id, attempt + 1)
        await asyncio.sleep(delay)
        await _dispatch_callback(job, attempt + 1)
    else:
        log.error(
            "Callback permanently failed after %d attempts  job_id=%s",
            CALLBACK_MAX_RETRIES, job.job_id,
        )


# ─── Persistence helpers ──────────────────────────────────────────────────────

def _persist_job(job: Job):
    """Write finished job to SQLite so results survive a restart."""
    try:
        import db as _db
        _db.save_job(job)
    except Exception as exc:
        log.warning("Could not persist job result: %s", exc)


# ─── Public query API ─────────────────────────────────────────────────────────

def get_job(job_id: str) -> Optional[Job]:
    """Look up a job first in memory, then in DB."""
    if job_id in _jobs:
        return _jobs[job_id]
    # Try DB (for jobs that finished before a restart)
    try:
        import db as _db
        row = _db.load_job(job_id)
        if row:
            return _reconstruct_job(row)
    except Exception:
        pass
    return None


def list_jobs(limit: int = 100, status_filter: Optional[str] = None) -> list[dict]:
    """Return in-memory jobs, optionally filtered by status."""
    jobs = list(_jobs.values())
    if status_filter:
        jobs = [j for j in jobs if j.status == status_filter]
    jobs.sort(key=lambda j: j.submitted_at, reverse=True)
    return [j.to_dict(include_result=False) for j in jobs[:limit]]


def queue_stats() -> dict:
    q = JobQueue.get()
    statuses: dict[str, int] = {}
    for j in _jobs.values():
        statuses[j.status] = statuses.get(j.status, 0) + 1
    return {
        "pending":     q.pending_count,
        "running":     q.running_count,
        "max_size":    QUEUE_MAX_SIZE,
        "concurrency": QUEUE_CONCURRENCY,
        "job_counts":  statuses,
    }


def evict_old_jobs() -> int:
    """Remove finished jobs from the in-memory store that exceed JOB_RESULT_TTL.
    Called by the purge loop in main.py so memory doesn't grow unbounded."""
    cutoff = time.time() - JOB_RESULT_TTL
    finished = {JobStatus.DONE, JobStatus.FAILED, JobStatus.TIMEOUT}
    to_remove = [
        jid for jid, j in _jobs.items()
        if j.status in finished and (j.completed_at or 0) < cutoff
    ]
    for jid in to_remove:
        del _jobs[jid]
    if to_remove:
        log.info("Evicted %d finished jobs from memory", len(to_remove))
    return len(to_remove)


def _reconstruct_job(row: dict) -> Job:
    """Reconstruct a read-only Job shell from a DB row (no coro)."""
    j             = object.__new__(Job)
    j.job_id      = row["job_id"]
    j.job_type    = row["job_type"]
    j.callback_url  = row.get("callback_url", "")
    j.callback_user = ""
    j.callback_pass = ""
    j.submitted_at  = row["submitted_at"]
    j.started_at    = row.get("started_at")
    j.completed_at  = row.get("completed_at")
    j.status        = JobStatus(row["status"])
    j.result        = json.loads(row["result"]) if row.get("result") else None
    j.error         = row.get("error")
    j._coro_factory = None  # type: ignore
    return j
