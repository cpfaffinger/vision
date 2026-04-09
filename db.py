"""
db.py – SQLite persistence layer
Tables:
  face_cache   – face_id → embedding + metadata, with TTL
  face_groups  – grouping sessions with results
  group_members – cluster→group assignment per session
"""

import json
import os
import sqlite3
import time
import uuid
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

log = logging.getLogger("vision-api.db")

DB_PATH           = Path(os.getenv("DB_PATH", "/app/data/vision.db"))
JOB_RESULT_TTL_SEC = int(os.getenv("JOB_RESULT_TTL_SECONDS", "43200"))


# ─── Bootstrap ────────────────────────────────────────────────────────────────

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS face_cache (
            face_id         TEXT PRIMARY KEY,
            source_file     TEXT,
            face_index      INTEGER NOT NULL DEFAULT 0,
            embedding       TEXT NOT NULL,          -- JSON array of floats
            embedding_dim   INTEGER NOT NULL,
            facial_area     TEXT,                   -- JSON {x,y,w,h}
            face_confidence REAL,
            face_quality_score REAL,                -- 0.0 (blurry) – 1.0 (sharp)
            model_name      TEXT NOT NULL,
            created_at      REAL NOT NULL,          -- unix timestamp
            expires_at      REAL NOT NULL           -- unix timestamp
        );

        CREATE INDEX IF NOT EXISTS idx_face_expires ON face_cache(expires_at);

        CREATE TABLE IF NOT EXISTS group_sessions (
            session_id   TEXT PRIMARY KEY,
            method       TEXT NOT NULL,
            params       TEXT NOT NULL,          -- JSON of algo params
            n_input      INTEGER NOT NULL,
            n_groups     INTEGER NOT NULL,
            n_noise      INTEGER NOT NULL,
            created_at   REAL NOT NULL,
            expires_at   REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_session_expires ON group_sessions(expires_at);

        CREATE TABLE IF NOT EXISTS group_members (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT NOT NULL REFERENCES group_sessions(session_id) ON DELETE CASCADE,
            group_id     INTEGER NOT NULL,
            face_id      TEXT,                   -- NULL if raw cluster_id was used
            cluster_id   TEXT NOT NULL,
            size         INTEGER NOT NULL DEFAULT 1,
            metadata     TEXT                    -- JSON passthrough
        );

        CREATE INDEX IF NOT EXISTS idx_member_session ON group_members(session_id);
        CREATE INDEX IF NOT EXISTS idx_member_group   ON group_members(session_id, group_id);

        CREATE TABLE IF NOT EXISTS jobs (
            job_id        TEXT PRIMARY KEY,
            job_type      TEXT NOT NULL,          -- 'embed' | 'cluster'
            status        TEXT NOT NULL,          -- PENDING|RUNNING|DONE|FAILED|TIMEOUT
            callback_url  TEXT,
            submitted_at  REAL NOT NULL,
            started_at    REAL,
            completed_at  REAL,
            result        TEXT,                   -- JSON result payload
            error         TEXT,                   -- error message on failure
            expires_at    REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_job_expires ON jobs(expires_at);
        CREATE INDEX IF NOT EXISTS idx_job_status  ON jobs(status);
        """)
    # ── Schema migrations run in their own connection (con above is already
    #    closed after the with-block exits — never pass it to _migrate)
    _migrate()
    log.info("Database initialised at %s", DB_PATH)


def _migrate() -> None:
    """Apply incremental schema changes without destroying existing data.
    Opens its own connection so it is safe to call after init_db's with-block."""
    additions = [
        ("face_cache", "face_quality_score", "REAL"),
    ]
    with _conn() as con:
        for table, column, typedef in additions:
            try:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {typedef}")
                log.info("Migration: added column %s.%s", table, column)
            except Exception:
                pass  # column already exists — safe to ignore


@contextmanager
def _conn():
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ─── Face Cache CRUD ──────────────────────────────────────────────────────────

def cache_face(
    embedding: list[float],
    model_name: str,
    ttl_seconds: int,
    facial_area: Optional[dict] = None,
    face_confidence: Optional[float] = None,
    face_quality_score: Optional[float] = None,
    source_file: Optional[str] = None,
    face_index: int = 0,
) -> str:
    face_id = str(uuid.uuid4())
    now = time.time()
    with _conn() as con:
        con.execute(
            """INSERT INTO face_cache
               (face_id, source_file, face_index, embedding, embedding_dim,
                facial_area, face_confidence, face_quality_score,
                model_name, created_at, expires_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                face_id,
                source_file,
                face_index,
                json.dumps(embedding),
                len(embedding),
                json.dumps(facial_area) if facial_area else None,
                face_confidence,
                face_quality_score,
                model_name,
                now,
                now + ttl_seconds,
            ),
        )
    return face_id


def get_face(face_id: str) -> Optional[dict]:
    now = time.time()
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM face_cache WHERE face_id=? AND expires_at>?",
            (face_id, now),
        ).fetchone()
    if not row:
        return None
    return _face_row(row)


def get_faces(face_ids: list[str]) -> dict[str, dict]:
    if not face_ids:
        return {}
    now = time.time()
    placeholders = ",".join("?" * len(face_ids))
    with _conn() as con:
        rows = con.execute(
            f"SELECT * FROM face_cache WHERE face_id IN ({placeholders}) AND expires_at>?",
            (*face_ids, now),
        ).fetchall()
    return {r["face_id"]: _face_row(r) for r in rows}


def list_faces(limit: int = 100, offset: int = 0) -> list[dict]:
    now = time.time()
    with _conn() as con:
        rows = con.execute(
            """SELECT face_id, source_file, face_index, embedding_dim,
                      face_confidence, face_quality_score, facial_area,
                      model_name, created_at, expires_at
               FROM face_cache WHERE expires_at>?
               ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            (now, limit, offset),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["expires_in_seconds"] = max(0.0, round(d["expires_at"] - now, 1))
        result.append(d)
    return result


def delete_face(face_id: str) -> bool:
    with _conn() as con:
        cur = con.execute("DELETE FROM face_cache WHERE face_id=?", (face_id,))
    return cur.rowcount > 0


def purge_expired() -> int:
    now = time.time()
    with _conn() as con:
        cur = con.execute("DELETE FROM face_cache WHERE expires_at<=?", (now,))
        cur2 = con.execute("DELETE FROM group_sessions WHERE expires_at<=?", (now,))
        cur3 = con.execute("DELETE FROM jobs WHERE expires_at<=?", (now,))
    total = cur.rowcount + cur2.rowcount + cur3.rowcount
    if total:
        log.info("Purged %d expired records", total)
    return total


def face_cache_stats() -> dict:
    now = time.time()
    with _conn() as con:
        total = con.execute("SELECT COUNT(*) FROM face_cache").fetchone()[0]
        active = con.execute(
            "SELECT COUNT(*) FROM face_cache WHERE expires_at>?", (now,)
        ).fetchone()[0]
        expired = total - active
    return {"total": total, "active": active, "expired": expired}


def get_all_embeddings(limit: int = 10_000) -> list[dict]:
    """Return face_id + embedding for all active (non-expired) cache entries.
    Used by the /faces/search and /faces/projection endpoints."""
    now = time.time()
    with _conn() as con:
        rows = con.execute(
            """SELECT face_id, embedding, source_file, face_confidence,
                      face_quality_score, model_name, embedding_dim
               FROM face_cache WHERE expires_at > ?
               ORDER BY created_at DESC LIMIT ?""",
            (now, limit),
        ).fetchall()
    return [
        {
            "face_id":            r["face_id"],
            "embedding":          json.loads(r["embedding"]),
            "source_file":        r["source_file"],
            "face_confidence":    r["face_confidence"],
            # face_quality_score was added via migration — may be NULL in old rows
            "face_quality_score": dict(r).get("face_quality_score"),
            "model_name":         r["model_name"],
            "embedding_dim":      r["embedding_dim"],
        }
        for r in rows
    ]


def _face_row(row) -> dict:
    d = dict(row)
    d["embedding"] = json.loads(d["embedding"])
    d["facial_area"] = json.loads(d["facial_area"]) if d.get("facial_area") else None
    d["expires_in_seconds"] = max(0.0, round(d["expires_at"] - time.time(), 1))
    return d


# ─── Group Sessions CRUD ──────────────────────────────────────────────────────

def save_group_session(
    method: str,
    params: dict,
    groups: list[dict],          # [{group_id, members:[{cluster_id,face_id?,size,metadata}]}]
    n_input: int,
    n_noise: int,
    ttl_seconds: int,
) -> str:
    session_id = str(uuid.uuid4())
    now = time.time()
    n_groups = len(groups)

    with _conn() as con:
        con.execute(
            """INSERT INTO group_sessions
               (session_id, method, params, n_input, n_groups, n_noise, created_at, expires_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, method, json.dumps(params), n_input, n_groups, n_noise,
             now, now + ttl_seconds),
        )
        for g in groups:
            for m in g["members"]:
                con.execute(
                    """INSERT INTO group_members
                       (session_id, group_id, face_id, cluster_id, size, metadata)
                       VALUES (?,?,?,?,?,?)""",
                    (
                        session_id,
                        g["group_id"],
                        m.get("face_id"),
                        m["cluster_id"],
                        m.get("size", 1),
                        json.dumps(m.get("metadata") or {}),
                    ),
                )
    return session_id


def get_group_session(session_id: str) -> Optional[dict]:
    now = time.time()
    with _conn() as con:
        sess = con.execute(
            "SELECT * FROM group_sessions WHERE session_id=? AND expires_at>?",
            (session_id, now),
        ).fetchone()
        if not sess:
            return None
        members = con.execute(
            "SELECT * FROM group_members WHERE session_id=? ORDER BY group_id",
            (session_id,),
        ).fetchall()

    group_map: dict[int, list] = {}
    for m in members:
        group_map.setdefault(m["group_id"], []).append({
            "cluster_id": m["cluster_id"],
            "face_id": m["face_id"],
            "size": m["size"],
            "metadata": json.loads(m["metadata"]) if m["metadata"] else {},
        })

    sess_dict = dict(sess)
    sess_dict["params"] = json.loads(sess_dict["params"])
    sess_dict["expires_in_seconds"] = max(0.0, round(sess_dict["expires_at"] - now, 1))
    sess_dict["groups"] = [
        {"group_id": gid, "cluster_count": len(ms), "members": ms}
        for gid, ms in sorted(group_map.items())
    ]
    return sess_dict


def count_active_sessions() -> int:
    now = time.time()
    with _conn() as con:
        return con.execute(
            "SELECT COUNT(*) FROM group_sessions WHERE expires_at>?", (now,)
        ).fetchone()[0]


def list_group_sessions(limit: int = 50) -> list[dict]:
    now = time.time()
    with _conn() as con:
        rows = con.execute(
            """SELECT session_id, method, params, n_input, n_groups, n_noise,
                      created_at, expires_at
               FROM group_sessions WHERE expires_at>?
               ORDER BY created_at DESC LIMIT ?""",
            (now, limit),
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["params"] = json.loads(d["params"])
        d["expires_in_seconds"] = max(0.0, round(d["expires_at"] - now, 1))
        result.append(d)
    return result


def delete_group_session(session_id: str) -> bool:
    with _conn() as con:
        cur = con.execute(
            "DELETE FROM group_sessions WHERE session_id=?", (session_id,)
        )
    return cur.rowcount > 0


# ─── Job persistence (for queue_worker.py) ────────────────────────────────────

def save_job(job) -> None:
    """Upsert a finished Job into the database."""
    now = time.time()
    result_json = json.dumps(job.result) if job.result is not None else None
    with _conn() as con:
        con.execute(
            """INSERT INTO jobs
               (job_id, job_type, status, callback_url,
                submitted_at, started_at, completed_at,
                result, error, expires_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(job_id) DO UPDATE SET
                 status=excluded.status,
                 started_at=excluded.started_at,
                 completed_at=excluded.completed_at,
                 result=excluded.result,
                 error=excluded.error,
                 expires_at=excluded.expires_at""",
            (
                job.job_id, job.job_type, str(job.status),
                job.callback_url,
                job.submitted_at, job.started_at, job.completed_at,
                result_json, job.error,
                now + JOB_RESULT_TTL_SEC,
            ),
        )


def load_job(job_id: str) -> Optional[dict]:
    now = time.time()
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM jobs WHERE job_id=? AND expires_at>?",
            (job_id, now),
        ).fetchone()
    return dict(row) if row else None


def list_jobs_db(limit: int = 100, status_filter: Optional[str] = None) -> list[dict]:
    now = time.time()
    params_status: tuple
    if status_filter:
        sql = """SELECT job_id, job_type, status, callback_url,
                        submitted_at, started_at, completed_at, error, expires_at
                 FROM jobs WHERE expires_at>? AND status=?
                 ORDER BY submitted_at DESC LIMIT ?"""
        params_status = (now, status_filter, limit)
    else:
        sql = """SELECT job_id, job_type, status, callback_url,
                        submitted_at, started_at, completed_at, error, expires_at
                 FROM jobs WHERE expires_at>?
                 ORDER BY submitted_at DESC LIMIT ?"""
        params_status = (now, limit)
    with _conn() as con:
        rows = con.execute(sql, params_status).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["expires_in_seconds"] = max(0.0, round(d["expires_at"] - now, 1))
        if d.get("completed_at") and d.get("submitted_at"):
            d["duration_seconds"] = round(d["completed_at"] - d["submitted_at"], 2)
        result.append(d)
    return result
