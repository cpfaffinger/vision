"""
Vision API
==========
Face embedding & clustering, cluster grouping, barcode detection.
All results can be persisted in the built-in SQLite cache for later retrieval.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator

import db
import clustering as cl
import queue_worker as qw
from errors import (
    ErrorCode,
    VisionAPIException,
    err_clustering,
    err_db,
    err_dim_mismatch,
    err_face_not_found,
    err_file_too_large,
    err_image_decode,
    err_inference,
    err_invalid_b64,
    err_missing_param,
    err_no_face,
    err_session_not_found,
    err_too_few_clusters,
    err_unknown_method,
    http_exception_handler,
    unhandled_exception_handler,
    vision_exception_handler,
)

# ─── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME       = os.getenv("MODEL_NAME", "VGG-Face")
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "retinaface")
MAX_UPLOAD_MB    = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_BYTES        = MAX_UPLOAD_MB * 1024 * 1024
FACE_TTL_SEC     = int(os.getenv("FACE_TTL_SECONDS",  str(60 * 60 * 24)))       # 24 h
GROUP_TTL_SEC    = int(os.getenv("GROUP_TTL_SECONDS", str(60 * 60 * 24 * 7)))   # 7 d
PURGE_INTERVAL   = int(os.getenv("PURGE_INTERVAL_SECONDS", "3600"))
UPLOAD_DIR       = Path("/app/uploads")
_ADMIN_HTML      = Path(__file__).parent / "admin.html"

# ─── Image quality & pre-processing ──────────────────────────────────────────
# MIN_FACE_CONFIDENCE: discard detections below this threshold (0.0 = accept all)
MIN_FACE_CONFIDENCE  = float(os.getenv("MIN_FACE_CONFIDENCE", "0.0"))
# MAX_IMAGE_PX: resize images larger than this before detection (0 = no resize)
MAX_IMAGE_PX         = int(os.getenv("MAX_IMAGE_PX", "1920"))
# ENABLE_PREPROCESSING: apply CLAHE luminance equalisation before detection
ENABLE_PREPROCESSING = os.getenv("ENABLE_PREPROCESSING", "false").lower() == "true"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("vision-api")

# ── Suppress perpetually-noisy third-party loggers ────────────────────────────
# These loggers are never useful for application-level debugging regardless of
# LOG_LEVEL:
#   h5py._conv   – internal HDF5 type-converter registrations (DEBUG flood)
#   tensorflow   – Python-level TF/Keras/TPU/TRT messages (already filtered at
#                  the C++ level via TF_CPP_MIN_LOG_LEVEL=3 in the env, but the
#                  Python logger still emits Cloud-TPU and TF-TRT warnings)
#   absl         – TF's abseil-py used by Keras (progress / config INFO spam)
for _noisy_logger in ("h5py._conv", "h5py", "tensorflow", "absl",
                      "numba", "numba.core", "numba.core.ssa", "numba.core.byteflow"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    log.info("Vision API starting  model=%s  detector=%s", MODEL_NAME, DETECTOR_BACKEND)
    purge_task = asyncio.create_task(_purge_loop())
    asyncio.create_task(_prewarm_model())
    qw.JobQueue.get().start()
    yield
    purge_task.cancel()
    qw.JobQueue.get().stop()


async def _prewarm_model():
    """Run a dummy inference on startup so the model is loaded before the first real request."""
    import tempfile
    log.info("Pre-warming DeepFace model in background…")
    # create a minimal white image – no face, but forces model weights to load
    img = np.ones((64, 64, 3), dtype=np.uint8) * 220
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=UPLOAD_DIR) as f:
        tmp_path = Path(f.name)
    try:
        cv2.imwrite(str(tmp_path), img)
        await _represent(str(tmp_path))   # will find no face – that's fine
    except Exception:
        pass  # expected: no face detected; model is loaded regardless
    finally:
        tmp_path.unlink(missing_ok=True)
    log.info("DeepFace model pre-warm complete")


async def _purge_loop():
    while True:
        await asyncio.sleep(PURGE_INTERVAL)
        try:
            n = db.purge_expired()
            if n:
                log.info("Auto-purge removed %d expired records", n)
        except Exception as exc:
            log.warning("Auto-purge failed: %s", exc)
        try:
            qw.evict_old_jobs()
        except Exception as exc:
            log.warning("Job memory eviction failed: %s", exc)


# ─── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vision API",
    version="2.0.0",
    description="""
## Vision API

A self-contained service for:

- **Face Embedding** – detect faces and compute deep feature vectors
- **Face Clustering** – group faces within a single image (DBSCAN)
- **Cluster Grouping** – merge many pre-computed clusters into person-groups
- **Barcode Detection** – decode 0..n barcodes of any common symbology

### Face ID Cache

Every embedded face receives a **`face_id`** (UUID).  
Store it client-side and pass it directly to grouping endpoints — no need to
resend raw vectors. IDs expire after a configurable TTL (default **24 hours**).

### Group Sessions

Every grouping result is stored as a **session** (default TTL **7 days**).  
Retrieve, inspect, or delete sessions at any time via the `/sessions` endpoints.
""",
    lifespan=lifespan,
)

app.add_exception_handler(VisionAPIException, vision_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


# ─── Lazy DeepFace loader ─────────────────────────────────────────────────────
_deepface = None

def get_deepface():
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace
            _deepface = DeepFace
            log.info("DeepFace loaded successfully")
        except Exception as exc:
            log.error("Failed to load DeepFace: %s", exc)
            raise VisionAPIException(
                503, ErrorCode.MODEL_LOAD_FAILED,
                "Face recognition model could not be loaded.",
                detail=str(exc),
            )
    return _deepface


async def _represent(img_path: str) -> list:
    """Run DeepFace.represent() in a thread pool so the event loop stays free."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_deepface().represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
        ),
    )


# ─── Image helpers ────────────────────────────────────────────────────────────
def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise err_image_decode()
    return img


def _validate_size(data: bytes):
    if len(data) > MAX_BYTES:
        raise err_file_too_large(MAX_UPLOAD_MB)


def _save_tmp(data: bytes) -> Path:
    p = UPLOAD_DIR / f"{uuid.uuid4()}.jpg"
    p.write_bytes(data)
    return p


def _preprocess_inplace(path: Path) -> None:
    """
    In-place image preprocessing to improve face detection quality:
      1. Resize: cap the longest edge at MAX_IMAGE_PX (speeds up inference,
         avoids OOM on very high-resolution uploads).
      2. CLAHE: adaptive histogram equalisation on the L channel (LAB colour
         space) — lifts shadows, reduces blown highlights.  Helps RetinaFace
         find faces in backlit / indoor / low-contrast photos.

    Both steps are no-ops when the respective config flag is disabled.
    """
    if not ENABLE_PREPROCESSING and MAX_IMAGE_PX == 0:
        return

    img = cv2.imread(str(path))
    if img is None:
        return  # unreadable — let DeepFace handle the error

    changed = False

    # ── Step 1: Resize ──────────────────────────────────────────────────────
    if MAX_IMAGE_PX > 0:
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > MAX_IMAGE_PX:
            scale = MAX_IMAGE_PX / longest
            img   = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            changed = True

    # ── Step 2: CLAHE luminance equalisation ────────────────────────────────
    if ENABLE_PREPROCESSING:
        lab        = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab        = cv2.merge([clahe.apply(l), a, b])
        img        = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        changed    = True

    if changed:
        cv2.imwrite(str(path), img)


def _face_quality_score(img: np.ndarray, facial_area: dict) -> float:
    """
    Estimate sharpness of a detected face crop using Laplacian variance.

    Returns a score in [0.0, 1.0]:
      ≥ 0.75 – sharp / high quality
      0.40 – 0.75 – acceptable
      < 0.40 – blurry or very small crop

    Formula: score = 1 − exp(−variance / 300)
    Typical Laplacian variance values:
      < 20    → heavily blurred
      20–150  → soft / motion blur
      150–500 → acceptable
      > 500   → sharp / high-res crop
    """
    x = int(facial_area.get("x", 0))
    y = int(facial_area.get("y", 0))
    w = int(facial_area.get("w", 0))
    h = int(facial_area.get("h", 0))

    if w < 8 or h < 8 or img is None:
        return 0.0

    # Clamp crop to image bounds
    ih, iw = img.shape[:2]
    x2, y2 = min(x + w, iw), min(y + h, ih)
    crop = img[max(0, y):y2, max(0, x):x2]
    if crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score   = 1.0 - float(np.exp(-lap_var / 300.0))
    return round(min(1.0, max(0.0, score)), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  META
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    tags=["Meta"],
    summary="Service health & configuration",
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "example": {
                            "status": "ok",
                            "config": {
                                "model": "VGG-Face",
                                "detector": "retinaface",
                                "max_upload_mb": 20,
                                "face_ttl_seconds": 86400,
                                "group_session_ttl_seconds": 604800
                            },
                            "cache": {
                                "faces": {"total": 142, "active": 138, "expired": 4},
                                "group_sessions_active": 7
                            },
                            "queue": {
                                "pending": 0, "running": 1,
                                "max_size": 50, "concurrency": 1,
                                "job_counts": {"DONE": 23, "RUNNING": 1}
                            }
                        }
                    }
                }
            }
        }
    }
)
def health():
    """Returns current status, active configuration, and cache statistics."""
    stats    = db.face_cache_stats()
    sessions = db.count_active_sessions()
    return {
        "status": "ok",
        "config": {
            "model": MODEL_NAME,
            "detector": DETECTOR_BACKEND,
            "max_upload_mb": MAX_UPLOAD_MB,
            "face_ttl_seconds": FACE_TTL_SEC,
            "group_session_ttl_seconds": GROUP_TTL_SEC,
        },
        "cache": {
            "faces": stats,
            "group_sessions_active": sessions,
        },
        "queue": qw.queue_stats(),
    }


@app.get("/admin", include_in_schema=False)
def admin_ui():
    """Serve the built-in Admin UI."""
    return FileResponse(str(_ADMIN_HTML), media_type="text/html")


@app.post("/admin/purge", tags=["Meta"], summary="Manually purge expired records")
def manual_purge():
    """Force-removes all expired face IDs and group sessions from the database."""
    try:
        n = db.purge_expired()
    except Exception as exc:
        raise err_db(exc)
    return {"purged_records": n}


# ══════════════════════════════════════════════════════════════════════════════
#  FACE CACHE  (management)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/faces/cache", tags=["Face Cache"], summary="List cached face IDs")
def list_face_cache(
    limit:  Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)]         = 0,
):
    """
    Returns all **active** (non-expired) face IDs in the cache.
    Embeddings are omitted here — use `GET /faces/cache/{face_id}` to retrieve one.
    """
    try:
        faces = db.list_faces(limit=limit, offset=offset)
    except Exception as exc:
        raise err_db(exc)
    return {"count": len(faces), "offset": offset, "faces": faces}


@app.get("/faces/cache/{face_id}", tags=["Face Cache"], summary="Retrieve a cached face")
def get_cached_face(
    face_id: str,
    include_embedding: bool = Query(True, description="Set false to omit the raw vector"),
):
    """
    Retrieve metadata and optionally the embedding vector for a cached face.
    Returns **404** if the face_id is unknown or has expired.
    """
    try:
        face = db.get_face(face_id)
    except Exception as exc:
        raise err_db(exc)
    if face is None:
        raise err_face_not_found(face_id)
    if not include_embedding:
        face.pop("embedding", None)
    return face


@app.delete("/faces/cache/{face_id}", tags=["Face Cache"], summary="Delete a cached face")
def delete_cached_face(face_id: str):
    """Immediately removes a face ID and its embedding from the cache."""
    try:
        ok = db.delete_face(face_id)
    except Exception as exc:
        raise err_db(exc)
    if not ok:
        raise err_face_not_found(face_id)
    return {"deleted": True, "face_id": face_id}


# ══════════════════════════════════════════════════════════════════════════════
#  FACE EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/faces/embed",
    tags=["Faces"],
    summary="Detect faces & compute embeddings",
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "examples": {
                            "two_faces": {
                                "summary": "2 Gesichter erkannt und gecacht",
                                "value": {
                                    "source_file": "gruppe.jpg",
                                    "faces_found": 2,
                                    "model": "VGG-Face",
                                    "faces": [
                                        {
                                            "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
                                            "face_index": 0,
                                            "face_confidence": 0.9821,
                                            "facial_area": {"x": 124, "y": 89, "w": 156, "h": 178},
                                            "embedding": [0.0312, -0.1547, 0.2089, -0.0743, "...(4096 total)"],
                                            "embedding_dim": 4096,
                                            "cached": True,
                                            "expires_at": 1712518400.0
                                        },
                                        {
                                            "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
                                            "face_index": 1,
                                            "face_confidence": 0.9543,
                                            "facial_area": {"x": 380, "y": 102, "w": 143, "h": 165},
                                            "embedding": [-0.0821, 0.1234, -0.0456, 0.2341, "...(4096 total)"],
                                            "embedding_dim": 4096,
                                            "cached": True,
                                            "expires_at": 1712518400.0
                                        }
                                    ]
                                }
                            },
                            "no_cache": {
                                "summary": "cache=false – Vektoren ohne Persistenz",
                                "value": {
                                    "source_file": "portrait.jpg",
                                    "faces_found": 1,
                                    "model": "VGG-Face",
                                    "faces": [
                                        {
                                            "face_id": None,
                                            "face_index": 0,
                                            "face_confidence": 0.997,
                                            "facial_area": {"x": 50, "y": 30, "w": 200, "h": 220},
                                            "embedding": [0.1123, -0.0934, "...(4096 total)"],
                                            "embedding_dim": 4096,
                                            "cached": False,
                                            "expires_at": None
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            },
            "422": {
                "description": "Kein Gesicht im Bild erkannt",
                "content": {
                    "application/json": {
                        "example": {
                            "error": True,
                            "code": "NO_FACE_DETECTED",
                            "message": "No faces could be detected in the image. Try a higher-resolution image or a different detector backend.",
                            "detail": {"image": "landschaft.jpg"}
                        }
                    }
                }
            },
            "413": {
                "description": "Datei zu groß",
                "content": {
                    "application/json": {
                        "example": {
                            "error": True,
                            "code": "FILE_TOO_LARGE",
                            "message": "Uploaded file exceeds the 20 MB limit.",
                            "detail": None
                        }
                    }
                }
            }
        }
    }
)
async def embed_faces(
    file:  UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, BMP, TIFF)"),
    cache: bool = Query(True, description="Store embeddings in the face cache"),
    ttl:   int  = Query(None, ge=60, description="Cache TTL in seconds (overrides default)"),
):
    """
    Detects **all faces** in the uploaded image and computes a deep embedding
    vector for each one.

    ### Returns (per face)
    | Field | Description |
    |---|---|
    | `face_id` | Stable UUID — pass to `/faces/cluster-group` later |
    | `embedding` | Raw float vector (dimension depends on model) |
    | `facial_area` | Bounding box `{x, y, w, h}` |
    | `face_confidence` | Detector confidence (0–1) |
    | `expires_at` | Unix timestamp when the cache entry expires |

    Set `cache=false` if you only need the vectors and don't want persistence.
    """
    data = await file.read()
    _validate_size(data)
    tmp = _save_tmp(data)

    try:
        _preprocess_inplace(tmp)
        img_bgr = cv2.imread(str(tmp))    # load once for quality scoring
        raw = await _represent(str(tmp))
    except VisionAPIException:
        raise
    except Exception as exc:
        raise err_inference(exc)
    finally:
        tmp.unlink(missing_ok=True)

    if not raw:
        raise err_no_face(file.filename or "")

    # ── Confidence filtering ───────────────────────────────────────────────
    if MIN_FACE_CONFIDENCE > 0.0:
        before = len(raw)
        raw = [r for r in raw if (r.get("face_confidence") or 0.0) >= MIN_FACE_CONFIDENCE]
        if len(raw) < before:
            log.debug(
                "Filtered %d low-confidence face(s) (threshold=%.2f)",
                before - len(raw), MIN_FACE_CONFIDENCE,
            )
    if not raw:
        raise err_no_face(file.filename or "")

    effective_ttl = ttl or FACE_TTL_SEC
    results = []
    for idx, item in enumerate(raw):
        embedding   = item["embedding"]
        facial_area = item.get("facial_area", {})
        confidence  = item.get("face_confidence")
        quality     = _face_quality_score(img_bgr, facial_area) if img_bgr is not None else None

        face_id    = None
        expires_at = None
        if cache:
            try:
                face_id = db.cache_face(
                    embedding=embedding,
                    model_name=MODEL_NAME,
                    ttl_seconds=effective_ttl,
                    facial_area=facial_area,
                    face_confidence=confidence,
                    face_quality_score=quality,
                    source_file=file.filename,
                    face_index=idx,
                )
                expires_at = time.time() + effective_ttl
            except Exception as exc:
                log.warning("Could not cache face: %s", exc)

        results.append({
            "face_id":            face_id,
            "face_index":         idx,
            "face_confidence":    confidence,
            "face_quality_score": quality,
            "facial_area":        facial_area,
            "embedding":          embedding,
            "embedding_dim":      len(embedding),
            "cached":             cache and face_id is not None,
            "expires_at":         expires_at,
        })

    return {
        "source_file":  file.filename,
        "faces_found":  len(results),
        "model":        MODEL_NAME,
        "faces":        results,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FACE CLUSTERING  (single image)
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/faces/cluster",
    tags=["Faces"],
    summary="Detect & cluster faces in one image",
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "examples": {
                            "five_faces_three_clusters": {
                                "summary": "5 Gesichter, 3 Cluster, 1 Noise",
                                "value": {
                                    "source_file": "gruppenphoto.jpg",
                                    "faces_found": 5,
                                    "n_clusters": 3,
                                    "n_noise": 1,
                                    "model": "VGG-Face",
                                    "min_similarity": 0.65,
                                    "faces": [
                                        {"face_id": "aabb1122-ccdd-3344-eeff-556677889900", "face_index": 0, "cluster_id": 0, "face_confidence": 0.987, "facial_area": {"x": 50, "y": 30, "w": 120, "h": 140}, "cached": True},
                                        {"face_id": "bbcc2233-ddee-4455-ffaa-667788990011", "face_index": 1, "cluster_id": 0, "face_confidence": 0.963, "facial_area": {"x": 210, "y": 45, "w": 118, "h": 138}, "cached": True},
                                        {"face_id": "ccdd3344-eeff-5566-aabb-778899001122", "face_index": 2, "cluster_id": 1, "face_confidence": 0.941, "facial_area": {"x": 400, "y": 55, "w": 130, "h": 150}, "cached": True},
                                        {"face_id": "ddee4455-ffaa-6677-bbcc-889900112233", "face_index": 3, "cluster_id": 2, "face_confidence": 0.978, "facial_area": {"x": 580, "y": 40, "w": 125, "h": 145}, "cached": True},
                                        {"face_id": "eeff5566-aabb-7788-ccdd-990011223344", "face_index": 4, "cluster_id": -1, "face_confidence": 0.512, "facial_area": {"x": 720, "y": 80, "w": 90, "h": 100}, "cached": True}
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def cluster_faces(
    file: UploadFile = File(...),
    min_similarity: float = Query(0.60, ge=0.0, le=1.0,
        description="Cosine similarity threshold — higher = stricter (0–1)"),
    cache: bool = Query(True,  description="Store embeddings in the face cache"),
    ttl:   int  = Query(None, ge=60, description="Cache TTL override in seconds"),
):
    """
    Detects all faces, embeds them, and assigns each a **cluster label** using
    DBSCAN with cosine distance.

    ### cluster_id values
    - `≥ 0` — cluster label (same label = similar faces)
    - `-1` — noise (face found but not similar enough to any cluster)

    For **cross-image** grouping of many clusters, see `/faces/cluster-group`.
    """
    data = await file.read()
    _validate_size(data)
    tmp = _save_tmp(data)

    try:
        _preprocess_inplace(tmp)
        img_bgr = cv2.imread(str(tmp))
        raw = await _represent(str(tmp))
    except VisionAPIException:
        raise
    except Exception as exc:
        raise err_inference(exc)
    finally:
        tmp.unlink(missing_ok=True)

    if not raw:
        raise err_no_face(file.filename or "")

    # ── Confidence filtering ───────────────────────────────────────────────
    if MIN_FACE_CONFIDENCE > 0.0:
        raw = [r for r in raw if (r.get("face_confidence") or 0.0) >= MIN_FACE_CONFIDENCE]
    if not raw:
        raise err_no_face(file.filename or "")

    try:
        vectors_norm = cl.normalize_vectors(np.array([e["embedding"] for e in raw]))
        labels       = cl.run_dbscan(vectors_norm, min_similarity)
    except Exception as exc:
        raise err_clustering(exc)

    effective_ttl = ttl or FACE_TTL_SEC
    faces = []
    for i, (item, label) in enumerate(zip(raw, labels)):
        quality = _face_quality_score(img_bgr, item.get("facial_area", {})) if img_bgr is not None else None
        face_id = None
        if cache:
            try:
                face_id = db.cache_face(
                    embedding=item["embedding"],
                    model_name=MODEL_NAME,
                    ttl_seconds=effective_ttl,
                    facial_area=item.get("facial_area"),
                    face_confidence=item.get("face_confidence"),
                    face_quality_score=quality,
                    source_file=file.filename,
                    face_index=i,
                )
            except Exception as exc:
                log.warning("Cache write failed: %s", exc)

        faces.append({
            "face_id":            face_id,
            "face_index":         i,
            "cluster_id":         label,
            "face_confidence":    item.get("face_confidence"),
            "face_quality_score": quality,
            "facial_area":        item.get("facial_area", {}),
            "cached":             cache and face_id is not None,
        })

    return {
        "source_file":    file.filename,
        "faces_found":    len(faces),
        "n_clusters":     len({l for l in labels if l >= 0}),
        "n_noise":        sum(1 for l in labels if l < 0),
        "model":          MODEL_NAME,
        "min_similarity": min_similarity,
        "faces":          faces,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER GROUPING
# ══════════════════════════════════════════════════════════════════════════════

class ClusterItemEmbedding(BaseModel):
    cluster_id: str           = Field(..., description="Your stable ID for this cluster")
    centroid:   list[float]   = Field(..., description="Representative embedding vector")
    size:       int           = Field(1, ge=1, description="Number of faces in this cluster")
    metadata:   Optional[dict] = Field(default_factory=dict)


class ClusterItemFaceId(BaseModel):
    cluster_id: str           = Field(..., description="Your stable ID for this cluster")
    face_id:    str           = Field(..., description="face_id from a previous /faces/embed call")
    size:       int           = Field(1, ge=1)
    metadata:   Optional[dict] = Field(default_factory=dict)


class GroupingRequest(BaseModel):
    """
    Send **`clusters`** (raw centroid vectors) and/or **`face_id_clusters`**
    (cached face IDs). Both lists are merged before grouping.
    """
    clusters: list[ClusterItemEmbedding] = Field(
        default_factory=list,
        description="Clusters with inline centroid vectors",
    )
    face_id_clusters: list[ClusterItemFaceId] = Field(
        default_factory=list,
        description="Clusters referenced only by cached face_id — no vector needed",
    )
    method: str = Field(
        "hdbscan",
        description="**hdbscan** | **agglomerative** | **kmeans**",
    )
    n_groups: Optional[int] = Field(
        None, ge=2,
        description="Target group count — required for `kmeans`, optional for `agglomerative`",
    )
    min_cluster_size: int = Field(
        2, ge=2,
        description="HDBSCAN: minimum cluster density",
    )
    distance_threshold: float = Field(
        0.40, ge=0.0, le=2.0,
        description="Agglomerative: cosine distance cutoff",
    )
    persist: bool = Field(True, description="Save result as a retrievable session")
    session_ttl: Optional[int] = Field(
        None, ge=3600,
        description="Session TTL in seconds (server default: 7 days)",
    )

    @field_validator("method")
    @classmethod
    def _valid_method(cls, v: str) -> str:
        if v not in ("hdbscan", "agglomerative", "kmeans"):
            raise ValueError(f"Unknown method '{v}'")
        return v


@app.post(
    "/faces/cluster-group",
    tags=["Faces"],
    summary="Group many clusters into person-groups (advanced / large-scale)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "mode_a_hdbscan": {
                            "summary": "Mode A – raw centroids, HDBSCAN",
                            "value": {
                                "clusters": [
                                    {"cluster_id": "foto001-gesicht0", "centroid": [0.0312, -0.1547, 0.2089, -0.0743], "size": 3, "metadata": {"quelle": "urlaub.jpg"}},
                                    {"cluster_id": "foto001-gesicht1", "centroid": [-0.0821, 0.1234, -0.0456, 0.2341], "size": 2, "metadata": {"quelle": "urlaub.jpg"}},
                                    {"cluster_id": "foto002-gesicht0", "centroid": [0.0298, -0.1531, 0.2101, -0.0758], "size": 5, "metadata": {"quelle": "geburtstag.jpg"}},
                                    {"cluster_id": "foto003-gesicht0", "centroid": [-0.0835, 0.1218, -0.0471, 0.2329], "size": 1, "metadata": {"quelle": "weihnachten.jpg"}}
                                ],
                                "method": "hdbscan",
                                "min_cluster_size": 2,
                                "persist": True
                            }
                        },
                        "mode_b_face_ids": {
                            "summary": "Mode B – nur face_ids aus Cache",
                            "value": {
                                "face_id_clusters": [
                                    {"cluster_id": "session-a-p1", "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d", "size": 4, "metadata": {"album": "Sommer 2023"}},
                                    {"cluster_id": "session-a-p2", "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d", "size": 2, "metadata": {"album": "Sommer 2023"}},
                                    {"cluster_id": "session-b-p1", "face_id": "aabb1122-ccdd-3344-eeff-556677889900", "size": 7, "metadata": {"album": "Winter 2023"}}
                                ],
                                "method": "agglomerative",
                                "distance_threshold": 0.35,
                                "persist": True,
                                "session_ttl": 259200
                            }
                        },
                        "mode_c_kmeans": {
                            "summary": "Mode C – gemischt + kmeans",
                            "value": {
                                "clusters": [
                                    {"cluster_id": "archiv-001", "centroid": [0.1231, -0.0892, 0.2341, 0.0543], "size": 12}
                                ],
                                "face_id_clusters": [
                                    {"cluster_id": "neu-gesicht0", "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d", "size": 2}
                                ],
                                "method": "kmeans",
                                "n_groups": 2,
                                "persist": False
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "example": {
                            "session_id": "9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f",
                            "method": "hdbscan",
                            "input_clusters": 4,
                            "n_groups": 2,
                            "n_noise": 0,
                            "persisted": True,
                            "groups": [
                                {
                                    "group_id": 0,
                                    "cluster_count": 2,
                                    "members": [
                                        {"cluster_id": "foto001-gesicht0", "face_id": None, "size": 3, "metadata": {"quelle": "urlaub.jpg"}},
                                        {"cluster_id": "foto002-gesicht0", "face_id": None, "size": 5, "metadata": {"quelle": "geburtstag.jpg"}}
                                    ]
                                },
                                {
                                    "group_id": 1,
                                    "cluster_count": 2,
                                    "members": [
                                        {"cluster_id": "foto001-gesicht1", "face_id": None, "size": 2, "metadata": {"quelle": "urlaub.jpg"}},
                                        {"cluster_id": "foto003-gesicht0", "face_id": None, "size": 1, "metadata": {"quelle": "weihnachten.jpg"}}
                                    ]
                                }
                            ]
                        }
                    }
                }
            },
            "400": {
                "description": "Dimension-Mismatch oder zu wenige Cluster",
                "content": {
                    "application/json": {
                        "examples": {
                            "dim_mismatch": {
                                "summary": "Verschiedene Modelle gemischt",
                                "value": {"error": True, "code": "EMBEDDING_DIM_MISMATCH", "message": "Embedding dimension mismatch for face_id 'cluster-x': expected 4096, got 512.", "detail": {"face_id": "cluster-x", "expected": 4096, "got": 512}}
                            },
                            "too_few": {
                                "summary": "Zu wenige Cluster",
                                "value": {"error": True, "code": "TOO_FEW_CLUSTERS", "message": "Need at least 2 clusters to group, got 1.", "detail": {"received": 1}}
                            }
                        }
                    }
                }
            }
        }
    }
)
def group_clusters(req: GroupingRequest):
    """
    **Advanced large-scale grouping** for pre-computed centroids or mixed sources.

    > **For the common workflow** (upload photos → get face_ids → group them),
    > use the simpler **`POST /faces/group`** endpoint instead.

    Accepts clusters from multiple images and merges them into person-groups
    using one of three algorithms.

    ---

    ### Input modes

    **Mode A – raw centroids**
    ```json
    {
      "clusters": [
        { "cluster_id": "photo1-face0", "centroid": [0.12, -0.34, ...], "size": 3 }
      ]
    }
    ```

    **Mode B – face IDs only** (embeddings loaded from cache)
    ```json
    {
      "face_id_clusters": [
        { "cluster_id": "photo1-face0", "face_id": "550e8400-...", "size": 3 }
      ]
    }
    ```

    **Mode C – mixed** (both fields are merged before grouping)

    ---

    ### Algorithms

    | Method | Best for | Key param |
    |---|---|---|
    | `hdbscan` *(default)* | Unknown group count, noisy data | `min_cluster_size` |
    | `agglomerative` | Hierarchical merging | `distance_threshold` or `n_groups` |
    | `kmeans` | Exact fixed group count | `n_groups` (required) |

    ---

    ### Persistence
    Results are stored as a **session**. Use `GET /sessions/{session_id}` to
    retrieve them. Set `persist=false` to skip storage.
    """

    # ── Resolve face_id clusters ──────────────────────────────────────────────
    resolved: list[dict] = []

    if req.face_id_clusters:
        ids = [fi.face_id for fi in req.face_id_clusters]
        try:
            cached = db.get_faces(ids)
        except Exception as exc:
            raise err_db(exc)

        for fi in req.face_id_clusters:
            face = cached.get(fi.face_id)
            if face is None:
                raise err_face_not_found(fi.face_id)
            resolved.append({
                "cluster_id": fi.cluster_id,
                "face_id":    fi.face_id,
                "centroid":   face["embedding"],
                "size":       fi.size,
                "metadata":   fi.metadata or {},
            })

    for c in req.clusters:
        resolved.append({
            "cluster_id": c.cluster_id,
            "face_id":    None,
            "centroid":   c.centroid,
            "size":       c.size,
            "metadata":   c.metadata or {},
        })

    if len(resolved) < 2:
        raise err_too_few_clusters(len(resolved))

    # ── Validate embedding dimensions ─────────────────────────────────────────
    first_dim = len(resolved[0]["centroid"])
    for r in resolved[1:]:
        if len(r["centroid"]) != first_dim:
            raise err_dim_mismatch(first_dim, len(r["centroid"]), r["cluster_id"])

    # ── Cluster ───────────────────────────────────────────────────────────────
    centroids = np.array([r["centroid"] for r in resolved], dtype=np.float32)
    try:
        norm = cl.normalize_vectors(centroids)
        if req.method == "hdbscan":
            labels = cl.run_hdbscan(norm, req.min_cluster_size)
        elif req.method == "agglomerative":
            labels = cl.run_agglomerative(norm, req.n_groups, req.distance_threshold)
        else:  # kmeans
            if not req.n_groups:
                raise err_missing_param("n_groups", "required for kmeans")
            labels = cl.run_kmeans(norm, req.n_groups)
    except VisionAPIException:
        raise
    except Exception as exc:
        raise err_clustering(exc)

    # ── Build output ──────────────────────────────────────────────────────────
    member_items = [
        {
            "cluster_id": r["cluster_id"],
            "face_id":    r["face_id"],
            "size":       r["size"],
            "metadata":   r["metadata"],
        }
        for r in resolved
    ]
    groups  = cl.build_groups(member_items, labels)
    n_noise = sum(1 for l in labels if l < 0)

    # ── Persist ───────────────────────────────────────────────────────────────
    session_id = None
    if req.persist:
        try:
            session_id = db.save_group_session(
                method=req.method,
                params={
                    "n_groups":           req.n_groups,
                    "min_cluster_size":   req.min_cluster_size,
                    "distance_threshold": req.distance_threshold,
                },
                groups=groups,
                n_input=len(resolved),
                n_noise=n_noise,
                ttl_seconds=req.session_ttl or GROUP_TTL_SEC,
            )
        except Exception as exc:
            log.warning("Failed to persist group session: %s", exc)

    return {
        "session_id":     session_id,
        "method":         req.method,
        "input_clusters": len(resolved),
        "n_groups":       len(groups),
        "n_noise":        n_noise,
        "persisted":      session_id is not None,
        "groups":         groups,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE FACE GROUPING  (new primary endpoint)
# ══════════════════════════════════════════════════════════════════════════════

class GroupByFaceIdsRequest(BaseModel):
    """
    Group a collection of cached faces by identity.

    Pass the `face_id` values you received from `/faces/embed` or
    `/faces/cluster`.  The server fetches the embeddings, runs the chosen
    algorithm, and returns **groups as arrays of face_ids** — no prior
    knowledge of the group count required.
    """
    face_ids: list[str] = Field(
        ..., min_length=2,
        description="Two or more `face_id` values from the cache",
    )
    method: str = Field(
        "hdbscan",
        description=(
            "**`connected`** — graph-based connected components: every pair of faces above "
            "`similarity_threshold` ends up in the same group. Most intuitive. No noise labels.  "
            "**`hdbscan`** — density-based, auto group count, supports noise.  "
            "**`agglomerative`** — hierarchical merging by distance threshold."
        ),
    )
    similarity_threshold: float = Field(
        0.65, ge=0.0, le=1.0,
        description=(
            "Connected components: cosine similarity required to link two faces into one group.  "
            "0.60 = loose · 0.68 = ArcFace default · 0.75 = strict.  Default: `0.65`"
        ),
    )
    min_cluster_size: int = Field(
        2, ge=2,
        description=(
            "HDBSCAN: minimum number of faces to form a group.  "
            "Lower values → more, smaller groups.  Default: `2`"
        ),
    )
    distance_threshold: float = Field(
        0.45, ge=0.0, le=2.0,
        description=(
            "Agglomerative: cosine-distance cutoff between groups.  "
            "Lower → fewer, tighter groups.  Default: `0.45`"
        ),
    )
    persist: bool = Field(
        True,
        description="Persist the result as a retrievable session (see `GET /sessions/{id}`)",
    )
    session_ttl: Optional[int] = Field(
        None, ge=3600,
        description="Session TTL in seconds (server default: 7 days)",
    )

    @field_validator("method")
    @classmethod
    def _valid_method(cls, v: str) -> str:
        allowed = ("connected", "hdbscan", "agglomerative")
        if v not in allowed:
            raise ValueError(f"Unknown method '{v}'. Use one of: {allowed}")
        return v


@app.post(
    "/faces/group",
    tags=["Faces"],
    summary="Group cached faces by identity",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "hdbscan_auto": {
                            "summary": "HDBSCAN – fully automatic group count",
                            "value": {
                                "face_ids": [
                                    "11111111-0000-0000-0000-000000000001",
                                    "11111111-0000-0000-0000-000000000002",
                                    "22222222-0000-0000-0000-000000000001",
                                    "22222222-0000-0000-0000-000000000002",
                                    "33333333-0000-0000-0000-000000000001",
                                ],
                                "method": "hdbscan",
                                "min_cluster_size": 2,
                                "persist": True,
                            },
                        },
                        "agglomerative_tight": {
                            "summary": "Agglomerative – tighter threshold",
                            "value": {
                                "face_ids": [
                                    "11111111-0000-0000-0000-000000000001",
                                    "11111111-0000-0000-0000-000000000002",
                                    "22222222-0000-0000-0000-000000000001",
                                ],
                                "method": "agglomerative",
                                "distance_threshold": 0.35,
                                "persist": False,
                            },
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "example": {
                            "groups": [
                                [
                                    "11111111-0000-0000-0000-000000000001",
                                    "11111111-0000-0000-0000-000000000002",
                                ],
                                [
                                    "22222222-0000-0000-0000-000000000001",
                                    "22222222-0000-0000-0000-000000000002",
                                ],
                            ],
                            "noise_face_ids": ["33333333-0000-0000-0000-000000000001"],
                            "n_groups": 2,
                            "n_noise": 1,
                            "n_input": 5,
                            "method": "hdbscan",
                            "session_id": "9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f",
                            "persisted": True,
                        }
                    }
                }
            }
        },
    },
)
def group_by_face_ids(req: GroupByFaceIdsRequest):
    """
    **The primary grouping endpoint.**

    Upload photos → collect `face_id` values from `/faces/embed` →
    send them here → receive back groups (arrays of face_ids).

    You do **not** need to know how many groups exist up front.
    Both supported algorithms discover the group count automatically:

    | Method | How it works | Key param |
    |---|---|---|
    | `hdbscan` *(default)* | Density-based; noise-tolerant | `min_cluster_size` (default 2) |
    | `agglomerative` | Hierarchical merging | `distance_threshold` (default 0.45) |

    ### Response structure
    ```json
    {
      "groups": [
        ["face_id_A1", "face_id_A2"],   // group 0 – person A
        ["face_id_B1", "face_id_B2"]    // group 1 – person B
      ],
      "noise_face_ids": ["face_id_X"],  // unassigned (HDBSCAN only)
      "n_groups": 2,
      "n_noise":  1,
      "n_input":  5
    }
    ```

    ### Notes
    - Duplicate `face_id` values are silently deduplicated.
    - All face_ids must belong to the **same embedding model** (same `model_name`
      in the cache). Mixing models raises `EMBEDDING_DIM_MISMATCH`.
    - With `persist=true` (default) the result is stored; retrieve it with
      `GET /sessions/{session_id}`.
    """
    # ── Deduplicate while preserving order ───────────────────────────────────
    seen: set[str] = set()
    unique_ids: list[str] = []
    for fid in req.face_ids:
        if fid not in seen:
            seen.add(fid)
            unique_ids.append(fid)

    if len(unique_ids) < 2:
        raise err_too_few_clusters(len(unique_ids))

    # ── Fetch embeddings from cache ───────────────────────────────────────────
    try:
        cached = db.get_faces(unique_ids)
    except Exception as exc:
        raise err_db(exc)

    missing = [fid for fid in unique_ids if fid not in cached]
    if missing:
        raise err_face_not_found(missing[0])

    # ── Validate embedding dimensions (must not mix models) ───────────────────
    dims = {cached[fid]["embedding_dim"] for fid in unique_ids}
    if len(dims) > 1:
        # find the mismatching pair for a helpful error message
        first_dim = cached[unique_ids[0]]["embedding_dim"]
        for fid in unique_ids[1:]:
            if cached[fid]["embedding_dim"] != first_dim:
                raise err_dim_mismatch(first_dim, cached[fid]["embedding_dim"], fid)

    # ── Build normalised embedding matrix ────────────────────────────────────
    embeddings = np.array(
        [cached[fid]["embedding"] for fid in unique_ids], dtype=np.float32
    )
    try:
        norm = cl.normalize_vectors(embeddings)
        if req.method == "connected":
            labels = cl.run_connected_components(norm, req.similarity_threshold)
        elif req.method == "hdbscan":
            labels = cl.run_hdbscan(norm, req.min_cluster_size)
        else:  # agglomerative
            labels = cl.run_agglomerative(norm, None, req.distance_threshold)
    except VisionAPIException:
        raise
    except Exception as exc:
        raise err_clustering(exc)

    # ── Map labels → groups ───────────────────────────────────────────────────
    group_map: dict[int, list[str]] = {}
    noise_ids: list[str] = []
    for fid, label in zip(unique_ids, labels):
        if label < 0:
            noise_ids.append(fid)
        else:
            group_map.setdefault(int(label), []).append(fid)

    groups = [group_map[k] for k in sorted(group_map)]

    # ── Persist ───────────────────────────────────────────────────────────────
    session_id = None
    if req.persist:
        session_groups = [
            {
                "group_id": i,
                "members": [
                    {"cluster_id": fid, "face_id": fid, "size": 1, "metadata": {}}
                    for fid in grp
                ],
            }
            for i, grp in enumerate(groups)
        ]
        if noise_ids:
            session_groups.append({
                "group_id": -1,
                "members": [
                    {"cluster_id": fid, "face_id": fid, "size": 1, "metadata": {}}
                    for fid in noise_ids
                ],
            })
        try:
            session_id = db.save_group_session(
                method=req.method,
                params={
                    "similarity_threshold": req.similarity_threshold,
                    "min_cluster_size":     req.min_cluster_size,
                    "distance_threshold":   req.distance_threshold,
                },
                groups=session_groups,
                n_input=len(unique_ids),
                n_noise=len(noise_ids),
                ttl_seconds=req.session_ttl or GROUP_TTL_SEC,
            )
        except Exception as exc:
            log.warning("Failed to persist group session: %s", exc)

    return {
        "groups":         groups,
        "noise_face_ids": noise_ids,
        "n_groups":       len(groups),
        "n_noise":        len(noise_ids),
        "n_input":        len(unique_ids),
        "method":         req.method,
        "session_id":     session_id,
        "persisted":      session_id is not None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GROUP SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/sessions", tags=["Sessions"], summary="List active group sessions")
def list_sessions(limit: Annotated[int, Query(ge=1, le=500)] = 50):
    """Returns all active (non-expired) group sessions, newest first."""
    try:
        sessions = db.list_group_sessions(limit=limit)
    except Exception as exc:
        raise err_db(exc)
    return {"count": len(sessions), "sessions": sessions}


@app.get("/sessions/{session_id}", tags=["Sessions"], summary="Retrieve a group session")
def get_session(session_id: str):
    """
    Returns the full grouping result — all groups and their member clusters.
    Returns **404** if the session is not found or has expired.
    """
    try:
        session = db.get_group_session(session_id)
    except Exception as exc:
        raise err_db(exc)
    if session is None:
        raise err_session_not_found(session_id)
    return session


@app.delete("/sessions/{session_id}", tags=["Sessions"], summary="Delete a group session")
def delete_session(session_id: str):
    """Immediately deletes a group session and all its member assignments."""
    try:
        ok = db.delete_group_session(session_id)
    except Exception as exc:
        raise err_db(exc)
    if not ok:
        raise err_session_not_found(session_id)
    return {"deleted": True, "session_id": session_id}


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING PROJECTION  (dimensionality reduction for visualisation)
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/faces/projection",
    tags=["Faces"],
    summary="Project face embeddings to 2-D or 3-D for visualisation",
)
def face_projection(
    method: str = Query("pca", description="Reduction algorithm: pca | tsne | umap"),
    dims:   int  = Query(2,     ge=2, le=3, description="Output dimensions: 2 or 3"),
    session_id: Optional[str] = Query(None, description="Colour points by this group session"),
    limit:  int  = Query(2000, ge=1, le=10000, description="Max faces to project"),
):
    """
    Reduces all cached face embeddings to **2-D or 3-D** coordinates suitable
    for scatter-plot visualisation.

    ### Algorithms
    | `method` | Speed | Quality | Notes |
    |---|---|---|---|
    | `pca` | ⚡⚡⚡ | ⭐⭐ | Linear, deterministic, instant |
    | `tsne` | ⚡ | ⭐⭐⭐⭐ | Best cluster separation, slow on >2 k points |
    | `umap` | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Best overall — requires `umap-learn` package |

    ### `session_id` colouring
    Pass a group `session_id` to map each point to its `group_id` so the
    scatter plot renders each person cluster in a distinct colour.

    ### Response shape
    ```json
    {
      "method": "pca", "dims": 2, "n_points": 142,
      "points": [
        {"face_id": "abc", "x": 1.2, "y": -0.4, "z": null,
         "group_id": 2, "source_file": "photo.jpg", "quality": 0.87}
      ]
    }
    ```
    """
    # ── 1. Load embeddings from cache ─────────────────────────────────────────
    try:
        rows = db.get_all_embeddings(limit=limit)
    except Exception as exc:
        raise err_db(exc)

    if len(rows) < 2:
        return {"method": method, "dims": dims, "n_points": len(rows), "points": []}

    # ── 2. Build matrix  (n × d) ──────────────────────────────────────────────
    ids     = [r["face_id"]           for r in rows]
    sources = [r.get("source_file")   for r in rows]
    quals   = [r.get("face_quality_score") for r in rows]
    mat     = np.array([r["embedding"] for r in rows], dtype=np.float32)

    # L2-normalise (ArcFace embeddings benefit from this for cosine geometry)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat   = mat / np.where(norms == 0, 1.0, norms)

    # ── 3. Dimensionality reduction ───────────────────────────────────────────
    n_components = dims
    try:
        m = method.lower()
        if m == "pca":
            from sklearn.decomposition import PCA
            # PCA can produce at most min(n_samples-1, n_features) components
            n_components = min(dims, len(rows) - 1, mat.shape[1])
            coords = PCA(n_components=n_components, random_state=42).fit_transform(mat)

        elif m == "tsne":
            from sklearn.manifold import TSNE
            n_samples = len(rows)
            # t-SNE needs at least n_components + 1 samples
            if n_samples < n_components + 1:
                raise VisionAPIException(
                    422, ErrorCode.INFERENCE_FAILED,
                    f"t-SNE requires at least {n_components + 1} faces for "
                    f"{n_components}D projection, but only {n_samples} found.",
                )
            # PCA pre-reduce to 50 dims for speed when embeddings are high-dim
            pre = min(50, mat.shape[1], n_samples - 1)
            if mat.shape[1] > pre and pre >= 1:
                from sklearn.decomposition import PCA as _PCA
                mat = _PCA(n_components=pre, random_state=42).fit_transform(mat)
            # perplexity must be strictly less than n_samples
            perplexity = min(30.0, max(2.0, n_samples / 4))
            perplexity = min(perplexity, n_samples - 1.0)
            coords = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                max_iter=500,
                random_state=42,
                n_jobs=-1,
            ).fit_transform(mat)

        elif m == "umap":
            try:
                import umap  # type: ignore
            except ImportError:
                raise VisionAPIException(
                    422, ErrorCode.UNKNOWN_METHOD,
                    "UMAP is not installed. Add 'umap-learn' to requirements.txt.",
                )
            n_neighbors = max(2, min(15, len(rows) - 1))
            if len(rows) < 2:
                raise VisionAPIException(
                    422, ErrorCode.INFERENCE_FAILED,
                    f"UMAP requires at least 2 faces, but only {len(rows)} found.",
                )
            # PCA pre-reduce high-dim embeddings for speed (keep more dims
            # than t-SNE for better UMAP quality)
            pre = min(100, mat.shape[1], len(rows) - 1)
            if mat.shape[1] > pre and pre >= 1:
                from sklearn.decomposition import PCA as _PCA2
                mat = _PCA2(n_components=pre, random_state=42).fit_transform(mat)
            coords = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                n_epochs=200,
                low_memory=True,
                random_state=42,
            ).fit_transform(mat)

        else:
            raise VisionAPIException(
                422, ErrorCode.UNKNOWN_METHOD,
                f"Unknown projection method '{method}'. Use: pca | tsne | umap",
            )
    except VisionAPIException:
        raise
    except Exception as exc:
        raise VisionAPIException(
            500, ErrorCode.INFERENCE_FAILED,
            "Projection failed.", detail=str(exc),
        )

    # Pad coords to the requested number of dims (n_components may be less
    # than dims when there are very few samples, e.g. 2 faces → max 1 PC).
    actual_cols = coords.shape[1] if coords.ndim == 2 else 1
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if actual_cols < dims:
        pad = np.zeros((coords.shape[0], dims - actual_cols), dtype=coords.dtype)
        coords = np.concatenate([coords, pad], axis=1)
    n_components = dims  # now always matches the requested dimensionality

    # ── 4. Resolve group colouring from session ───────────────────────────────
    group_map: dict[str, int] = {}   # face_id → group_id
    if session_id:
        try:
            sess = db.get_group_session(session_id)
            if sess:
                for grp in sess.get("groups", []):
                    for member in grp.get("members", []):
                        fid = member.get("face_id")
                        if fid:
                            group_map[fid] = grp["group_id"]
        except Exception:
            pass   # non-fatal — just no colour mapping

    # ── 5. Build response ─────────────────────────────────────────────────────
    points = []
    for i, fid in enumerate(ids):
        c = coords[i]
        points.append({
            "face_id":     fid,
            "x":           round(float(c[0]), 5),
            "y":           round(float(c[1]), 5),
            "z":           round(float(c[2]), 5) if n_components == 3 else None,
            "group_id":    group_map.get(fid),
            "source_file": sources[i],
            "quality":     quals[i],
        })

    return {
        "method":   method,
        "dims":     n_components,
        "n_points": len(points),
        "points":   points,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FACE COMPARE & SEARCH
# ══════════════════════════════════════════════════════════════════════════════

# Cosine-distance thresholds per model for "same person" verdict.
# Source: DeepFace internal thresholds, cosine metric.
_COSINE_THRESHOLDS: dict[str, float] = {
    "ArcFace":      0.68,
    "VGG-Face":     0.40,
    "Facenet512":   0.30,
    "Facenet":      0.40,
    "GhostFaceNet": 0.65,
    "DeepFace":     0.23,
    "DeepID":       0.015,
    "Dlib":         0.07,
    "SFace":        0.593,
    "OpenFace":     0.10,
}


def _cosine(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (similarity, distance) for two raw embedding vectors."""
    an = a / (np.linalg.norm(a) + 1e-10)
    bn = b / (np.linalg.norm(b) + 1e-10)
    sim  = float(np.dot(an, bn))
    dist = float(1.0 - sim)
    return round(sim, 6), round(dist, 6)


class CompareRequest(BaseModel):
    """Compare two faces. Supply face_ids (loaded from cache) or raw embeddings."""
    face_id_a:   Optional[str]        = Field(None, description="face_id from cache (option A)")
    face_id_b:   Optional[str]        = Field(None, description="face_id from cache (option A)")
    embedding_a: Optional[list[float]] = Field(None, description="Raw embedding vector (option B)")
    embedding_b: Optional[list[float]] = Field(None, description="Raw embedding vector (option B)")


@app.post("/faces/compare", tags=["Faces"], summary="Compare two faces – similarity score")
def compare_faces(req: CompareRequest):
    """
    Returns the **cosine similarity** and **distance** between two faces and a
    *verified* flag based on the server-configured recognition model threshold.

    Supply either two `face_id` values (looked up from cache) or two raw
    `embedding` vectors, or mix both.
    """
    def _resolve(face_id: Optional[str], emb: Optional[list[float]], label: str) -> np.ndarray:
        if emb is not None:
            return np.array(emb, dtype=np.float32)
        if face_id:
            try:
                face = db.get_face(face_id)
            except Exception as exc:
                raise err_db(exc)
            if face is None:
                raise err_face_not_found(face_id)
            return np.array(face["embedding"], dtype=np.float32)
        raise err_missing_param(label, "provide either face_id or embedding")

    vec_a = _resolve(req.face_id_a, req.embedding_a, "face_id_a / embedding_a")
    vec_b = _resolve(req.face_id_b, req.embedding_b, "face_id_b / embedding_b")

    if len(vec_a) != len(vec_b):
        raise err_dim_mismatch(len(vec_a), len(vec_b), "embedding_b")

    similarity, distance = _cosine(vec_a, vec_b)
    threshold = _COSINE_THRESHOLDS.get(MODEL_NAME, 0.40)
    return {
        "similarity":      similarity,
        "distance":        distance,
        "verified":        distance < threshold,
        "threshold":       threshold,
        "model":           MODEL_NAME,
        "face_id_a":       req.face_id_a,
        "face_id_b":       req.face_id_b,
    }


class SearchRequest(BaseModel):
    """Find the most similar faces in the cache for a given query."""
    face_id:         Optional[str]        = Field(None, description="Query by cached face_id")
    embedding:       Optional[list[float]] = Field(None, description="Query by raw embedding vector")
    top_k:           int   = Field(5,   ge=1, le=100,  description="Number of results to return")
    min_similarity:  float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity to include")
    exclude_self:    bool  = Field(True, description="Exclude the query face_id from results")


@app.post("/faces/search", tags=["Faces"], summary="Find similar faces in the cache")
def search_faces(req: SearchRequest):
    """
    Nearest-neighbour search across **all active cached embeddings**.

    Computes cosine similarity between the query vector and every cached face,
    returns the top-K closest matches above `min_similarity`.

    Supply either a `face_id` (embedding loaded from cache) or a raw `embedding`.
    """
    # ── Resolve query embedding ───────────────────────────────────────────────
    if req.embedding is not None:
        query_vec = np.array(req.embedding, dtype=np.float32)
        query_id  = None
    elif req.face_id:
        try:
            face = db.get_face(req.face_id)
        except Exception as exc:
            raise err_db(exc)
        if face is None:
            raise err_face_not_found(req.face_id)
        query_vec = np.array(face["embedding"], dtype=np.float32)
        query_id  = req.face_id
    else:
        raise err_missing_param("face_id / embedding", "provide one of them")

    # ── Load all cached embeddings ────────────────────────────────────────────
    try:
        candidates = db.get_all_embeddings(limit=10_000)
    except Exception as exc:
        raise err_db(exc)

    if not candidates:
        return {"query_face_id": query_id, "results": [], "searched": 0}

    # ── Batch cosine similarity (vectorised) ──────────────────────────────────
    matrix   = np.array([c["embedding"] for c in candidates], dtype=np.float32)
    norms    = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    mat_norm = matrix / norms
    q_norm   = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    sims     = mat_norm @ q_norm                                    # shape (N,)

    # ── Collect results ───────────────────────────────────────────────────────
    results = []
    for idx in np.argsort(sims)[::-1]:
        c   = candidates[idx]
        sim = float(sims[idx])
        if sim < req.min_similarity:
            break
        if req.exclude_self and c["face_id"] == query_id:
            continue
        results.append({
            "face_id":         c["face_id"],
            "similarity":      round(sim, 6),
            "distance":        round(1.0 - sim, 6),
            "source_file":     c["source_file"],
            "face_confidence": c["face_confidence"],
            "model_name":      c["model_name"],
        })
        if len(results) >= req.top_k:
            break

    threshold = _COSINE_THRESHOLDS.get(MODEL_NAME, 0.40)
    return {
        "query_face_id": query_id,
        "model":         MODEL_NAME,
        "threshold":     threshold,
        "searched":      len(candidates),
        "results":       results,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BARCODE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

BARCODE_FORMAT_MAP = {
    "CODE128":    "Code 128",
    "CODE39":     "Code 39",
    "CODE93":     "Code 93",
    "EAN13":      "EAN-13",
    "EAN8":       "EAN-8",
    "UPCA":       "UPC-A",
    "UPCE":       "UPC-E",
    "I25":        "Interleaved 2 of 5",
    "ITF":        "ITF / 2of5",
    "PDF417":     "PDF417",
    "QRCODE":     "QR Code",
    "DATAMATRIX": "Data Matrix",
    "CODABAR":    "Codabar",
    "AZTEC":      "Aztec Code",
}


def _parse_barcode(bc) -> dict:
    """
    Safely unpack a pyzbar Decoded object into a plain dict.

    pyzbar's Decoded namedtuple and the underlying libzbar differ across versions:
      - Standard 0.1.10:  data, type, rect(left,top,width,height),
                          polygon([Point(x,y)…]), quality, orientation
      - Slim/old zbar:    rect/polygon may be absent or zero-filled
      - 1-D barcodes:     polygon is an empty list; use rect instead

    We use getattr + duck-typing throughout so this function never raises.
    """
    # ── bounding rect ──────────────────────────────────────────────────────────
    rect = getattr(bc, "rect", None)
    left = top = width = height = 0
    if rect is not None:
        try:
            left, top, width, height = (
                int(rect.left), int(rect.top), int(rect.width), int(rect.height)
            )
        except AttributeError:
            try:
                left, top, width, height = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
            except Exception:
                pass

    # ── polygon ────────────────────────────────────────────────────────────────
    raw_poly = getattr(bc, "polygon", None)
    poly: list[dict] = []
    if raw_poly:
        try:
            poly = [{"x": int(p.x), "y": int(p.y)} for p in raw_poly]
        except AttributeError:
            try:
                poly = [{"x": int(p[0]), "y": int(p[1])} for p in raw_poly]
            except Exception:
                pass

    # If polygon is still empty but we have a valid rect, synthesise corners
    if not poly and width > 0:
        poly = [
            {"x": left,         "y": top},
            {"x": left + width, "y": top},
            {"x": left + width, "y": top + height},
            {"x": left,         "y": top + height},
        ]

    return {
        "data":               bc.data.decode("utf-8", errors="replace"),
        "symbology":          str(bc.type),
        "symbology_friendly": BARCODE_FORMAT_MAP.get(bc.type, str(bc.type)),
        "polygon":            poly,
        "bounding_rect":      {"left": left, "top": top, "width": width, "height": height},
        "quality":            getattr(bc, "quality", 0) or 0,
    }


def _build_preprocessed_images(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Generate multiple preprocessed RGB variants of the input image.

    Each variant targets a different class of degradation (poor contrast,
    uneven lighting, blur, small print, …).  The caller runs pyzbar on each
    variant and merges the results with position-aware deduplication so the
    same physical barcode is never counted twice.

    All images are returned as **RGB** uint8 arrays (pyzbar / PIL convention).
    """
    variants: list[np.ndarray] = []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── 1. Original colour image (best quality baseline) ─────────────────────
    variants.append(img_rgb)

    # ── 2. Global histogram equalisation → RGB ───────────────────────────────
    eq = cv2.equalizeHist(gray)
    variants.append(cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB))

    # ── 3. CLAHE on luminance (L channel in LAB) ────────────────────────────
    #    Adaptive local contrast enhancement – much better than global
    #    equalisation for unevenly lit images (shadows, glare, reflections).
    try:
        lab        = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b    = cv2.split(lab)
        clahe      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab        = cv2.merge([clahe.apply(l), a, b])
        clahe_bgr  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        variants.append(cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB))
    except Exception:
        pass

    # ── 4. Adaptive Gaussian threshold ───────────────────────────────────────
    #    Handles uneven lighting where a global threshold fails entirely.
    try:
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, blockSize=51, C=11,
        )
        variants.append(cv2.cvtColor(adapt, cv2.COLOR_GRAY2RGB))
    except Exception:
        pass

    # ── 5. Otsu binarisation ─────────────────────────────────────────────────
    #    Automatic optimal threshold – works well for clean bimodal images
    #    (printed barcodes on plain background).
    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB))
    except Exception:
        pass

    # ── 6. Sharpened image (unsharp mask) ────────────────────────────────────
    #    Recovers slightly blurred / out-of-focus barcodes.
    try:
        blurred  = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=3)
        sharp    = cv2.addWeighted(img_bgr, 1.5, blurred, -0.5, 0)
        variants.append(cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB))
    except Exception:
        pass

    # ── 7. Morphological close then Otsu ─────────────────────────────────────
    #    Fills small gaps / noise in damaged barcodes before binarising.
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        _, closed_otsu = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(closed_otsu, cv2.COLOR_GRAY2RGB))
    except Exception:
        pass

    # ── 8. 2× upscale (for small / distant barcodes) ────────────────────────
    #    Only when the image is small enough that 2× is still reasonable.
    h, w = img_bgr.shape[:2]
    if max(h, w) <= 1200:
        try:
            up = cv2.resize(img_bgr, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            variants.append(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
        except Exception:
            pass

    return variants


def _dedup_key_from_rect(rect, scale: float = 1.0) -> tuple[int, int]:
    """Return quantised (cx, cy) from a pyzbar rect for deduplication."""
    cx = cy = 0
    if rect is not None:
        try:
            cx = int(rect.left) + int(rect.width) // 2
            cy = int(rect.top) + int(rect.height) // 2
        except AttributeError:
            try:
                cx = int(rect[0]) + int(rect[2]) // 2
                cy = int(rect[1]) + int(rect[3]) // 2
            except Exception:
                pass
    # Undo any upscale so coordinates match the original image grid
    cx = int(cx / scale)
    cy = int(cy / scale)
    return cx // 50, cy // 50


def _cv2_qr_detect(img_gray: np.ndarray) -> list[dict]:
    """
    Use OpenCV's built-in QR detector as a **full decoder**.

    Returns a list of parsed barcode dicts (same schema as _parse_barcode)
    for every QR code that cv2 can decode.  This catches QR codes that
    pyzbar / libzbar misses (different Reed-Solomon error correction
    implementation, different binarisation).
    """
    results: list[dict] = []
    try:
        detector = cv2.QRCodeDetector()
        ok, texts, pts, _ = detector.detectAndDecodeMulti(img_gray)
        if not ok or pts is None:
            return results
        for text, corners in zip(texts, pts):
            if not text or corners is None:
                continue
            poly = [
                {"x": int(corners[i][0]), "y": int(corners[i][1])}
                for i in range(len(corners))
            ]
            xs = [p["x"] for p in poly]
            ys = [p["y"] for p in poly]
            results.append({
                "data":               text,
                "symbology":          "QRCODE",
                "symbology_friendly": "QR Code",
                "polygon":            poly,
                "bounding_rect": {
                    "left":   min(xs),
                    "top":    min(ys),
                    "width":  max(xs) - min(xs),
                    "height": max(ys) - min(ys),
                },
                "quality":            1,
            })
    except Exception:
        pass
    return results


def _barcode_center(b: dict) -> tuple[int, int]:
    """Return (cx, cy) of a parsed barcode dict."""
    r = b["bounding_rect"]
    return r["left"] + r["width"] // 2, r["top"] + r["height"] // 2


def _barcode_overlaps_any(b: dict, existing: list[dict], threshold: int = 40) -> bool:
    """True if barcode *b* is within *threshold* px of any already-found one."""
    cx, cy = _barcode_center(b)
    for e in existing:
        ex, ey = _barcode_center(e)
        if abs(cx - ex) < threshold and abs(cy - ey) < threshold:
            return True
    return False


def _pyzbar_scan(img_rgb, scale: float, h_orig: int,
                 found: list[dict], seen: set) -> None:
    """Run pyzbar on a single image, dedup against *found*/*seen*, append new."""
    from pyzbar import pyzbar as _pyzbar

    for bc in _pyzbar.decode(Image.fromarray(img_rgb)):
        rect = getattr(bc, "rect", None)
        qx, qy = _dedup_key_from_rect(rect, scale)
        key = (bc.data, bc.type, qx, qy)
        if key in seen:
            continue
        seen.add(key)
        try:
            parsed = _parse_barcode(bc)
            if abs(scale - 1.0) > 0.01:
                inv = 1.0 / scale
                parsed["polygon"] = [
                    {"x": int(p["x"] * inv), "y": int(p["y"] * inv)}
                    for p in parsed["polygon"]
                ]
                r = parsed["bounding_rect"]
                parsed["bounding_rect"] = {
                    "left":   int(r["left"]   * inv),
                    "top":    int(r["top"]    * inv),
                    "width":  int(r["width"]  * inv),
                    "height": int(r["height"] * inv),
                }
            found.append(parsed)
        except Exception as exc:
            log.warning(
                "Could not parse barcode object (type=%s fields=%s): %s",
                type(bc).__name__,
                getattr(bc, "_fields", "?"),
                exc,
            )


def _cv2_dedup_and_add(cv2_results: list[dict], found: list[dict], seen: set,
                       offset_x: int = 0, offset_y: int = 0) -> None:
    """Add cv2 QR results to *found*, deduplicating against existing."""
    for cv2_bc in cv2_results:
        if offset_x or offset_y:
            cv2_bc["polygon"] = [
                {"x": p["x"] + offset_x, "y": p["y"] + offset_y}
                for p in cv2_bc["polygon"]
            ]
            r = cv2_bc["bounding_rect"]
            cv2_bc["bounding_rect"] = {
                "left": r["left"] + offset_x, "top": r["top"] + offset_y,
                "width": r["width"], "height": r["height"],
            }
        r = cv2_bc["bounding_rect"]
        cx = r["left"] + r["width"] // 2
        cy = r["top"]  + r["height"] // 2
        key = (cv2_bc["data"].encode("utf-8", errors="replace"), "QRCODE", cx // 50, cy // 50)
        if key in seen:
            continue
        seen.add(key)
        found.append(cv2_bc)


def _run_barcode_detection(img_bgr: np.ndarray) -> list[dict]:
    """
    Detect and decode **all** barcodes in a BGR image.

    Runs a comprehensive multi-pass strategy to maximise recall even on
    poorly printed, damaged, low-contrast, blurred, or very small barcodes:

      1. **pyzbar** on 8+ preprocessed image variants (original colour,
         histogram-equalised, CLAHE, adaptive threshold, Otsu, sharpened,
         morphological close + Otsu, 2× upscale).
      2. **OpenCV QRCodeDetector** (``detectAndDecodeMulti``) as a full
         secondary decoder – its Reed-Solomon implementation catches QR
         codes that libzbar cannot reconstruct.
      3. **Region-based re-scan**: after full-image passes, previously
         decoded barcode regions are masked out and the remainder is
         scanned again.  This defeats libzbar's tendency to return only
         *one* result when multiple identical barcodes are present.
      4. Position-aware deduplication ensures each physical barcode is
         reported exactly once even when multiple passes decode it.
    """
    from pyzbar import pyzbar

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_orig, w_orig = img_bgr.shape[:2]

    # ── Generate all preprocessed variants ─────────────────────────────────────
    variants = _build_preprocessed_images(img_bgr)

    found: list[dict] = []
    seen:  set        = set()

    # ── Pass A: pyzbar on every variant ────────────────────────────────────────
    for src in variants:
        src_h = src.shape[0]
        scale = src_h / h_orig if h_orig > 0 else 1.0
        _pyzbar_scan(src, scale, h_orig, found, seen)

    # ── Pass B: OpenCV QR decoder as secondary engine ──────────────────────────
    _cv2_dedup_and_add(_cv2_qr_detect(gray), found, seen)
    for extra_gray in (cv2.equalizeHist(gray),):
        _cv2_dedup_and_add(_cv2_qr_detect(extra_gray), found, seen)

    # ── Pass C: Region-mask re-scan ────────────────────────────────────────────
    #   pyzbar/libzbar often returns only ONE barcode per image even when
    #   multiple are present (especially with identical data).  We mask out
    #   every already-found barcode with white, then re-scan to pick up the
    #   rest.  Repeat until no new barcodes appear (max 10 rounds).
    if found:
        mask_img_bgr = img_bgr.copy()
        prev_count = 0
        for _round in range(10):
            if len(found) == prev_count and _round > 0:
                break
            prev_count = len(found)
            # White-out every known barcode region (with generous padding)
            for b in found:
                r = b["bounding_rect"]
                pad = max(r["width"], r["height"]) // 4 + 10
                x1 = max(0, r["left"] - pad)
                y1 = max(0, r["top"]  - pad)
                x2 = min(w_orig, r["left"] + r["width"]  + pad)
                y2 = min(h_orig, r["top"]  + r["height"] + pad)
                mask_img_bgr[y1:y2, x1:x2] = 255

            # pyzbar on masked image (original + CLAHE)
            masked_rgb = cv2.cvtColor(mask_img_bgr, cv2.COLOR_BGR2RGB)
            _pyzbar_scan(masked_rgb, 1.0, h_orig, found, seen)

            try:
                lab = cv2.cvtColor(mask_img_bgr, cv2.COLOR_BGR2LAB)
                l, a, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab = cv2.merge([clahe.apply(l), a, b_ch])
                clahe_rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
                _pyzbar_scan(clahe_rgb, 1.0, h_orig, found, seen)
            except Exception:
                pass

            # cv2 QR on masked grayscale
            masked_gray = cv2.cvtColor(mask_img_bgr, cv2.COLOR_BGR2GRAY)
            _cv2_dedup_and_add(_cv2_qr_detect(masked_gray), found, seen)

    # ── Pass D: Tile-based scan for large images ───────────────────────────────
    #   Scan overlapping quadrants so that barcodes near the centre that span
    #   a tile boundary still get a chance in at least one tile.
    if max(h_orig, w_orig) >= 600:
        tile_specs = [
            (0, 0, w_orig // 2 + w_orig // 8, h_orig // 2 + h_orig // 8),
            (w_orig // 2 - w_orig // 8, 0, w_orig, h_orig // 2 + h_orig // 8),
            (0, h_orig // 2 - h_orig // 8, w_orig // 2 + w_orig // 8, h_orig),
            (w_orig // 2 - w_orig // 8, h_orig // 2 - h_orig // 8, w_orig, h_orig),
        ]
        for tx1, ty1, tx2, ty2 in tile_specs:
            tile = img_bgr[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue
            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            # pyzbar on tile
            for bc in pyzbar.decode(Image.fromarray(tile_rgb)):
                rect = getattr(bc, "rect", None)
                # Map tile coords back to full image
                abs_rect = None
                if rect is not None:
                    try:
                        abs_rect = type(rect)(
                            rect.left + tx1, rect.top + ty1, rect.width, rect.height)
                    except Exception:
                        abs_rect = (rect[0] + tx1, rect[1] + ty1, rect[2], rect[3])
                qx, qy = _dedup_key_from_rect(abs_rect, 1.0)
                key = (bc.data, bc.type, qx, qy)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    parsed = _parse_barcode(bc)
                    # Offset polygon & rect to full-image coordinates
                    parsed["polygon"] = [
                        {"x": p["x"] + tx1, "y": p["y"] + ty1}
                        for p in parsed["polygon"]
                    ]
                    r = parsed["bounding_rect"]
                    parsed["bounding_rect"] = {
                        "left": r["left"] + tx1, "top": r["top"] + ty1,
                        "width": r["width"], "height": r["height"],
                    }
                    # Only add if not overlapping an existing barcode
                    if not _barcode_overlaps_any(parsed, found):
                        found.append(parsed)
                except Exception:
                    pass

            # cv2 QR on tile grayscale
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            _cv2_dedup_and_add(
                _cv2_qr_detect(tile_gray), found, seen,
                offset_x=tx1, offset_y=ty1,
            )

    # ── Patch zero-polygon QR codes ────────────────────────────────────────────
    needs_patch = [
        b for b in found
        if b["symbology"] == "QRCODE" and b["bounding_rect"]["width"] == 0
    ]
    if needs_patch:
        cv2_results = _cv2_qr_detect(gray)
        pos_map: dict[str, list[dict]] = {}
        for cr in cv2_results:
            pos_map.setdefault(cr["data"], []).append(cr)
        for b in needs_patch:
            entries = pos_map.get(b["data"])
            if entries:
                donor = entries.pop(0)
                b["polygon"]       = donor["polygon"]
                b["bounding_rect"] = donor["bounding_rect"]

    return found


@app.post(
    "/barcodes/detect",
    tags=["Barcodes"],
    summary="Detect barcodes in an image",
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "examples": {
                            "two_barcodes": {
                                "summary": "EAN-13 + QR Code erkannt",
                                "value": {
                                    "source_file": "produkt_regal.jpg",
                                    "barcodes_found": 2,
                                    "barcodes": [
                                        {
                                            "data": "9783161484100",
                                            "symbology": "EAN13",
                                            "symbology_friendly": "EAN-13",
                                            "polygon": [{"x": 142, "y": 380}, {"x": 312, "y": 380}, {"x": 312, "y": 420}, {"x": 142, "y": 420}],
                                            "bounding_rect": {"left": 142, "top": 380, "width": 170, "height": 40},
                                            "quality": 1
                                        },
                                        {
                                            "data": "https://example.com/produkt/12345",
                                            "symbology": "QRCODE",
                                            "symbology_friendly": "QR Code",
                                            "polygon": [{"x": 500, "y": 200}, {"x": 650, "y": 200}, {"x": 650, "y": 350}, {"x": 500, "y": 350}],
                                            "bounding_rect": {"left": 500, "top": 200, "width": 150, "height": 150},
                                            "quality": 1
                                        }
                                    ]
                                }
                            },
                            "none_found": {
                                "summary": "Kein Barcode im Bild",
                                "value": {
                                    "source_file": "portrait.jpg",
                                    "barcodes_found": 0,
                                    "barcodes": []
                                }
                            },
                            "i2of5": {
                                "summary": "Interleaved 2 of 5 (Lagerlogistik)",
                                "value": {
                                    "source_file": "lager_etikett.jpg",
                                    "barcodes_found": 1,
                                    "barcodes": [
                                        {
                                            "data": "0012345678905",
                                            "symbology": "I25",
                                            "symbology_friendly": "Interleaved 2 of 5",
                                            "polygon": [{"x": 20, "y": 60}, {"x": 280, "y": 60}, {"x": 280, "y": 95}, {"x": 20, "y": 95}],
                                            "bounding_rect": {"left": 20, "top": 60, "width": 260, "height": 35},
                                            "quality": 1
                                        }
                                    ]
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def detect_barcodes(
    file: UploadFile = File(..., description="Image containing 0..n barcodes"),
):
    """
    Scans the image for **any number of barcodes** (including zero).

    Uses a **multi-pass detection pipeline** to maximise recall:

    1. **8 preprocessing variants** — original colour, histogram-equalised,
       CLAHE, adaptive Gaussian threshold, Otsu binarisation, sharpened
       (unsharp mask), morphological close + Otsu, 2× upscale (small images).
    2. **pyzbar (libzbar)** runs on every variant — best all-round decoder.
    3. **OpenCV QRCodeDetector** as a secondary decoder — different ECC and
       binarisation catches QR codes that libzbar cannot reconstruct.
    4. **Position-aware deduplication** ensures each physical barcode is
       reported exactly once even when multiple passes decode it.

    This pipeline is specifically designed to handle:
    - Low contrast / poor lighting / shadows / glare
    - Blurry or slightly out-of-focus images
    - Small or distant barcodes
    - Damaged or partially obscured codes
    - Multiple identical barcodes at different positions

    ### Supported symbologies
    QR Code · Code 128 · Code 39 · Code 93 · EAN-13 · EAN-8 ·
    UPC-A · UPC-E · Interleaved 2 of 5 · ITF · PDF417 · Data Matrix ·
    Codabar · Aztec Code

    ### Returns (per barcode)
    | Field | Description |
    |---|---|
    | `data` | Decoded text value |
    | `symbology` | Raw format string |
    | `symbology_friendly` | Human-readable format name |
    | `polygon` | Precise corner coordinates |
    | `bounding_rect` | Axis-aligned bounding box |
    | `quality` | Decoder confidence score |
    """
    data = await file.read()
    _validate_size(data)

    try:
        img      = _decode_image(data)
        barcodes = _run_barcode_detection(img)
    except VisionAPIException:
        raise
    except Exception as exc:
        log.exception("Barcode detection error")
        raise VisionAPIException(500, ErrorCode.INFERENCE_FAILED,
            "Barcode detection failed.", detail=str(exc))

    return {
        "source_file":    file.filename,
        "barcodes_found": len(barcodes),
        "barcodes":       barcodes,
    }


@app.post("/barcodes/detect-base64", tags=["Barcodes"],
          summary="Detect barcodes from a base64-encoded image")
async def detect_barcodes_b64(body: dict):
    """
    Same as `POST /barcodes/detect` but accepts JSON:

    ```json
    { "image_b64": "/9j/4AAQSkZJRgAB..." }
    ```
    """
    raw_b64 = body.get("image_b64", "")
    if not raw_b64:
        raise err_missing_param("image_b64", "must be a non-empty base64 string")
    try:
        raw = base64.b64decode(raw_b64)
    except Exception:
        raise err_invalid_b64()

    _validate_size(raw)
    try:
        img      = _decode_image(raw)
        barcodes = _run_barcode_detection(img)
    except VisionAPIException:
        raise
    except Exception as exc:
        log.exception("Barcode detection error (b64)")
        raise VisionAPIException(500, ErrorCode.INFERENCE_FAILED,
            "Barcode detection failed.", detail=str(exc))

    return {"barcodes_found": len(barcodes), "barcodes": barcodes}


# ══════════════════════════════════════════════════════════════════════════════
#  ASYNC JOB QUEUE – Submit & inspect
# ══════════════════════════════════════════════════════════════════════════════

class CallbackConfig(BaseModel):
    """
    Webhook configuration for async job results.
    All fields are optional — missing fields fall back to server-side defaults
    from environment variables (DEFAULT_CALLBACK_URL, DEFAULT_CALLBACK_AUTH_*).
    """
    url:       str           = Field(...,  description="HTTP(S) endpoint that receives the result")
    auth_user: Optional[str] = Field(None, description="HTTP Basic Auth username")
    auth_pass: Optional[str] = Field(None, description="HTTP Basic Auth password")


class AsyncEmbedRequest(BaseModel):
    """
    Request body for async face embedding.
    The image must be provided as a base64-encoded string.
    """
    image_b64:  str                      = Field(..., description="Base64-encoded image")
    source_file: Optional[str]           = Field(None, description="Original filename for reference")
    cache:       bool                    = Field(True,  description="Cache embeddings in face-ID store")
    ttl:         Optional[int]           = Field(None,  ge=60, description="Cache TTL override (seconds)")
    callback:    Optional[CallbackConfig] = Field(None,
        description="Webhook to receive the result. Falls back to DEFAULT_CALLBACK_URL env var.")


class AsyncClusterRequest(BaseModel):
    """Request body for async face clustering (DBSCAN within one image)."""
    image_b64:      str                      = Field(...)
    source_file:    Optional[str]            = Field(None)
    min_similarity: float                    = Field(0.60, ge=0.0, le=1.0)
    cache:          bool                     = Field(True)
    ttl:            Optional[int]            = Field(None, ge=60)
    callback:       Optional[CallbackConfig] = Field(None)


def _resolve_callback(cb: Optional[CallbackConfig]) -> tuple[str, str, str]:
    """Return (url, user, pass) merging request-level override with env defaults."""
    url  = (cb.url       if cb and cb.url       else qw.DEFAULT_CALLBACK_URL)  or ""
    user = (cb.auth_user if cb and cb.auth_user else qw.DEFAULT_CALLBACK_USER) or ""
    pw   = (cb.auth_pass if cb and cb.auth_pass else qw.DEFAULT_CALLBACK_PASS) or ""
    return url, user, pw


def _b64_to_tmp(image_b64: str) -> Path:
    """Decode base64 image → temp file. Raises on bad b64 or invalid image."""
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        raise err_invalid_b64()
    _validate_size(raw)
    # quick sanity-check — will raise err_image_decode if not a valid image
    _decode_image(raw)
    p = _save_tmp(raw)
    return p


# ── POST /faces/embed/async ────────────────────────────────────────────────────

@app.post(
    "/faces/embed/async",
    tags=["Async Jobs"],
    status_code=202,
    summary="Queue a face-embedding job (non-blocking)",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "with_callback_auth": {
                            "summary": "Mit Webhook + Basic Auth",
                            "value": {
                                "image_b64": "/9j/4AAQSkZJRgAB...(base64)",
                                "source_file": "gruppe.jpg",
                                "cache": True,
                                "ttl": 86400,
                                "callback": {
                                    "url": "https://meinserver.de/api/webhooks/vision",
                                    "auth_user": "webhook-user",
                                    "auth_pass": "geheimespasswort"
                                }
                            }
                        },
                        "polling_only": {
                            "summary": "Kein Callback – nur Polling via GET /jobs/{job_id}",
                            "value": {
                                "image_b64": "/9j/4AAQSkZJRgAB...(base64)",
                                "source_file": "portrait.jpg",
                                "cache": True
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "202": {
                "description": "Job erfolgreich eingereiht",
                "content": {
                    "application/json": {
                        "example": {
                            "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
                            "status": "PENDING",
                            "message": "Job queued. Poll GET /jobs/{job_id} or await the callback.",
                            "callback_url": "https://meinserver.de/api/webhooks/vision",
                            "auth_enabled": True,
                            "queue_depth": 2
                        }
                    }
                }
            },
            "503": {
                "description": "Queue ist voll",
                "content": {
                    "application/json": {
                        "example": {
                            "error": True,
                            "code": "QUEUE_FULL",
                            "message": "The job queue is full (50 items). Try again later.",
                            "detail": None
                        }
                    }
                }
            }
        }
    }
)
async def embed_faces_async(req: AsyncEmbedRequest):
    """
    Accepts an image, **immediately returns a `job_id`** (HTTP 202), then
    processes the embedding in the background.

    When finished the result is POSTed to `callback.url` (or the server-wide
    `DEFAULT_CALLBACK_URL`).

    ### Callback payload
    ```json
    {
      "job_id": "...",
      "status": "DONE",
      "job_type": "embed",
      "submitted_at": 1712345678.0,
      "completed_at": 1712345690.1,
      "duration_seconds": 12.1,
      "result": { ... },   // same shape as POST /faces/embed
      "error": null
    }
    ```

    ### Callback authentication
    Set `callback.auth_user` / `callback.auth_pass` for HTTP Basic Auth.
    Credentials are **never stored** — only kept in memory for the job lifetime.

    ### Polling
    Use `GET /jobs/{job_id}` to poll status instead of (or in addition to)
    the callback webhook.
    """
    if qw.JobQueue.get().is_full():
        raise VisionAPIException(503, ErrorCode.QUEUE_FULL,
            f"The job queue is full ({qw.QUEUE_MAX_SIZE} items). Try again later.")

    cb_url, cb_user, cb_pass = _resolve_callback(req.callback)

    effective_ttl = req.ttl or FACE_TTL_SEC

    # Capture values needed inside the coroutine closure
    image_b64   = req.image_b64
    source_file = req.source_file or "async_upload"
    cache       = req.cache

    async def _work():
        tmp = _b64_to_tmp(image_b64)
        try:
            raw = await _represent(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)

        if not raw:
            raise VisionAPIException(422, ErrorCode.NO_FACE_DETECTED,
                "No faces detected in the submitted image.")

        results = []
        for idx, item in enumerate(raw):
            embedding   = item["embedding"]
            facial_area = item.get("facial_area", {})
            confidence  = item.get("face_confidence")
            face_id = None
            expires_at = None
            if cache:
                try:
                    face_id = db.cache_face(
                        embedding=embedding, model_name=MODEL_NAME,
                        ttl_seconds=effective_ttl, facial_area=facial_area,
                        face_confidence=confidence, source_file=source_file,
                        face_index=idx,
                    )
                    expires_at = time.time() + effective_ttl
                except Exception as e:
                    log.warning("Cache write failed in async job: %s", e)
            results.append({
                "face_id": face_id, "face_index": idx,
                "face_confidence": confidence, "facial_area": facial_area,
                "embedding": embedding, "embedding_dim": len(embedding),
                "cached": cache and face_id is not None,
                "expires_at": expires_at,
            })
        return {
            "source_file": source_file,
            "faces_found": len(results),
            "model": MODEL_NAME,
            "faces": results,
        }

    job_id = str(uuid.uuid4())
    job = qw.Job(
        job_id=job_id, job_type="embed",
        coro_factory=_work,
        callback_url=cb_url, callback_user=cb_user, callback_pass=cb_pass,
    )
    accepted = await qw.JobQueue.get().submit(job)
    if not accepted:
        raise VisionAPIException(503, ErrorCode.QUEUE_FULL,
            f"The job queue is full ({qw.QUEUE_MAX_SIZE} items). Try again later.")

    return {
        "job_id":       job_id,
        "status":       "PENDING",
        "message":      "Job queued. Poll GET /jobs/{job_id} or await the callback.",
        "callback_url": cb_url or None,
        "auth_enabled": bool(cb_user),
        "queue_depth":  qw.JobQueue.get().pending_count,
    }


# ── POST /faces/cluster/async ──────────────────────────────────────────────────

@app.post(
    "/faces/cluster/async",
    tags=["Async Jobs"],
    status_code=202,
    summary="Queue a face-clustering job (non-blocking)",
)
async def cluster_faces_async(req: AsyncClusterRequest):
    """
    Same as `POST /faces/cluster` but non-blocking.
    Returns `job_id` immediately (HTTP 202); result is POSTed to the callback URL.

    See `POST /faces/embed/async` for full callback documentation.
    """
    if qw.JobQueue.get().is_full():
        raise VisionAPIException(503, ErrorCode.QUEUE_FULL,
            f"The job queue is full ({qw.QUEUE_MAX_SIZE} items). Try again later.")

    cb_url, cb_user, cb_pass = _resolve_callback(req.callback)

    image_b64      = req.image_b64
    source_file    = req.source_file or "async_upload"
    min_similarity = req.min_similarity
    cache          = req.cache
    effective_ttl  = req.ttl or FACE_TTL_SEC

    async def _work():
        tmp = _b64_to_tmp(image_b64)
        try:
            raw = await _represent(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)

        if not raw:
            raise VisionAPIException(422, ErrorCode.NO_FACE_DETECTED,
                "No faces detected in the submitted image.")

        vectors_norm = cl.normalize_vectors(
            np.array([e["embedding"] for e in raw])
        )
        labels = cl.run_dbscan(vectors_norm, min_similarity)

        faces = []
        for i, (item, label) in enumerate(zip(raw, labels)):
            face_id = None
            if cache:
                try:
                    face_id = db.cache_face(
                        embedding=item["embedding"], model_name=MODEL_NAME,
                        ttl_seconds=effective_ttl,
                        facial_area=item.get("facial_area"),
                        face_confidence=item.get("face_confidence"),
                        source_file=source_file, face_index=i,
                    )
                except Exception as e:
                    log.warning("Cache write failed in async cluster job: %s", e)
            faces.append({
                "face_id": face_id, "face_index": i, "cluster_id": label,
                "face_confidence": item.get("face_confidence"),
                "facial_area": item.get("facial_area", {}),
                "cached": cache and face_id is not None,
            })

        return {
            "source_file":    source_file,
            "faces_found":    len(faces),
            "n_clusters":     len({l for l in labels if l >= 0}),
            "n_noise":        sum(1 for l in labels if l < 0),
            "model":          MODEL_NAME,
            "min_similarity": min_similarity,
            "faces":          faces,
        }

    job_id = str(uuid.uuid4())
    job = qw.Job(
        job_id=job_id, job_type="cluster",
        coro_factory=_work,
        callback_url=cb_url, callback_user=cb_user, callback_pass=cb_pass,
    )
    accepted = await qw.JobQueue.get().submit(job)
    if not accepted:
        raise VisionAPIException(503, ErrorCode.QUEUE_FULL,
            f"The job queue is full ({qw.QUEUE_MAX_SIZE} items). Try again later.")

    return {
        "job_id":       job_id,
        "status":       "PENDING",
        "message":      "Job queued. Poll GET /jobs/{job_id} or await the callback.",
        "callback_url": cb_url or None,
        "auth_enabled": bool(cb_user),
        "queue_depth":  qw.JobQueue.get().pending_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  JOB STATUS / MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/jobs", tags=["Async Jobs"], summary="List all jobs")
def list_jobs(
    limit:  Annotated[int, Query(ge=1, le=500)] = 100,
    status: Optional[str] = Query(None,
        description="Filter by status: PENDING | RUNNING | DONE | FAILED | TIMEOUT"),
):
    """
    Returns jobs from in-memory store and the persistent DB (for jobs that
    survived a restart).

    In-memory jobs always take precedence over DB records.
    """
    # Merge in-memory + DB, de-duplicate by job_id
    mem_jobs  = {j["job_id"]: j for j in qw.list_jobs(limit=limit, status_filter=status)}
    try:
        db_jobs = {r["job_id"]: r for r in db.list_jobs_db(limit=limit, status_filter=status)
                   if r["job_id"] not in mem_jobs}
    except Exception:
        db_jobs = {}

    all_jobs = sorted(
        list(mem_jobs.values()) + list(db_jobs.values()),
        key=lambda j: j.get("submitted_at", 0),
        reverse=True,
    )
    return {
        "count":     len(all_jobs[:limit]),
        "queue":     qw.queue_stats(),
        "jobs":      all_jobs[:limit],
    }


@app.get(
    "/jobs/{job_id}",
    tags=["Async Jobs"],
    summary="Get job status & result",
    openapi_extra={
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "examples": {
                            "done": {
                                "summary": "DONE – Ergebnis verfügbar",
                                "value": {
                                    "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
                                    "job_type": "embed",
                                    "status": "DONE",
                                    "callback_url": "https://meinserver.de/webhook",
                                    "submitted_at": 1712432000.0,
                                    "started_at": 1712432003.5,
                                    "completed_at": 1712432015.8,
                                    "duration_seconds": 15.8,
                                    "error": None,
                                    "result": {
                                        "source_file": "gruppe.jpg",
                                        "faces_found": 2,
                                        "model": "VGG-Face",
                                        "faces": [
                                            {"face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d", "face_index": 0, "face_confidence": 0.9821, "facial_area": {"x": 124, "y": 89, "w": 156, "h": 178}, "embedding": [0.0312, -0.1547, "..."], "embedding_dim": 4096, "cached": True, "expires_at": 1712518400.0}
                                        ]
                                    }
                                }
                            },
                            "pending": {
                                "summary": "PENDING – wartet in Queue",
                                "value": {
                                    "job_id": "b2c3d4e5-f6a7-8901-bcde-f01234567890",
                                    "job_type": "embed",
                                    "status": "PENDING",
                                    "callback_url": None,
                                    "submitted_at": 1712432025.0,
                                    "started_at": None,
                                    "completed_at": None,
                                    "duration_seconds": None,
                                    "error": None,
                                    "result": None
                                }
                            },
                            "failed": {
                                "summary": "FAILED – Fehler während Verarbeitung",
                                "value": {
                                    "job_id": "c3d4e5f6-a7b8-9012-cdef-012345678901",
                                    "job_type": "cluster",
                                    "status": "FAILED",
                                    "callback_url": "https://meinserver.de/webhook",
                                    "submitted_at": 1712432100.0,
                                    "started_at": 1712432101.2,
                                    "completed_at": 1712432101.8,
                                    "duration_seconds": 0.6,
                                    "error": "VisionAPIException: No faces could be detected in the image.",
                                    "result": None
                                }
                            },
                            "timeout": {
                                "summary": "TIMEOUT – Verarbeitung zu langsam",
                                "value": {
                                    "job_id": "d4e5f6a7-b8c9-0123-defa-123456789012",
                                    "job_type": "embed",
                                    "status": "TIMEOUT",
                                    "callback_url": None,
                                    "submitted_at": 1712432200.0,
                                    "started_at": 1712432201.0,
                                    "completed_at": 1712432321.0,
                                    "duration_seconds": 120.0,
                                    "error": "Job exceeded timeout of 120s",
                                    "result": None
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
def get_job(job_id: str):
    """
    Returns current status of a job.

    - **PENDING** — waiting in queue
    - **RUNNING** — currently processing
    - **DONE** — result available in `result` field
    - **FAILED** — error message in `error` field
    - **TIMEOUT** — exceeded `QUEUE_JOB_TIMEOUT_SECONDS`

    The `result` field has the same structure as the synchronous endpoint.
    """
    job = qw.get_job(job_id)
    if job is None:
        raise VisionAPIException(404, ErrorCode.JOB_NOT_FOUND,
            f"Job '{job_id}' not found. It may have expired or never existed.",
            detail={"job_id": job_id})
    return job.to_dict(include_result=True)


@app.delete("/jobs/{job_id}", tags=["Async Jobs"], summary="Delete a job record")
def delete_job(job_id: str):
    """
    Removes a finished job from memory and the DB.
    Running or pending jobs cannot be deleted.
    """
    job = qw.get_job(job_id)
    if job is None:
        raise VisionAPIException(404, ErrorCode.JOB_NOT_FOUND,
            f"Job '{job_id}' not found.", detail={"job_id": job_id})

    from queue_worker import JobStatus
    if job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        raise VisionAPIException(409, ErrorCode.JOB_ACTIVE,
            "Cannot delete a PENDING or RUNNING job.",
            detail={"job_id": job_id, "status": str(job.status)})

    # Remove from memory
    qw._jobs.pop(job_id, None)
    # Remove from DB
    try:
        with db._conn() as con:
            con.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
    except Exception as exc:
        log.warning("Could not delete job from DB: %s", exc)

    return {"deleted": True, "job_id": job_id}
