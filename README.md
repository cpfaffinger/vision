# Vision API

> Self-contained Docker service for **face recognition**, **face grouping**, and **barcode detection** — powered by [DeepFace](https://github.com/serengil/deepface) and served through a FastAPI REST interface.

[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What it does

| Feature | Description |
|---|---|
| 🔍 **Face Embedding** | Detect faces in a photo and compute deep feature vectors (embeddings) |
| 👥 **Face Grouping** | Group a collection of face IDs by person identity across many photos |
| 🔎 **Face Search** | Find the most similar faces in the cache by vector nearest-neighbour |
| ⚖️ **Face Compare** | Compute similarity score between any two faces |
| 📦 **Barcode Detection** | Detect and decode QR codes, EAN-13, Code 128, PDF417, and more |
| 🗄️ **Face Cache** | TTL-based SQLite store — every detected face gets a persistent `face_id` |
| ⚡ **Async Jobs** | Non-blocking embed and cluster jobs with webhook callbacks |
| 🖥️ **Admin UI** | Built-in web dashboard for uploads, cache inspection, and debugging |

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration Profiles](#configuration-profiles)
- [API Overview](#api-overview)
- [Typical Workflows](#typical-workflows)
- [cURL Examples](#curl-examples)
- [Admin Dashboard](#admin-dashboard)
- [Architecture](#architecture)
- [Deployment Profiles](#deployment-profiles)
- [FAQ](#faq)

---

## Quick Start

```bash
git clone https://github.com/your-org/vision-api.git
cd vision-api

docker compose up --build
```

That's it. The API is live at:

| URL | Description |
|---|---|
| `http://localhost:8000` | REST API |
| `http://localhost:8000/docs` | Interactive Swagger UI |
| `http://localhost:8000/redoc` | ReDoc reference |
| `http://localhost:8000/admin` | Admin dashboard |

> **First start takes 2–5 minutes.** DeepFace downloads model weights (~300–500 MB) at build time and caches them in a Docker volume. Subsequent starts are instant.

---

## Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| [Docker](https://docs.docker.com/get-docker/) | 24.x | Desktop or Engine |
| [Docker Compose](https://docs.docker.com/compose/install/) | v2.x | Bundled with Docker Desktop |
| RAM | 2 GB | 4 GB recommended for ArcFace + retinaface |
| Disk | 2 GB free | For model weights and Docker image |

No Python, no local dependencies — everything runs inside the container.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/vision-api.git
cd vision-api
```

### 2. Choose a configuration profile

Copy one of the pre-configured env files to `.env`:

```bash
# Balanced (recommended for most users)
cp envs/.env.medium .env

# Development — hot-reload, verbose logging
cp envs/.env.dev .env

# High accuracy — ArcFace + retinaface, more RAM
cp envs/.env.high .env
```

> See [Configuration Profiles](#configuration-profiles) for a full comparison.

### 3. Build and start

```bash
docker compose up --build
```

Or run detached in the background:

```bash
docker compose up --build -d
docker compose logs -f   # follow logs
```

### 4. Verify

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "config": { "model": "ArcFace", "detector": "retinaface", ... }
}
```

---

## Configuration Profiles

All settings live in `envs/`. Copy the one that fits your machine to `.env` and restart.

| Profile | File | Model | Detector | RAM | Use case |
|---|---|---|---|---|---|
| **dev** | `.env.dev` | ArcFace | retinaface | 3 GB | Local development, hot-reload |
| **low** | `.env.low` | Facenet512 | opencv | 1 GB | Raspberry Pi / low-power servers |
| **medium** | `.env.medium` | VGG-Face | retinaface | 2 GB | General purpose |
| **high** | `.env.high` | ArcFace | retinaface | 4 GB | Best accuracy, production |
| **gpu** | `.env.gpu` | ArcFace | yolov8 | 4 GB + GPU | CUDA-enabled GPU machines |

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `VGG-Face` | Recognition model: `VGG-Face` · `ArcFace` · `Facenet512` |
| `DETECTOR_BACKEND` | `retinaface` | Detector: `retinaface` · `mtcnn` · `opencv` · `ssd` |
| `WORKERS` | `2` | Uvicorn worker processes |
| `MIN_FACE_CONFIDENCE` | `0.70` | Discard detections below this threshold (0 = accept all) |
| `MAX_IMAGE_PX` | `1920` | Resize images to this max dimension before detection |
| `ENABLE_PREPROCESSING` | `true` | Apply CLAHE contrast enhancement before detection |
| `FACE_TTL_SECONDS` | `86400` | How long face embeddings are kept in cache (24 h) |
| `GROUP_TTL_SECONDS` | `604800` | How long group sessions are kept (7 d) |
| `MAX_UPLOAD_MB` | `20` | Maximum upload file size |
| `LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` |

---

## API Overview

### Meta

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health, config, cache stats, job queue status |
| `POST` | `/admin/purge` | Manually purge all expired cache entries |

### Face Embedding & Cache

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/faces/embed` | Detect faces and store embeddings; returns `face_id` per face |
| `POST` | `/faces/embed/async` | Same, non-blocking; returns a `job_id` |
| `GET` | `/faces/cache` | List all cached faces |
| `GET` | `/faces/cache/{face_id}` | Get a single cached face by ID |
| `DELETE` | `/faces/cache/{face_id}` | Delete a cached face |

### Face Analysis

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/faces/cluster` | DBSCAN clustering within a single image |
| `POST` | `/faces/cluster/async` | Same, non-blocking |
| `POST` | `/faces/cluster-group` | Group many pre-computed cluster centroids by person |
| `POST` | `/faces/group` | Group a list of `face_id`s by person identity |
| `POST` | `/faces/search` | Nearest-neighbour search across the face cache |
| `POST` | `/faces/compare` | Similarity score between two faces |

### Group Sessions

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/sessions` | List all grouping sessions |
| `GET` | `/sessions/{session_id}` | Get a grouping session with all members |
| `DELETE` | `/sessions/{session_id}` | Delete a session |

### Barcodes

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/barcodes/detect` | Detect barcodes in an uploaded image |
| `POST` | `/barcodes/detect-base64` | Same, accepts base64-encoded image in JSON |

### Async Jobs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/jobs` | List all jobs (with optional status filter) |
| `GET` | `/jobs/{job_id}` | Get job status and result |
| `DELETE` | `/jobs/{job_id}` | Cancel or delete a job |

---

## Typical Workflows

### Workflow A — Group photos by person

This is the primary use case: given a folder of photos, find which faces belong to the same person.

```
1. Upload each photo
   POST /faces/embed
   → You get back one face_id per detected face

2. Collect all face_ids across all photos

3. Send them to the group endpoint
   POST /faces/group
   → You get back groups: [[face_id_1, face_id_4], [face_id_2], ...]
     Each inner list = one unique person

4. (Optional) Save the session_id returned — retrieve results later via
   GET /sessions/{session_id}
```

### Workflow B — Find similar faces

```
1. Upload a reference photo
   POST /faces/embed
   → note the face_id

2. Search for similar faces
   POST /faces/search  {"face_id": "...", "top_k": 10, "min_similarity": 0.6}
   → ranked list of matching face_ids with similarity scores

3. Inspect any pair
   POST /faces/compare  {"face_id_a": "...", "face_id_b": "..."}
   → {"similarity": 0.84, "verified": true, "threshold": 0.68}
```

### Workflow C — Barcode scanning

```
1. POST /barcodes/detect  (multipart image)
   → [{"data": "https://example.com", "symbology": "QRCODE",
       "polygon": [...], "bounding_rect": {...}}]
```

---

## cURL Examples

### Health check

```bash
curl http://localhost:8000/health
```

### Embed faces in a photo

```bash
curl -X POST http://localhost:8000/faces/embed \
  -F "file=@photo.jpg"
```

Response:
```json
{
  "faces_found": 2,
  "embeddings": [
    {
      "face_id": "3f7a...",
      "face_confidence": 0.998,
      "face_quality_score": 0.87,
      "facial_area": {"x": 112, "y": 48, "w": 96, "h": 96},
      "embedding_dim": 512,
      "cached": true
    }
  ]
}
```

### Group faces by person identity

```bash
curl -X POST http://localhost:8000/faces/group \
  -H "Content-Type: application/json" \
  -d '{
    "face_ids": ["abc123", "def456", "ghi789", "jkl012"],
    "method": "connected",
    "similarity_threshold": 0.65
  }'
```

Response:
```json
{
  "n_input": 4,
  "n_groups": 2,
  "n_noise": 0,
  "groups": [["abc123", "ghi789"], ["def456", "jkl012"]],
  "noise_face_ids": [],
  "method": "connected",
  "session_id": "..."
}
```

### Search for similar faces

```bash
curl -X POST http://localhost:8000/faces/search \
  -H "Content-Type: application/json" \
  -d '{
    "face_id": "abc123",
    "top_k": 5,
    "min_similarity": 0.60
  }'
```

### Compare two faces

```bash
curl -X POST http://localhost:8000/faces/compare \
  -H "Content-Type: application/json" \
  -d '{
    "face_id_a": "abc123",
    "face_id_b": "def456"
  }'
```

### Detect barcodes

```bash
curl -X POST http://localhost:8000/barcodes/detect \
  -F "file=@product_label.jpg"
```

### Submit an async embed job

```bash
# Submit
curl -X POST "http://localhost:8000/faces/embed/async" \
  -F "file=@photo.jpg"
# → {"job_id": "xyz789", "status": "PENDING"}

# Poll for result
curl http://localhost:8000/jobs/xyz789
# → {"status": "DONE", "result": {...}}
```

---

## Admin Dashboard

Open **http://localhost:8000/admin** in your browser for a visual interface.

| Tab | What you can do |
|---|---|
| **Upload** | Upload images, see detected faces and barcodes with overlay boxes |
| **Async Embed** | Batch-upload multiple images with live progress tracking |
| **Face Cache** | Browse all cached face embeddings, trigger group analysis, inspect quality scores |
| **Jobs** | Monitor async job queue, view results, cancel running jobs |
| **Tools** | Face Search, Face Compare, and Face Group with visual results |

---

## Architecture

```
vision-api/
├── main.py              ← FastAPI app, all endpoints, image processing
├── db.py                ← SQLite persistence (face cache, sessions, jobs)
├── clustering.py        ← Connected components, HDBSCAN, agglomerative
├── queue_worker.py      ← Async job queue with webhook callbacks
├── errors.py            ← Typed error codes and exception handlers
├── admin.html           ← Single-file admin dashboard UI
├── envs/
│   ├── .env.dev         ← Development profile
│   ├── .env.low         ← Low-resource profile
│   ├── .env.medium      ← Balanced profile
│   ├── .env.high        ← High-accuracy profile
│   └── .env.gpu         ← GPU profile
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Data flow

```
Image upload
    │
    ▼
Image decode + resize (MAX_IMAGE_PX)
    │
    ▼
CLAHE pre-processing (if ENABLE_PREPROCESSING=true)
    │
    ▼
DeepFace.represent()
  ├─ Face detection (retinaface / mtcnn / opencv)
  └─ Embedding (ArcFace / VGG-Face / Facenet512)
    │
    ▼
Quality score (Laplacian sharpness)
Confidence filter (MIN_FACE_CONFIDENCE)
    │
    ▼
SQLite face cache  →  face_id (UUID) returned to caller
    │
    ▼
/faces/group  →  cosine similarity matrix  →  connected components / HDBSCAN
    │
    ▼
Group session stored in SQLite  →  session_id returned
```

### Grouping algorithms

| Algorithm | Flag | Best for |
|---|---|---|
| **Connected Components** | `"connected"` | Person grouping with explicit similarity threshold; no noise labels; recommended default |
| **HDBSCAN** | `"hdbscan"` | Large collections with noise/outliers; fully automatic group count |
| **Agglomerative** | `"agglomerative"` | Hierarchical merging; predictable via `distance_threshold` |

---

## Deployment Profiles

### Standard CPU (default)

```bash
cp envs/.env.medium .env
docker compose up --build -d
```

### High accuracy

```bash
cp envs/.env.high .env
docker compose up --build -d
```

### GPU (NVIDIA)

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
cp envs/.env.gpu .env
docker compose --profile gpu up --build -d
```

### Development (hot-reload)

```bash
cp envs/.env.dev .env
docker compose --profile dev up
```

Source files are mounted into the container — any change to `.py` or `.html` files reloads immediately without rebuilding the image.

---

## FAQ

**Q: The first start is slow / model weights are re-downloaded every time.**  
A: Make sure Docker volumes are not being deleted between runs. The `models` volume persists weights at `/root/.deepface/weights`. Run `docker volume ls` to verify it exists.

**Q: Face detection misses faces or returns low confidence.**  
A: Try lowering `MIN_FACE_CONFIDENCE` (e.g. `0.50`) or switching to a more sensitive detector (`DETECTOR_BACKEND=mtcnn`). For dark/backlit images, enable `ENABLE_PREPROCESSING=true`.

**Q: The grouping puts different people in the same group.**  
A: Raise `similarity_threshold` (e.g. `0.70` or `0.75`). For the best accuracy, use `MODEL_NAME=ArcFace`.

**Q: I want to use this without Docker.**  
A: Install dependencies with `pip install -r requirements.txt` and run `uvicorn main:app --host 0.0.0.0 --port 8000`. You also need `libzbar0` installed at the OS level (`apt install libzbar0` / `brew install zbar`).

**Q: How do I reset all cached data?**  
A: `docker compose down -v` removes all volumes including the SQLite database and model weights. To keep model weights but clear the database: `docker volume rm vision-api_data`.

**Q: Can I run multiple workers with the face cache?**  
A: Yes. SQLite runs in WAL mode, which supports concurrent readers and one writer safely. For high concurrency (>10 req/s) consider switching `DB_PATH` to a PostgreSQL-compatible store.

---

## License

MIT — see [LICENSE](LICENSE).
