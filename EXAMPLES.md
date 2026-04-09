# Vision API – Vollständige Endpoint-Beispiele

Alle Beispiele beziehen sich auf `http://localhost:8000`.  
Interaktive Swagger-UI: **http://localhost:8000/docs**  
ReDoc: **http://localhost:8000/redoc**

---

## Inhaltsverzeichnis

1. [Meta](#1-meta)
   - [GET /health](#get-health)
   - [POST /admin/purge](#post-adminpurge)
2. [Faces – Embedding](#2-faces--embedding)
   - [POST /faces/embed](#post-facesembed)
3. [Faces – Clustering](#3-faces--clustering)
   - [POST /faces/cluster](#post-facescluster)
4. [Faces – Cluster Grouping](#4-faces--cluster-grouping)
   - [POST /faces/cluster-group (Mode A: raw centroids)](#post-facescluster-group--mode-a-raw-centroids)
   - [POST /faces/cluster-group (Mode B: face IDs)](#post-facescluster-group--mode-b-face-ids)
   - [POST /faces/cluster-group (Mode C: mixed + kmeans)](#post-facescluster-group--mode-c-mixed--kmeans)
5. [Face Cache](#5-face-cache)
   - [GET /faces/cache](#get-facescache)
   - [GET /faces/cache/{face_id}](#get-facescacheface_id)
   - [DELETE /faces/cache/{face_id}](#delete-facescacheface_id)
6. [Group Sessions](#6-group-sessions)
   - [GET /sessions](#get-sessions)
   - [GET /sessions/{session_id}](#get-sessionssession_id)
   - [DELETE /sessions/{session_id}](#delete-sessionssession_id)
7. [Barcodes](#7-barcodes)
   - [POST /barcodes/detect](#post-barcodesdetect)
   - [POST /barcodes/detect-base64](#post-barcodesdetect-base64)
8. [Async Jobs](#8-async-jobs)
   - [POST /faces/embed/async](#post-facesembedasync)
   - [POST /faces/cluster/async](#post-facesclusterasync)
   - [GET /jobs](#get-jobs)
   - [GET /jobs/{job_id}](#get-jobsjob_id)
   - [DELETE /jobs/{job_id}](#delete-jobsjob_id)
9. [Fehlerformate](#9-fehlerformate)
10. [Typische Workflows](#10-typische-workflows)

---

## 1. Meta

### GET /health

Gibt den aktuellen Servicestatus, Konfiguration und Cache-Statistiken zurück.

**Request**
```bash
curl http://localhost:8000/health
```

**Response 200**
```json
{
  "status": "ok",
  "config": {
    "model": "VGG-Face",
    "detector": "retinaface",
    "max_upload_mb": 20,
    "face_ttl_seconds": 86400,
    "group_session_ttl_seconds": 604800
  },
  "cache": {
    "faces": {
      "total": 142,
      "active": 138,
      "expired": 4
    },
    "group_sessions_active": 7
  },
  "queue": {
    "pending": 0,
    "running": 1,
    "max_size": 50,
    "concurrency": 1,
    "job_counts": {
      "DONE": 23,
      "FAILED": 1,
      "RUNNING": 1
    }
  }
}
```

---

### POST /admin/purge

Löscht sofort alle abgelaufenen Face-IDs und Group-Sessions.

**Request**
```bash
curl -X POST http://localhost:8000/admin/purge
```

**Response 200**
```json
{
  "purged_records": 17
}
```

---

## 2. Faces – Embedding

### POST /faces/embed

Erkennt alle Gesichter in einem Bild und berechnet Embedding-Vektoren.  
Jedes Gesicht bekommt eine `face_id` und wird im Cache gespeichert.

**Request**
```bash
curl -X POST http://localhost:8000/faces/embed \
  -F "file=@foto_gruppe.jpg"
```

Mit TTL-Override (12 Stunden statt Default):
```bash
curl -X POST "http://localhost:8000/faces/embed?ttl=43200" \
  -F "file=@foto_gruppe.jpg"
```

Ohne Cache (nur Vektoren zurückgeben):
```bash
curl -X POST "http://localhost:8000/faces/embed?cache=false" \
  -F "file=@portrait.jpg"
```

**Response 200** – 2 Gesichter erkannt
```json
{
  "source_file": "foto_gruppe.jpg",
  "faces_found": 2,
  "model": "VGG-Face",
  "faces": [
    {
      "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
      "face_index": 0,
      "face_confidence": 0.9821,
      "facial_area": {
        "x": 124,
        "y": 89,
        "w": 156,
        "h": 178
      },
      "embedding": [0.0312, -0.1547, 0.2089, -0.0743, 0.1892, "... (4096 Werte)"],
      "embedding_dim": 4096,
      "cached": true,
      "expires_at": 1712518400.0
    },
    {
      "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
      "face_index": 1,
      "face_confidence": 0.9543,
      "facial_area": {
        "x": 380,
        "y": 102,
        "w": 143,
        "h": 165
      },
      "embedding": [-0.0821, 0.1234, -0.0456, 0.2341, -0.1123, "... (4096 Werte)"],
      "embedding_dim": 4096,
      "cached": true,
      "expires_at": 1712518400.0
    }
  ]
}
```

**Response 422** – kein Gesicht erkannt
```json
{
  "error": true,
  "code": "NO_FACE_DETECTED",
  "message": "No faces could be detected in the image. Try a higher-resolution image or a different detector backend.",
  "detail": {
    "image": "landschaft.jpg"
  }
}
```

---

## 3. Faces – Clustering

### POST /faces/cluster

Erkennt Gesichter in einem Bild und clustert sie via DBSCAN.  
Nützlich um doppelte / ähnliche Gesichter im selben Bild zu gruppieren.

**Request**
```bash
curl -X POST "http://localhost:8000/faces/cluster?min_similarity=0.65" \
  -F "file=@gruppenphoto.jpg"
```

**Response 200** – 5 Gesichter in 3 Cluster
```json
{
  "source_file": "gruppenphoto.jpg",
  "faces_found": 5,
  "n_clusters": 3,
  "n_noise": 1,
  "model": "VGG-Face",
  "min_similarity": 0.65,
  "faces": [
    {
      "face_id": "aabb1122-ccdd-3344-eeff-556677889900",
      "face_index": 0,
      "cluster_id": 0,
      "face_confidence": 0.987,
      "facial_area": { "x": 50, "y": 30, "w": 120, "h": 140 },
      "cached": true
    },
    {
      "face_id": "bbcc2233-ddee-4455-ffaa-667788990011",
      "face_index": 1,
      "cluster_id": 0,
      "face_confidence": 0.963,
      "facial_area": { "x": 210, "y": 45, "w": 118, "h": 138 },
      "cached": true
    },
    {
      "face_id": "ccdd3344-eeff-5566-aabb-778899001122",
      "face_index": 2,
      "cluster_id": 1,
      "face_confidence": 0.941,
      "facial_area": { "x": 400, "y": 55, "w": 130, "h": 150 },
      "cached": true
    },
    {
      "face_id": "ddee4455-ffaa-6677-bbcc-889900112233",
      "face_index": 3,
      "cluster_id": 2,
      "face_confidence": 0.978,
      "facial_area": { "x": 580, "y": 40, "w": 125, "h": 145 },
      "cached": true
    },
    {
      "face_id": "eeff5566-aabb-7788-ccdd-990011223344",
      "face_index": 4,
      "cluster_id": -1,
      "face_confidence": 0.512,
      "facial_area": { "x": 720, "y": 80, "w": 90, "h": 100 },
      "cached": true
    }
  ]
}
```

> `cluster_id: -1` = Noise — Gesicht erkannt, aber zu niedrige Ähnlichkeit zu jedem Cluster.

---

## 4. Faces – Cluster Grouping

### POST /faces/cluster-group – Mode A: raw centroids

Mehrere vorberechnete Cluster (z.B. aus verschiedenen Fotos) zu Personen-Gruppen zusammenführen.

**Request**
```bash
curl -X POST http://localhost:8000/faces/cluster-group \
  -H "Content-Type: application/json" \
  -d '{
    "clusters": [
      {
        "cluster_id": "foto001-gesicht0",
        "centroid": [0.0312, -0.1547, 0.2089, -0.0743],
        "size": 3,
        "metadata": { "quelle": "urlaub2023.jpg", "datum": "2023-07-15" }
      },
      {
        "cluster_id": "foto001-gesicht1",
        "centroid": [-0.0821, 0.1234, -0.0456, 0.2341],
        "size": 2,
        "metadata": { "quelle": "urlaub2023.jpg", "datum": "2023-07-15" }
      },
      {
        "cluster_id": "foto002-gesicht0",
        "centroid": [0.0298, -0.1531, 0.2101, -0.0758],
        "size": 5,
        "metadata": { "quelle": "geburtstag.jpg", "datum": "2023-11-20" }
      },
      {
        "cluster_id": "foto003-gesicht0",
        "centroid": [-0.0835, 0.1218, -0.0471, 0.2329],
        "size": 1,
        "metadata": { "quelle": "weihnachten.jpg", "datum": "2023-12-24" }
      }
    ],
    "method": "hdbscan",
    "min_cluster_size": 2,
    "persist": true
  }'
```

**Response 200**
```json
{
  "session_id": "9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f",
  "method": "hdbscan",
  "input_clusters": 4,
  "n_groups": 2,
  "n_noise": 0,
  "persisted": true,
  "groups": [
    {
      "group_id": 0,
      "cluster_count": 2,
      "members": [
        {
          "cluster_id": "foto001-gesicht0",
          "face_id": null,
          "size": 3,
          "metadata": { "quelle": "urlaub2023.jpg", "datum": "2023-07-15" }
        },
        {
          "cluster_id": "foto002-gesicht0",
          "face_id": null,
          "size": 5,
          "metadata": { "quelle": "geburtstag.jpg", "datum": "2023-11-20" }
        }
      ]
    },
    {
      "group_id": 1,
      "cluster_count": 2,
      "members": [
        {
          "cluster_id": "foto001-gesicht1",
          "face_id": null,
          "size": 2,
          "metadata": { "quelle": "urlaub2023.jpg", "datum": "2023-07-15" }
        },
        {
          "cluster_id": "foto003-gesicht0",
          "face_id": null,
          "size": 1,
          "metadata": { "quelle": "weihnachten.jpg", "datum": "2023-12-24" }
        }
      ]
    }
  ]
}
```

---

### POST /faces/cluster-group – Mode B: face IDs

Nur Face-IDs aus dem Cache senden – keine Vektoren nötig.

**Request**
```bash
curl -X POST http://localhost:8000/faces/cluster-group \
  -H "Content-Type: application/json" \
  -d '{
    "face_id_clusters": [
      {
        "cluster_id": "session-a-person1",
        "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
        "size": 4,
        "metadata": { "album": "Sommer 2023" }
      },
      {
        "cluster_id": "session-a-person2",
        "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
        "size": 2,
        "metadata": { "album": "Sommer 2023" }
      },
      {
        "cluster_id": "session-b-person1",
        "face_id": "aabb1122-ccdd-3344-eeff-556677889900",
        "size": 7,
        "metadata": { "album": "Winter 2023" }
      }
    ],
    "method": "agglomerative",
    "distance_threshold": 0.35,
    "persist": true,
    "session_ttl": 259200
  }'
```

**Response 200**
```json
{
  "session_id": "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
  "method": "agglomerative",
  "input_clusters": 3,
  "n_groups": 2,
  "n_noise": 0,
  "persisted": true,
  "groups": [
    {
      "group_id": 0,
      "cluster_count": 2,
      "members": [
        {
          "cluster_id": "session-a-person1",
          "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
          "size": 4,
          "metadata": { "album": "Sommer 2023" }
        },
        {
          "cluster_id": "session-b-person1",
          "face_id": "aabb1122-ccdd-3344-eeff-556677889900",
          "size": 7,
          "metadata": { "album": "Winter 2023" }
        }
      ]
    },
    {
      "group_id": 1,
      "cluster_count": 1,
      "members": [
        {
          "cluster_id": "session-a-person2",
          "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
          "size": 2,
          "metadata": { "album": "Sommer 2023" }
        }
      ]
    }
  ]
}
```

**Response 404** – face_id nicht im Cache oder abgelaufen
```json
{
  "error": true,
  "code": "FACE_ID_NOT_FOUND",
  "message": "Face ID '7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d' was not found in the cache.",
  "detail": {
    "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d"
  }
}
```

---

### POST /faces/cluster-group – Mode C: mixed + kmeans

Gemischter Input (raw centroids + face IDs) mit fixer Gruppenanzahl via k-Means.

**Request**
```bash
curl -X POST http://localhost:8000/faces/cluster-group \
  -H "Content-Type: application/json" \
  -d '{
    "clusters": [
      {
        "cluster_id": "archiv-alt-001",
        "centroid": [0.1231, -0.0892, 0.2341, 0.0543],
        "size": 12
      }
    ],
    "face_id_clusters": [
      {
        "cluster_id": "neu-foto-gesicht0",
        "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
        "size": 2
      },
      {
        "cluster_id": "neu-foto-gesicht1",
        "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
        "size": 1
      }
    ],
    "method": "kmeans",
    "n_groups": 2,
    "persist": false
  }'
```

**Response 200** – `persist: false` → keine session_id
```json
{
  "session_id": null,
  "method": "kmeans",
  "input_clusters": 3,
  "n_groups": 2,
  "n_noise": 0,
  "persisted": false,
  "groups": [
    {
      "group_id": 0,
      "cluster_count": 2,
      "members": [
        {
          "cluster_id": "archiv-alt-001",
          "face_id": null,
          "size": 12,
          "metadata": {}
        },
        {
          "cluster_id": "neu-foto-gesicht0",
          "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
          "size": 2,
          "metadata": {}
        }
      ]
    },
    {
      "group_id": 1,
      "cluster_count": 1,
      "members": [
        {
          "cluster_id": "neu-foto-gesicht1",
          "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
          "size": 1,
          "metadata": {}
        }
      ]
    }
  ]
}
```

**Response 400** – Embedding-Dimensionen stimmen nicht überein (verschiedene Modelle gemischt)
```json
{
  "error": true,
  "code": "EMBEDDING_DIM_MISMATCH",
  "message": "Embedding dimension mismatch for face_id 'neu-foto-gesicht1': expected 4096, got 512. All embeddings must use the same model.",
  "detail": {
    "face_id": "neu-foto-gesicht1",
    "expected": 4096,
    "got": 512
  }
}
```

---

## 5. Face Cache

### GET /faces/cache

Listet alle aktiven (nicht abgelaufenen) Face-IDs. Kein Embedding-Vektor enthalten.

**Request**
```bash
curl "http://localhost:8000/faces/cache?limit=10&offset=0"
```

**Response 200**
```json
{
  "count": 3,
  "offset": 0,
  "faces": [
    {
      "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
      "source_file": "gruppenphoto.jpg",
      "face_index": 0,
      "embedding_dim": 4096,
      "face_confidence": 0.9821,
      "facial_area": { "x": 124, "y": 89, "w": 156, "h": 178 },
      "model_name": "VGG-Face",
      "created_at": 1712432000.0,
      "expires_at": 1712518400.0,
      "expires_in_seconds": 81234.5
    },
    {
      "face_id": "7a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
      "source_file": "gruppenphoto.jpg",
      "face_index": 1,
      "embedding_dim": 4096,
      "face_confidence": 0.9543,
      "facial_area": { "x": 380, "y": 102, "w": 143, "h": 165 },
      "model_name": "VGG-Face",
      "created_at": 1712432000.0,
      "expires_at": 1712518400.0,
      "expires_in_seconds": 81234.5
    }
  ]
}
```

---

### GET /faces/cache/{face_id}

Ruft einen einzelnen gecachten Eintrag ab – optional mit Embedding-Vektor.

**Request** – mit Embedding
```bash
curl "http://localhost:8000/faces/cache/3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d"
```

**Request** – nur Metadaten (ohne Vektor)
```bash
curl "http://localhost:8000/faces/cache/3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d?include_embedding=false"
```

**Response 200**
```json
{
  "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
  "source_file": "gruppenphoto.jpg",
  "face_index": 0,
  "embedding": [0.0312, -0.1547, 0.2089, -0.0743, "... (4096 Werte)"],
  "embedding_dim": 4096,
  "facial_area": { "x": 124, "y": 89, "w": 156, "h": 178 },
  "face_confidence": 0.9821,
  "model_name": "VGG-Face",
  "created_at": 1712432000.0,
  "expires_at": 1712518400.0,
  "expires_in_seconds": 81122.3
}
```

**Response 404** – nicht gefunden oder abgelaufen
```json
{
  "error": true,
  "code": "FACE_ID_NOT_FOUND",
  "message": "Face ID '3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d' was not found in the cache.",
  "detail": {
    "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d"
  }
}
```

---

### DELETE /faces/cache/{face_id}

Löscht sofort einen gecachten Eintrag.

**Request**
```bash
curl -X DELETE \
  "http://localhost:8000/faces/cache/3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d"
```

**Response 200**
```json
{
  "deleted": true,
  "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d"
}
```

---

## 6. Group Sessions

### GET /sessions

Listet alle aktiven (nicht abgelaufenen) Grouping-Sessions.

**Request**
```bash
curl "http://localhost:8000/sessions?limit=20"
```

**Response 200**
```json
{
  "count": 2,
  "sessions": [
    {
      "session_id": "9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f",
      "method": "hdbscan",
      "params": {
        "n_groups": null,
        "min_cluster_size": 2,
        "distance_threshold": 0.4
      },
      "n_input": 4,
      "n_groups": 2,
      "n_noise": 0,
      "created_at": 1712432100.0,
      "expires_at": 1713036900.0,
      "expires_in_seconds": 604321.8
    },
    {
      "session_id": "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
      "method": "agglomerative",
      "params": {
        "n_groups": null,
        "min_cluster_size": 2,
        "distance_threshold": 0.35
      },
      "n_input": 3,
      "n_groups": 2,
      "n_noise": 0,
      "created_at": 1712433200.0,
      "expires_at": 1712692400.0,
      "expires_in_seconds": 258723.1
    }
  ]
}
```

---

### GET /sessions/{session_id}

Gibt das vollständige Grouping-Ergebnis einer Session zurück.

**Request**
```bash
curl "http://localhost:8000/sessions/9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f"
```

**Response 200**
```json
{
  "session_id": "9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f",
  "method": "hdbscan",
  "params": {
    "n_groups": null,
    "min_cluster_size": 2,
    "distance_threshold": 0.4
  },
  "n_input": 4,
  "n_groups": 2,
  "n_noise": 0,
  "created_at": 1712432100.0,
  "expires_at": 1713036900.0,
  "expires_in_seconds": 604185.2,
  "groups": [
    {
      "group_id": 0,
      "cluster_count": 2,
      "members": [
        {
          "cluster_id": "foto001-gesicht0",
          "face_id": null,
          "size": 3,
          "metadata": { "quelle": "urlaub2023.jpg" }
        },
        {
          "cluster_id": "foto002-gesicht0",
          "face_id": null,
          "size": 5,
          "metadata": { "quelle": "geburtstag.jpg" }
        }
      ]
    },
    {
      "group_id": 1,
      "cluster_count": 2,
      "members": [
        {
          "cluster_id": "foto001-gesicht1",
          "face_id": null,
          "size": 2,
          "metadata": { "quelle": "urlaub2023.jpg" }
        },
        {
          "cluster_id": "foto003-gesicht0",
          "face_id": null,
          "size": 1,
          "metadata": { "quelle": "weihnachten.jpg" }
        }
      ]
    }
  ]
}
```

**Response 404** – Session nicht gefunden oder abgelaufen
```json
{
  "error": true,
  "code": "SESSION_NOT_FOUND",
  "message": "Group session '9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f' not found or has expired.",
  "detail": {
    "session_id": "9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f"
  }
}
```

---

### DELETE /sessions/{session_id}

**Request**
```bash
curl -X DELETE \
  "http://localhost:8000/sessions/9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f"
```

**Response 200**
```json
{
  "deleted": true,
  "session_id": "9f8e7d6c-5b4a-3c2d-1e0f-9a8b7c6d5e4f"
}
```

---

## 7. Barcodes

### POST /barcodes/detect

Erkennt alle Barcodes in einem hochgeladenen Bild (0 bis n Stück).

**Request** – Datei-Upload
```bash
curl -X POST http://localhost:8000/barcodes/detect \
  -F "file=@produkt_regal.jpg"
```

**Response 200** – 2 Barcodes gefunden
```json
{
  "source_file": "produkt_regal.jpg",
  "barcodes_found": 2,
  "barcodes": [
    {
      "data": "9783161484100",
      "symbology": "EAN13",
      "symbology_friendly": "EAN-13",
      "polygon": [
        { "x": 142, "y": 380 },
        { "x": 312, "y": 380 },
        { "x": 312, "y": 420 },
        { "x": 142, "y": 420 }
      ],
      "bounding_rect": {
        "left": 142,
        "top": 380,
        "width": 170,
        "height": 40
      },
      "quality": 1
    },
    {
      "data": "https://example.com/produkt/12345",
      "symbology": "QRCODE",
      "symbology_friendly": "QR Code",
      "polygon": [
        { "x": 500, "y": 200 },
        { "x": 650, "y": 200 },
        { "x": 650, "y": 350 },
        { "x": 500, "y": 350 }
      ],
      "bounding_rect": {
        "left": 500,
        "top": 200,
        "width": 150,
        "height": 150
      },
      "quality": 1
    }
  ]
}
```

**Response 200** – kein Barcode gefunden (kein Fehler – leeres Array)
```json
{
  "source_file": "portrait.jpg",
  "barcodes_found": 0,
  "barcodes": []
}
```

---

### POST /barcodes/detect-base64

Gleiche Funktion, aber Bild wird als Base64-String im JSON-Body übergeben.

**Request**
```bash
# Bild zu Base64 konvertieren und senden
IMAGE_B64=$(base64 -w 0 produkt.jpg)

curl -X POST http://localhost:8000/barcodes/detect-base64 \
  -H "Content-Type: application/json" \
  -d "{\"image_b64\": \"${IMAGE_B64}\"}"
```

**Request Body (JSON)**
```json
{
  "image_b64": "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQE..."
}
```

**Response 200** – gleiche Struktur wie `/barcodes/detect`, nur ohne `source_file`
```json
{
  "barcodes_found": 1,
  "barcodes": [
    {
      "data": "4006381333931",
      "symbology": "EAN13",
      "symbology_friendly": "EAN-13",
      "polygon": [
        { "x": 80, "y": 120 },
        { "x": 240, "y": 120 },
        { "x": 240, "y": 160 },
        { "x": 80, "y": 160 }
      ],
      "bounding_rect": { "left": 80, "top": 120, "width": 160, "height": 40 },
      "quality": 1
    }
  ]
}
```

**Response 400** – ungültiger Base64-String
```json
{
  "error": true,
  "code": "INVALID_BASE64",
  "message": "The provided string is not valid base64.",
  "detail": null
}
```

---

## 8. Async Jobs

### POST /faces/embed/async

Bild in die Queue einreihen – sofort `job_id` zurück (HTTP 202).  
Ergebnis wird per Webhook an den konfigurierten Callback-Server gepostet.

**Request** – mit Callback und Basic Auth
```bash
IMAGE_B64=$(base64 -w 0 gruppe.jpg)

curl -X POST http://localhost:8000/faces/embed/async \
  -H "Content-Type: application/json" \
  -d "{
    \"image_b64\": \"${IMAGE_B64}\",
    \"source_file\": \"gruppe.jpg\",
    \"cache\": true,
    \"ttl\": 86400,
    \"callback\": {
      \"url\": \"https://meinserver.de/api/webhooks/vision\",
      \"auth_user\": \"webhook-user\",
      \"auth_pass\": \"geheimespasswort\"
    }
  }"
```

**Request** – ohne Callback (nur Polling)
```bash
curl -X POST http://localhost:8000/faces/embed/async \
  -H "Content-Type: application/json" \
  -d "{
    \"image_b64\": \"${IMAGE_B64}\",
    \"source_file\": \"portrait.jpg\"
  }"
```

**Response 202** – Job angenommen
```json
{
  "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
  "status": "PENDING",
  "message": "Job queued. Poll GET /jobs/{job_id} or await the callback.",
  "callback_url": "https://meinserver.de/api/webhooks/vision",
  "auth_enabled": true,
  "queue_depth": 2
}
```

**Response 503** – Queue ist voll
```json
{
  "error": true,
  "code": "QUEUE_FULL",
  "message": "The job queue is full (50 items). Try again later.",
  "detail": null
}
```

---

#### Callback-Payload (wird vom Server an deine URL gepostet)

```http
POST https://meinserver.de/api/webhooks/vision
Authorization: Basic d2ViaG9vay11c2VyOmdlaGVpbWVzcGFzc3dvcnQ=
Content-Type: application/json
X-Vision-Job-Id: f1e2d3c4-b5a6-7890-abcd-ef1234567890
```

```json
{
  "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
  "status": "DONE",
  "job_type": "embed",
  "submitted_at": 1712432000.0,
  "started_at": 1712432003.5,
  "completed_at": 1712432015.8,
  "duration_seconds": 15.8,
  "error": null,
  "result": {
    "source_file": "gruppe.jpg",
    "faces_found": 2,
    "model": "VGG-Face",
    "faces": [
      {
        "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
        "face_index": 0,
        "face_confidence": 0.9821,
        "facial_area": { "x": 124, "y": 89, "w": 156, "h": 178 },
        "embedding": [0.0312, -0.1547, "..."],
        "embedding_dim": 4096,
        "cached": true,
        "expires_at": 1712518400.0
      }
    ]
  }
}
```

#### Callback bei Fehler

```json
{
  "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
  "status": "FAILED",
  "job_type": "embed",
  "submitted_at": 1712432000.0,
  "started_at": 1712432003.5,
  "completed_at": 1712432004.1,
  "duration_seconds": 0.6,
  "error": "VisionAPIException: No faces could be detected in the image.",
  "result": null
}
```

#### Callback bei Timeout

```json
{
  "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
  "status": "TIMEOUT",
  "job_type": "embed",
  "submitted_at": 1712432000.0,
  "started_at": 1712432003.5,
  "completed_at": 1712432123.5,
  "duration_seconds": 120.0,
  "error": "Job exceeded timeout of 120s",
  "result": null
}
```

---

### POST /faces/cluster/async

Gleich wie `/faces/embed/async`, aber mit DBSCAN-Clustering.

**Request**
```bash
IMAGE_B64=$(base64 -w 0 gruppenphoto.jpg)

curl -X POST http://localhost:8000/faces/cluster/async \
  -H "Content-Type: application/json" \
  -d "{
    \"image_b64\": \"${IMAGE_B64}\",
    \"source_file\": \"gruppenphoto.jpg\",
    \"min_similarity\": 0.65,
    \"cache\": true,
    \"callback\": {
      \"url\": \"https://meinserver.de/api/webhooks/vision\"
    }
  }"
```

**Response 202**
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef0987654321",
  "status": "PENDING",
  "message": "Job queued. Poll GET /jobs/{job_id} or await the callback.",
  "callback_url": "https://meinserver.de/api/webhooks/vision",
  "auth_enabled": false,
  "queue_depth": 1
}
```

---

### GET /jobs

Alle Jobs abrufen (Memory + DB), optional nach Status filtern.

**Request** – alle Jobs
```bash
curl "http://localhost:8000/jobs"
```

**Request** – nur fertige Jobs
```bash
curl "http://localhost:8000/jobs?status=DONE&limit=50"
```

**Response 200**
```json
{
  "count": 3,
  "queue": {
    "pending": 1,
    "running": 1,
    "max_size": 50,
    "concurrency": 1,
    "job_counts": {
      "PENDING": 1,
      "RUNNING": 1,
      "DONE": 1
    }
  },
  "jobs": [
    {
      "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
      "job_type": "embed",
      "status": "DONE",
      "callback_url": "https://meinserver.de/api/webhooks/vision",
      "submitted_at": 1712432000.0,
      "started_at": 1712432003.5,
      "completed_at": 1712432015.8,
      "duration_seconds": 15.8,
      "error": null
    },
    {
      "job_id": "a1b2c3d4-e5f6-7890-abcd-ef0987654321",
      "job_type": "cluster",
      "status": "RUNNING",
      "callback_url": "https://meinserver.de/api/webhooks/vision",
      "submitted_at": 1712432020.0,
      "started_at": 1712432021.1,
      "completed_at": null,
      "duration_seconds": null,
      "error": null
    },
    {
      "job_id": "b2c3d4e5-f6a7-8901-bcde-f01234567890",
      "job_type": "embed",
      "status": "PENDING",
      "callback_url": null,
      "submitted_at": 1712432025.0,
      "started_at": null,
      "completed_at": null,
      "duration_seconds": null,
      "error": null
    }
  ]
}
```

---

### GET /jobs/{job_id}

Status und Ergebnis eines einzelnen Jobs.

**Request**
```bash
curl "http://localhost:8000/jobs/f1e2d3c4-b5a6-7890-abcd-ef1234567890"
```

**Response 200 – DONE** (Ergebnis vollständig enthalten)
```json
{
  "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890",
  "job_type": "embed",
  "status": "DONE",
  "callback_url": "https://meinserver.de/api/webhooks/vision",
  "submitted_at": 1712432000.0,
  "started_at": 1712432003.5,
  "completed_at": 1712432015.8,
  "duration_seconds": 15.8,
  "error": null,
  "result": {
    "source_file": "gruppe.jpg",
    "faces_found": 2,
    "model": "VGG-Face",
    "faces": [
      {
        "face_id": "3f7a2b1c-e4d5-4a6b-9c8d-1e2f3a4b5c6d",
        "face_index": 0,
        "face_confidence": 0.9821,
        "facial_area": { "x": 124, "y": 89, "w": 156, "h": 178 },
        "embedding": [0.0312, -0.1547, "..."],
        "embedding_dim": 4096,
        "cached": true,
        "expires_at": 1712518400.0
      }
    ]
  }
}
```

**Response 200 – PENDING** (noch in der Queue)
```json
{
  "job_id": "b2c3d4e5-f6a7-8901-bcde-f01234567890",
  "job_type": "embed",
  "status": "PENDING",
  "callback_url": null,
  "submitted_at": 1712432025.0,
  "started_at": null,
  "completed_at": null,
  "duration_seconds": null,
  "error": null,
  "result": null
}
```

**Response 404** – Job nicht gefunden
```json
{
  "error": true,
  "code": "JOB_NOT_FOUND",
  "message": "Job 'xyz' not found. It may have expired or never existed.",
  "detail": {
    "job_id": "xyz"
  }
}
```

---

### DELETE /jobs/{job_id}

Fertigen Job löschen (PENDING/RUNNING können nicht gelöscht werden).

**Request**
```bash
curl -X DELETE \
  "http://localhost:8000/jobs/f1e2d3c4-b5a6-7890-abcd-ef1234567890"
```

**Response 200**
```json
{
  "deleted": true,
  "job_id": "f1e2d3c4-b5a6-7890-abcd-ef1234567890"
}
```

**Response 409** – Job läuft noch
```json
{
  "error": true,
  "code": "JOB_ACTIVE",
  "message": "Cannot delete a PENDING or RUNNING job.",
  "detail": {
    "job_id": "a1b2c3d4-e5f6-7890-abcd-ef0987654321",
    "status": "RUNNING"
  }
}
```

---

## 9. Fehlerformate

Alle Fehler folgen einem einheitlichen Schema:

```json
{
  "error": true,
  "code": "MACHINE_READABLE_CODE",
  "message": "Menschenlesbare Fehlerbeschreibung.",
  "detail": { }
}
```

### Vollständige Fehlercodes

| Code | HTTP | Beschreibung |
|---|---|---|
| `IMAGE_DECODE_FAILED` | 400 | Datei ist kein gültiges Bildformat |
| `FILE_TOO_LARGE` | 413 | Datei überschreitet `MAX_UPLOAD_MB` |
| `NO_FACE_DETECTED` | 422 | Kein Gesicht im Bild erkannt |
| `INVALID_BASE64` | 400 | Ungültiger Base64-String |
| `INVALID_PARAMETER` | 400 | Ungültiger Query- oder Body-Parameter |
| `FACE_ID_NOT_FOUND` | 404 | face_id nicht im Cache oder unbekannt |
| `FACE_ID_EXPIRED` | 410 | face_id existiert aber ist abgelaufen |
| `SESSION_NOT_FOUND` | 404 | Grouping-Session nicht gefunden |
| `TOO_FEW_CLUSTERS` | 400 | Weniger als 2 Cluster für Grouping |
| `UNKNOWN_METHOD` | 400 | Unbekannter Clustering-Algorithmus |
| `MISSING_PARAMETER` | 400 | Pflichtparameter fehlt |
| `EMBEDDING_DIM_MISMATCH` | 400 | Verschiedene Embedding-Dimensionen gemischt |
| `MODEL_LOAD_FAILED` | 503 | DeepFace-Modell konnte nicht geladen werden |
| `INFERENCE_FAILED` | 500 | Fehler während der Modell-Inferenz |
| `CLUSTERING_FAILED` | 500 | Fehler im Clustering-Algorithmus |
| `DB_ERROR` | 500 | Datenbankfehler |
| `INTERNAL_ERROR` | 500 | Unerwarteter interner Fehler |
| `QUEUE_FULL` | 503 | Job-Queue hat maximale Kapazität erreicht |
| `JOB_NOT_FOUND` | 404 | Job-ID nicht gefunden oder abgelaufen |
| `JOB_ACTIVE` | 409 | Aktiven Job kann nicht gelöscht werden |

---

## 10. Typische Workflows

### Workflow A: Foto-Bibliothek indizieren

```
Für jedes Foto:
  1.  POST /faces/embed
      → face_id pro Gesicht speichern (clientseitig)
      → Centroid = embedding[0] für Einzelgesichter

  Alle Fotos verarbeitet:
  2.  POST /faces/cluster-group  (Mode B: face_ids)
      → session_id speichern

  Später:
  3.  GET /sessions/{session_id}
      → Gruppen abrufen und UI rendern
```

### Workflow B: Realtime-Upload mit Webhook

```
1.  POST /faces/embed/async
    → job_id sofort zurück (HTTP 202)
    → Nutzer bekommt sofortige Bestätigung

2.  Server empfängt Callback:
    POST https://meinserver.de/webhook
    Authorization: Basic ...
    → result.faces[].face_id cachen

3.  GET /jobs/{job_id}  (optional, Polling als Fallback)
```

### Workflow C: Barcode-Scanner Pipeline

```
1.  POST /barcodes/detect
    → barcodes[] auswerten

    Falls barcodes_found == 0:
    → Bild vorverarbeiten (erhöhter Kontrast, Zuschnitt)
    → POST /barcodes/detect-base64  (bereits im Speicher als b64)

2.  Ergebnis an ERP/WMS weitergeben
```

### Workflow D: Inkrementelles Grouping

```
Täglicher Batch neuer Fotos:
  1.  POST /faces/embed für neue Fotos
      → neue face_ids

  2.  GET /faces/cache  (alle aktiven IDs)
      → bestehende + neue face_ids kombinieren

  3.  POST /faces/cluster-group
      → altes Session-Ergebnis wird überschrieben
      → neue session_id persistieren

  4.  DELETE /sessions/{alte_session_id}
      → alten Stand bereinigen
```
