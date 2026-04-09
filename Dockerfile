FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# NOTE: libgl1-mesa-glx was removed in Debian Trixie (python:3.11-slim base).
#       libgl1 is the correct replacement and works on Bullseye/Bookworm/Trixie.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libzbar0 \
    libzbar-dev \
    zbar-tools \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download DeepFace model weights & detector ────────────────────────────
# Downloads at build time so containers start instantly without internet access.
# ArcFace  = dev / high / gpu profiles  (~300 MB)
# VGG-Face = medium profile             (~500 MB)
# RetinaFace detector weights are fetched by running a dummy inference.
# All steps fail gracefully if the build host has no internet.
RUN python - <<'EOF'
import sys, numpy as np, cv2, tempfile, os
from deepface import DeepFace

# ── Recognition models ────────────────────────────────────────────────────────
for model in ["ArcFace", "VGG-Face"]:
    try:
        DeepFace.build_model(model)
        print(f"[prewarm] {model} downloaded OK", flush=True)
    except Exception as e:
        print(f"[prewarm] {model} skipped: {e}", flush=True)

# ── RetinaFace detector weights ───────────────────────────────────────────────
# Triggered by a dummy represent() call; the weights are cached in
# /root/.deepface/weights and persisted via the 'models' Docker volume.
try:
    img = np.ones((80, 80, 3), dtype=np.uint8) * 200
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img)
        tmp = f.name
    DeepFace.represent(
        tmp,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=False,
    )
    os.unlink(tmp)
    print("[prewarm] retinaface detector downloaded OK", flush=True)
except Exception as e:
    print(f"[prewarm] retinaface skipped: {e}", flush=True)
EOF

# ── App source ────────────────────────────────────────────────────────────────
COPY . .

# ── Runtime directories ───────────────────────────────────────────────────────
RUN mkdir -p /app/uploads /app/data

EXPOSE 8000

# Shell form so ${WORKERS} is expanded from the container environment.
# docker-compose passes WORKERS from the env-file; default = 2 if not set.
CMD uvicorn main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-2}
