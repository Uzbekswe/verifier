---
title: FaceCheck
emoji: 🤳
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Photo Verification API

A production-grade **selfie verification API** that mimics the pose-challenge system used by apps like Bumble and Tinder. It detects whether you're a real person (liveness check) and whether you correctly completed a pose challenge (look left, look right, smile, etc.).

Ships with a **browser UI** for testing and a **Docker setup** for one-command local runs and free cloud deployment.

Built with FastAPI, MediaPipe, and DeepFace. No custom model training required — everything uses pretrained weights.

---

## What It Does

The API runs a 3-stage pipeline on every submitted photo:

1. **Liveness Detection** — Uses DeepFace (MiniFASNet / Silent-Face-Anti-Spoofing) to detect whether the image is a real live person or a printed photo / screen replay.
2. **Face Landmark Extraction** — Uses MediaPipe FaceLandmarker to extract 478 3D facial landmarks.
3. **Head Pose Estimation + Challenge Matching** — Solves a PnP (Perspective-n-Point) problem using OpenCV to compute yaw, pitch, and roll in degrees, then checks if the pose matches the requested challenge.

---

## Project Structure

```
photo-app/
├── app/
│   ├── main.py                      # FastAPI app factory, lifespan, middleware wiring
│   ├── core/
│   │   ├── config.py                # All settings (env-var backed via pydantic-settings)
│   │   ├── dependencies.py          # CV thread pool, validated image upload dependency
│   │   ├── exceptions.py            # Domain exception hierarchy (ImageDecodeError, etc.)
│   │   └── logging_config.py        # JSON / text structured logging setup
│   ├── middleware/
│   │   ├── correlation_id.py        # X-Request-ID propagation via ContextVar
│   │   ├── timing.py                # Per-request structured access log
│   │   └── rate_limit.py            # slowapi limiter singleton
│   ├── metrics/
│   │   └── prometheus.py            # Prometheus instrumentation + domain metrics
│   ├── routers/
│   │   ├── health.py                # GET /api/v1/health
│   │   └── verification.py          # POST /api/v1/verify/challenge & /submit/{type}
│   ├── schemas/
│   │   ├── verification.py          # Pydantic request/response models
│   │   └── errors.py                # Typed error response schema
│   ├── services/
│   │   ├── face_analysis.py         # MediaPipe FaceLandmarker + PnP head pose
│   │   ├── liveness.py              # DeepFace anti-spoofing
│   │   └── challenge_matcher.py     # Pose threshold logic per challenge type
│   └── static/
│       ├── index.html               # Browser UI
│       ├── style.css                # Dark card-based styling
│       └── app.js                   # Camera capture + API calls + result display
├── models/
│   └── face_landmarker.task         # MediaPipe pretrained model (~3.7MB)
├── tests/
│   ├── conftest.py                  # Pytest fixtures with mock services
│   └── test_pipeline.py             # Integration tests
├── Dockerfile                       # Multi-stage build for containerisation
├── docker-compose.yml               # Local dev stack with health checks + volumes
├── .dockerignore                    # Excludes venv/git/tests from build context
├── .env.example                     # Template for environment variable overrides
├── requirements.txt
└── README.md
```

---

## API Endpoints

All endpoints are versioned under `/api/v1`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Service health, model readiness, uptime |
| POST | `/api/v1/verify/challenge` | Get a random (or specific) pose challenge |
| POST | `/api/v1/verify/submit/{challenge_type}` | Submit a selfie for verification |
| GET | `/metrics` | Prometheus metrics (HTTP + domain counters) |
| GET | `/docs` | Swagger UI — interactive API explorer |

### Challenge Types

| Challenge | What To Do |
|-----------|-----------|
| `look_left` | Turn head to the left (yaw < -15°) |
| `look_right` | Turn head to the right (yaw > +15°) |
| `look_up` | Tilt head up (pitch < -12°) |
| `look_down` | Tilt head down (pitch > +12°) |
| `smile` | Show a big smile (smile score > 0.35) |

---

## Running with Docker (recommended)

Docker is the easiest way to run the app — no Python install, no dependency conflicts.

### Requirements
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Option A — Docker Compose (recommended)

```bash
# Clone the repo
git clone https://github.com/Uzbekswe/verifier.git
cd verifier

# Build and start (first build takes ~5–10 min to install TensorFlow)
docker compose up
```

The app will be available at **http://localhost:8000**

- The UI opens automatically at the root URL
- API docs at **http://localhost:8000/docs**
- Health check at **http://localhost:8000/api/v1/health**

Check container health:
```bash
docker compose ps      # shows "healthy" after ~60s (model load time)
docker compose logs    # stream structured logs
docker compose down    # stop everything
```

### Option B — Plain Docker

```bash
# Build the image
docker build -t photo-verifier .

# Run the container
docker run -p 8000:7860 photo-verifier
```

With environment variable overrides:
```bash
docker run \
  -p 8000:7860 \
  -e LOG_FORMAT=text \
  -e LIVENESS_THRESHOLD=0.7 \
  -e CV_THREAD_POOL_SIZE=4 \
  photo-verifier
```

With a volume to persist DeepFace model weights across runs:
```bash
docker run \
  -p 8000:7860 \
  -v deepface-cache:/root/.deepface \
  photo-verifier
```

### How the Dockerfile works

The image uses a **multi-stage build**:

```
Stage 1 (builder)   → installs all Python deps into /install/deps
Stage 2 (final)     → copies only the installed packages + app code
                       into a clean python:3.12-slim image
```

`requirements.txt` is copied before the application code so the expensive `pip install` layer is **cached** — rebuilds after code-only changes take seconds, not minutes.

The `PORT` environment variable controls which port uvicorn listens on (default `7860`). Docker Compose maps host port `8000` → container port `7860`.

---

## Running Locally (without Docker)

### Requirements
- Python 3.12 (TensorFlow 2.x does not support Python 3.13+ yet)

### Install Python 3.12
```bash
brew install python@3.12
```

### Create virtual environment and install dependencies
```bash
cd photo-app
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Note: `pip install` pulls in `tensorflow`, `keras`, and other heavy ML dependencies through `deepface`. Total install is ~1.5GB. This is normal.

### Download the face landmark model
The MediaPipe model is already included at `models/face_landmarker.task`. If you ever need to re-download it:
```bash
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

### Start the server
```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

The server starts on **http://localhost:8000**

---

## Browser UI

The app ships with a browser UI served at the root URL (`/`).

**Flow:**
1. **Get Challenge** — click the button, receive a random pose instruction
2. **Selfie** — the webcam opens; position your face and follow the instruction, then capture
3. **Result** — the image is submitted to the API and the full result is displayed: verified ✅/❌, liveness score, pose angles, confidence, and per-stage details

The UI works on desktop and mobile (front-facing camera). No frameworks — plain HTML/CSS/JS, served directly by FastAPI's `StaticFiles`.

---

## Testing

### Swagger UI (recommended)
Open **http://localhost:8000/docs** — interactive endpoint explorer with file upload support.

### Health check
```bash
curl http://localhost:8000/api/v1/health
```

### Full verification flow (terminal)

**Step 1 — Get a challenge:**
```bash
curl -s -X POST http://localhost:8000/api/v1/verify/challenge | python3 -m json.tool
```

Example response:
```json
{
  "challenge_id": "a3f2...",
  "challenge_type": "look_left",
  "instruction": "👈 Turn your head to the LEFT",
  "expires_in_seconds": 120
}
```

**Step 2 — Submit a selfie:**
```bash
curl -s -X POST \
  "http://localhost:8000/api/v1/verify/submit/look_left" \
  -F "file=@/path/to/your/photo.jpg" | python3 -m json.tool
```

Example response:
```json
{
  "verified": true,
  "challenge_type": "look_left",
  "liveness_score": 0.91,
  "liveness_passed": true,
  "pose_matched": true,
  "pose_angles": { "yaw": -22.4, "pitch": 1.2, "roll": 0.8 },
  "face_detected": true,
  "confidence": 0.87,
  "message": "✅ Verified! Challenge 'look_left' passed.",
  "details": {
    "smile_score": 0.12,
    "challenge_measured": -22.4,
    "challenge_required_range": [-90.0, -15.0],
    "liveness_real_prob": 0.91,
    "liveness_spoof_prob": 0.09,
    "match_reason": "✅ Challenge passed: yaw=-22.4° (required < -15)"
  }
}
```

### Run pipeline tests
```bash
source venv/bin/activate
python3 tests/test_pipeline.py
```

---

## Free Cloud Deployment

The Dockerfile is pre-configured for **Hugging Face Spaces** — the best free option for ML apps (no RAM limits, no credit card required).

1. Create an account at [huggingface.co](https://huggingface.co)
2. New Space → SDK: **Docker** → Hardware: **CPU Basic (free)**
3. Connect the `Uzbekswe/verifier` GitHub repo — auto-deploys on every push
4. Hugging Face sets `PORT=7860` automatically — the Dockerfile already reads it

Your public URL: `https://huggingface.co/spaces/Uzbekswe/verifier`

**Alternative:** [Railway.app](https://railway.app) — connect GitHub repo, free $5/month credit, auto-detects the Dockerfile.

---

## How It Was Built

### Problem
Build a backend API that verifies a user is a real, live human and can perform an on-screen gesture — the same kind of challenge used in dating apps to prevent fake profile photos.

### Stack choices

**FastAPI** was chosen for its async support, automatic OpenAPI docs, Pydantic validation, and built-in `StaticFiles` support — ideal for a photo-processing API that also serves a browser UI.

**MediaPipe FaceLandmarker** (Tasks API, v0.10.33) extracts 478 3D facial landmarks per frame. The newer Tasks API was used instead of the legacy `mp.solutions` because `mp.solutions` was removed in mediapipe >= 0.10.14. The pretrained model runs entirely on-device.

**Head Pose via PnP Solver** — Maps 6 stable MediaPipe landmarks (nose tip, chin, eye corners, mouth corners) to a known 3D canonical face model, then calls OpenCV's `solvePnP` to solve for the rotation vector. Decomposed into yaw, pitch, and roll using Rodrigues + `decomposeProjectionMatrix`. No training needed.

**DeepFace anti-spoofing** wraps MinivisionAI's Silent-Face-Anti-Spoofing (MiniFASNet). It classifies each face as real or spoof with a confidence score. Weights download automatically from GitHub on first use (~4MB). A texture-based Laplacian variance fallback is included in case DeepFace fails.

**Python 3.12** is required because TensorFlow 2.x does not yet have wheels for Python 3.13+.

### Key engineering decisions
- Models load once at startup via FastAPI's `lifespan` context — not on the first request — so the first API call is fast.
- MediaPipe, OpenCV, and DeepFace run in a dedicated `ThreadPoolExecutor` via `run_in_executor` — the asyncio event loop is never blocked, enabling real concurrency under load.
- Liveness crops the face region with 15% padding before running anti-spoof, reducing background noise.
- The multi-stage Dockerfile copies `requirements.txt` before application code so the 1.5GB dependency layer is cached and code-only rebuilds take seconds.
- Challenge thresholds, rate limits, thread pool size, and all other tunables are configurable via environment variables without touching source code.

---

## Configuration

All settings are backed by environment variables and can be overridden via a `.env` file (copy `.env.example` to `.env`):

| Setting | Default | Meaning |
|---------|---------|---------|
| `YAW_THRESHOLD` | 15.0° | Degrees of head turn needed for look_left / look_right |
| `PITCH_THRESHOLD` | 12.0° | Degrees of tilt needed for look_up / look_down |
| `LIVENESS_THRESHOLD` | 0.6 | Minimum anti-spoof score to pass liveness |
| `MAX_IMAGE_SIZE_MB` | 10 | Max upload size |
| `MAX_IMAGE_DIMENSION` | 4096 | Max image width or height in pixels |
| `CV_THREAD_POOL_SIZE` | 4 | Workers for blocking CV operations |
| `RATE_LIMIT_GLOBAL` | 200/minute | Global rate limit per IP |
| `RATE_LIMIT_SUBMIT` | 10/minute | Rate limit on `/verify/submit` |
| `RATE_LIMIT_CHALLENGE` | 30/minute | Rate limit on `/verify/challenge` |
| `LOG_LEVEL` | INFO | Python logging level |
| `LOG_FORMAT` | json | `json` for production, `text` for local dev |
| `CORS_ORIGINS` | ["*"] | Allowed CORS origins (restrict in production) |
| `PORT` | 7860 | Port uvicorn listens on inside the container |

---

## Production Features

- **Async CV execution** — MediaPipe, OpenCV, and DeepFace run in a dedicated `ThreadPoolExecutor`, keeping the event loop unblocked under concurrent load.
- **Structured JSON logging** — Every log line is JSON with `request_id`, `logger`, `level`, and timing fields. Correlation IDs propagate from `X-Request-ID` headers through all logs for a request.
- **Rate limiting** — `slowapi` enforces configurable per-IP limits globally and per-endpoint. Swap to Redis (`RATE_LIMIT_STORAGE_URI=redis://...`) for multi-instance deployments.
- **Prometheus metrics** at `/metrics` — auto-instrumented HTTP metrics plus domain counters: `verification_attempts_total`, `cv_processing_seconds`, `liveness_score_distribution`.
- **Typed error responses** — All errors return `{ "error": { "error_code": "...", "message": "...", "request_id": "...", "context": {} } }` with machine-readable codes (`IMAGE_TOO_LARGE`, `INVALID_MIME_TYPE`, `RATE_LIMIT_EXCEEDED`, etc.).
- **Input validation** — MIME type (JPEG/PNG/WebP only), file size, and pixel dimension checks run before any CV processing.
- **API versioning** — All endpoints live under `/api/v1`.
- **Docker** — Multi-stage Dockerfile with layer caching, named volumes for model persistence, and Compose health checks.
