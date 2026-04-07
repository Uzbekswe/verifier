# Photo Verification API

A production-style **selfie verification API** that mimics the pose-challenge system used by apps like Bumble and Tinder. It detects whether you're a real person (liveness check) and whether you correctly completed a pose challenge (look left, look right, smile, etc.).

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
│   ├── main.py                  # FastAPI app factory, startup/shutdown lifecycle
│   ├── core/
│   │   └── config.py            # Settings (thresholds, challenge list, limits)
│   ├── routers/
│   │   └── verification.py      # POST /verify/challenge, POST /verify/submit/{type}
│   ├── schemas/
│   │   └── verification.py      # Pydantic request/response models
│   └── services/
│       ├── face_analysis.py     # MediaPipe FaceLandmarker + PnP head pose
│       ├── liveness.py          # DeepFace anti-spoofing
│       └── challenge_matcher.py # Pose threshold logic per challenge type
├── models/
│   └── face_landmarker.task     # MediaPipe pretrained face landmark model (~3.7MB)
├── tests/
│   └── test_pipeline.py         # Integration tests
├── requirements.txt
├── .gitignore
└── README.md
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Check server status and loaded models |
| POST | `/verify/challenge` | Get a random pose challenge |
| POST | `/verify/submit/{challenge_type}` | Submit a selfie for verification |

### Challenge Types

| Challenge | What To Do |
|-----------|-----------|
| `look_left` | Turn head to the left (yaw < -15°) |
| `look_right` | Turn head to the right (yaw > +15°) |
| `look_up` | Tilt head up (pitch < -12°) |
| `look_down` | Tilt head down (pitch > +12°) |
| `smile` | Show a big smile (smile score > 0.35) |

---

## Setup & Run

### Requirements
- macOS (Apple Silicon or Intel)
- Python 3.12 (important — tensorflow does not support Python 3.14 yet)

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

> Note: `pip install` will also pull in `tensorflow`, `keras`, and other heavy dependencies through `deepface`. Total install is ~1.5GB. This is normal.

### Download the face landmark model
The MediaPipe face landmarker model is already included at `models/face_landmarker.task`. If you ever need to re-download it:
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

## Testing

### Interactive API docs (recommended)
Open **http://localhost:8000/docs** in your browser. Swagger UI lets you call every endpoint, upload photos, and inspect responses without writing any code.

### Health check
```bash
curl http://localhost:8000/health
```

### Full verification flow (terminal)

**Step 1 — Get a challenge:**
```bash
curl -s -X POST http://localhost:8000/verify/challenge | python3 -m json.tool
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
  "http://localhost:8000/verify/submit/look_left" \
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
  "message": "✅ Verified! Challenge 'look_left' passed."
}
```

---

## How It Was Built

### Problem
Build a backend API that verifies a user is a real, live human and can perform an on-screen gesture — the same kind of challenge used in dating apps to prevent fake profile photos.

### Stack choices

**FastAPI** was chosen for its async support, automatic OpenAPI docs, and Pydantic validation — ideal for a photo-processing API where requests involve file uploads and structured JSON responses.

**MediaPipe FaceLandmarker** (Tasks API, v0.10.33) extracts 478 3D facial landmarks per frame. The newer Tasks API was used instead of the legacy `mp.solutions` because `mp.solutions` was removed in mediapipe >= 0.10.14. The pretrained `face_landmarker.task` model runs entirely on-device with no external API calls.

**Head Pose via PnP Solver** — Rather than training a pose estimation model, the implementation maps 6 stable MediaPipe landmarks (nose tip, chin, eye corners, mouth corners) to a known 3D canonical face model, then calls OpenCV's `solvePnP` to solve for the rotation vector. This is decomposed into yaw, pitch, and roll using Rodrigues + `decomposeProjectionMatrix`. No training data needed.

**DeepFace anti-spoofing** wraps MinivisionAI's Silent-Face-Anti-Spoofing (MiniFASNet). It classifies each face as real or spoof with a confidence score. Model weights download automatically from GitHub on first use (~4MB). A texture-based Laplacian variance fallback is included in case DeepFace fails.

**Python 3.12** is required because tensorflow 2.21 (a deepface dependency) does not yet have wheels for Python 3.14. The venv is built against `/opt/homebrew/bin/python3.12`.

### Key engineering decisions
- Models load once at startup via FastAPI's `lifespan` context — not on the first request — so the first API call is fast.
- MediaPipe and DeepFace run synchronously inside async route handlers (acceptable for single-user local use; would need `run_in_executor` for production concurrency).
- Liveness crops the face region with 15% padding before running anti-spoof, reducing background noise.
- Challenge thresholds (yaw, pitch, smile) are configurable in `app/core/config.py` without touching business logic.

---

## Configuration

Edit `app/core/config.py` to tune:

| Setting | Default | Meaning |
|---------|---------|---------|
| `YAW_THRESHOLD` | 15.0° | Degrees of head turn needed for look_left / look_right |
| `PITCH_THRESHOLD` | 12.0° | Degrees of tilt needed for look_up / look_down |
| `LIVENESS_THRESHOLD` | 0.6 | Minimum anti-spoof score to pass |
| `MAX_IMAGE_SIZE_MB` | 10 | Max upload size |
