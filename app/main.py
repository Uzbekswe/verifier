"""
Photo Verification API — FastAPI app factory.
Models are loaded at startup via lifespan (not at import time).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import verification

# Silence TensorFlow noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all CV models at startup so first request isn't slow."""
    logger.info("🚀 Starting Photo Verification API — loading models...")

    from app.services.face_analysis import face_analysis_service
    from app.services.liveness import liveness_service

    face_analysis_service.load()
    liveness_service.load()

    logger.info("✅ All models loaded. API ready.")
    yield
    logger.info("🛑 Shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="""
## Photo Verification API

Simulates the pose-challenge verification system used in apps like Bumble/Tinder.

### Pipeline
1. **Liveness detection** — MiniFASNet (Silent-Face-Anti-Spoofing) via DeepFace
2. **Face landmark extraction** — MediaPipe FaceMesh (468 3D points)
3. **Head pose estimation** — PnP solver (yaw/pitch/roll in degrees)
4. **Challenge matching** — threshold-based pose verification

### No training needed — all models are pretrained.
        """,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(verification.router)

    @app.get("/health", tags=["Health"])
    async def health():
        from app.services.face_analysis import face_analysis_service
        from app.services.liveness import liveness_service
        return {
            "status": "ok",
            "models": {
                "face_mesh": face_analysis_service._loaded,
                "liveness": liveness_service._loaded,
            },
            "challenges": settings.CHALLENGES,
        }

    return app


app = create_app()
