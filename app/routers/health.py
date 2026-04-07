"""
Health check endpoint.
"""
import time
import logging

from fastapi import APIRouter, Depends

from app.schemas.verification import HealthResponse, ModelStatus
from app.core.config import settings
from app.services.face_analysis import face_analysis_service
from app.services.liveness import liveness_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health():
    face_ready = face_analysis_service._loaded
    liveness_ready = liveness_service._loaded
    status = "ok" if (face_ready and liveness_ready) else "degraded"
    return HealthResponse(
        status=status,
        version=settings.VERSION,
        models=ModelStatus(
            face_landmarker=face_ready,
            liveness=liveness_ready,
        ),
        uptime_seconds=round(time.time() - _start_time, 1),
    )
