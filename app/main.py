"""
Photo Verification API — FastAPI app factory.
"""

import logging
import os
import time

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging_config import configure_logging
from app.middleware.correlation_id import CorrelationIdMiddleware, get_request_id
from app.middleware.rate_limit import limiter
from app.middleware.timing import TimingMiddleware
from app.metrics.prometheus import setup_metrics
from app.routers import verification
from app.routers import health as health_router
from app.schemas.errors import ErrorResponse, ErrorDetail

# Silence TensorFlow noise before any TF import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all CV models at startup; drain thread pool on shutdown."""
    from app.services.face_analysis import face_analysis_service
    from app.services.liveness import liveness_service
    from app.core.dependencies import cv_executor

    logger.info("starting up", extra={"version": settings.VERSION})
    face_analysis_service.load()
    liveness_service.load()
    logger.info("all models loaded — API ready")

    yield

    logger.info("shutting down — draining CV thread pool")
    cv_executor.shutdown(wait=True, cancel_futures=False)
    logger.info("shutdown complete")


def _make_error_response(status_code: int, error_code: str, message: str, context: dict = {}) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=ErrorDetail(
                error_code=error_code,
                message=message,
                request_id=get_request_id(),
                context=context,
            )
        ).model_dump(),
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="""
## Photo Verification API

Simulates the pose-challenge verification system used in apps like Bumble/Tinder.

### Pipeline
1. **Liveness detection** — MiniFASNet (Silent-Face-Anti-Spoofing) via DeepFace
2. **Face landmark extraction** — MediaPipe FaceLandmarker (478 3D points)
3. **Head pose estimation** — PnP solver (yaw/pitch/roll in degrees)
4. **Challenge matching** — threshold-based pose verification

### No training needed — all models are pretrained.
        """,
        contact={"name": "API Support"},
        openapi_tags=[
            {"name": "Verification v1", "description": "Pose challenge verification endpoints"},
            {"name": "Health", "description": "Service health and readiness"},
        ],
        docs_url="/docs" if settings.DEBUG else "/docs",  # keep docs accessible for demo
        redoc_url="/redoc" if settings.DEBUG else "/redoc",
        lifespan=lifespan,
    )

    # --- Rate limiter state ---
    app.state.limiter = limiter

    # --- Middleware (last added = outermost on request) ---
    app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(TimingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # --- Exception handlers ---
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        return _make_error_response(exc.status_code, exc.error_code, exc.message, exc.context)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return _make_error_response(
            422,
            "VALIDATION_ERROR",
            "Request validation failed.",
            context={"errors": exc.errors()},
        )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return _make_error_response(429, "RATE_LIMIT_EXCEEDED", "Too many requests. Please slow down.")

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        logger.exception("unhandled exception", extra={"path": request.url.path})
        return _make_error_response(500, "INTERNAL_ERROR", "An unexpected error occurred.")

    # --- Routers ---
    app.include_router(health_router.router, prefix="/api/v1")
    app.include_router(verification.router, prefix="/api/v1")

    # --- Prometheus metrics ---
    setup_metrics(app)

    return app


app = create_app()
