"""
Verification API routes.

POST /verify/challenge          → Get a random pose challenge
POST /verify/submit/{challenge} → Submit photo for a specific challenge
"""

import logging
import random
import time
import uuid

import io
import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.dependencies import run_in_cv_executor, validated_image_bytes
from app.core.exceptions import ImageDecodeError, UnknownChallengeError
from app.metrics.prometheus import (
    cv_processing_seconds,
    liveness_score_histogram,
    verification_attempts,
)
from app.middleware.rate_limit import limiter
from app.schemas.verification import (
    ChallengeResponse,
    ChallengeType,
    PoseAngles,
    VerificationDetails,
    VerificationResult,
)
from app.services.challenge_matcher import CHALLENGE_CONFIG, challenge_matcher
from app.services.face_analysis import face_analysis_service
from app.services.liveness import liveness_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/verify", tags=["Verification v1"])


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes to BGR numpy array, applying EXIF rotation (runs in CV thread pool).

    cv2.imdecode ignores EXIF orientation metadata, which causes face detection
    to fail on mobile selfies (the image arrives sideways/upside-down to MediaPipe).
    PIL's exif_transpose() corrects the orientation before handing off to OpenCV.
    """
    pil_img = Image.open(io.BytesIO(file_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if img is None:
        raise ImageDecodeError()
    return img


@router.post("/challenge", response_model=ChallengeResponse, summary="Get a pose challenge")
@limiter.limit(settings.RATE_LIMIT_CHALLENGE)
async def get_challenge(request: Request, challenge_type: ChallengeType | None = None):
    """
    Returns a random pose challenge (or a specific one if `challenge_type` is provided).
    In production, persist `challenge_id` in Redis with a TTL to prevent replay attacks.
    """
    selected = challenge_type.value if challenge_type else random.choice(settings.CHALLENGES)
    instruction = challenge_matcher.get_instruction(selected)

    return ChallengeResponse(
        challenge_id=str(uuid.uuid4()),
        challenge_type=ChallengeType(selected),
        instruction=instruction,
        expires_in_seconds=120,
    )


@router.post(
    "/submit/{challenge_type}",
    response_model=VerificationResult,
    summary="Submit photo for verification",
)
@limiter.limit(settings.RATE_LIMIT_SUBMIT)
async def submit_verification(
    request: Request,
    challenge_type: ChallengeType,
    file_bytes: bytes = Depends(validated_image_bytes),
):
    """
    Main verification endpoint. Runs the full pipeline:
    decode → face analysis → liveness → pose match.

    All blocking CV operations run in a dedicated thread pool so the event loop
    is never blocked.
    """
    challenge_str = challenge_type.value
    total_start = time.perf_counter()

    # --- 1. Decode image (CPU-bound → thread pool) ---
    with cv_processing_seconds.labels(stage="decode").time():
        image_bgr = await run_in_cv_executor(_decode_image, file_bytes)

    # --- 2. Face analysis: landmarks + pose (CPU-bound → thread pool) ---
    with cv_processing_seconds.labels(stage="face_analysis").time():
        face_analysis = await run_in_cv_executor(face_analysis_service.analyze, image_bgr)

    if not face_analysis.face_detected:
        verification_attempts.labels(challenge_type=challenge_str, result="no_face").inc()
        return VerificationResult(
            verified=False,
            challenge_type=challenge_str,
            liveness_score=0.0,
            liveness_passed=False,
            pose_matched=False,
            face_detected=False,
            confidence=0.0,
            message="❌ No face detected. Please ensure your face is clearly visible.",
        )

    # --- 3. Liveness check (CPU-bound → thread pool) ---
    with cv_processing_seconds.labels(stage="liveness").time():
        liveness_result = await run_in_cv_executor(
            liveness_service.check, image_bgr, face_analysis.face_bbox
        )

    liveness_score_histogram.observe(liveness_result.score)

    # --- 4. Challenge / pose matching ---
    match_result = challenge_matcher.match(face_analysis, challenge_str)

    # --- 5. Final decision ---
    verified = liveness_result.is_live and match_result.matched
    overall_confidence = (liveness_result.score + match_result.confidence) / 2.0

    total_ms = (time.perf_counter() - total_start) * 1000
    cv_processing_seconds.labels(stage="total").observe(total_ms / 1000)

    if verified:
        result_label = "verified"
        message = f"✅ Verified! Challenge '{challenge_str}' passed."
    elif not liveness_result.is_live:
        result_label = "liveness_failed"
        message = f"❌ Liveness check failed (score={liveness_result.score:.2f}). Please use a real selfie."
    else:
        result_label = "pose_failed"
        message = f"❌ Pose not matched. {match_result.reason}"

    verification_attempts.labels(challenge_type=challenge_str, result=result_label).inc()

    logger.info(
        "verification complete",
        extra={
            "challenge_type": challenge_str,
            "verified": verified,
            "liveness_score": round(liveness_result.score, 3),
            "yaw": round(face_analysis.yaw, 2),
            "pitch": round(face_analysis.pitch, 2),
            "duration_ms": round(total_ms, 2),
        },
    )

    return VerificationResult(
        verified=verified,
        challenge_type=challenge_str,
        liveness_score=round(liveness_result.score, 3),
        liveness_passed=liveness_result.is_live,
        pose_matched=match_result.matched,
        pose_angles=PoseAngles(
            yaw=round(face_analysis.yaw, 2),
            pitch=round(face_analysis.pitch, 2),
            roll=round(face_analysis.roll, 2),
        ),
        face_detected=True,
        confidence=round(overall_confidence, 3),
        message=message,
        details=VerificationDetails(
            smile_score=round(face_analysis.smile_score, 3),
            challenge_measured=round(match_result.measured_value, 2),
            challenge_required_range=list(match_result.required_range),
            liveness_real_prob=round(liveness_result.is_real_prob, 3),
            liveness_spoof_prob=round(liveness_result.is_spoof_prob, 3),
            match_reason=match_result.reason,
        ),
    )
