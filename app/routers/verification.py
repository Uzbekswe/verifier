"""
Verification API routes.

POST /verify/challenge          → Get a random pose challenge
POST /verify/submit/{challenge} → Submit photo for a specific challenge
POST /verify/submit-auto        → Submit with challenge type in body
"""

import uuid
import random
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

import numpy as np
import cv2

from app.schemas.verification import (
    ChallengeResponse, ChallengeType, VerificationResult, PoseAngles
)
from app.services.face_analysis import face_analysis_service
from app.services.liveness import liveness_service
from app.services.challenge_matcher import challenge_matcher, CHALLENGE_CONFIG
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/verify", tags=["Verification"])


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes to BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it's a valid JPEG/PNG.")
    return img


@router.post("/challenge", response_model=ChallengeResponse, summary="Get a pose challenge")
async def get_challenge(challenge_type: str = None):
    """
    Returns a random pose challenge (or a specific one if requested).
    In a real app you'd store challenge_id in Redis with a TTL.
    """
    if challenge_type and challenge_type not in CHALLENGE_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid challenge. Valid options: {list(CHALLENGE_CONFIG.keys())}"
        )

    selected = challenge_type or random.choice(settings.CHALLENGES)
    instruction = challenge_matcher.get_instruction(selected)

    return ChallengeResponse(
        challenge_id=str(uuid.uuid4()),
        challenge_type=ChallengeType(selected),
        instruction=instruction,
        expires_in_seconds=120,
    )


@router.post("/submit/{challenge_type}", response_model=VerificationResult, summary="Submit photo for verification")
async def submit_verification(
    challenge_type: str,
    file: UploadFile = File(..., description="Selfie image (JPEG/PNG)"),
):
    """
    Main verification endpoint. Accepts a selfie + challenge type,
    runs the full pipeline: decode → liveness → face analysis → pose match.
    """
    if challenge_type not in CHALLENGE_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown challenge: '{challenge_type}'. Valid: {list(CHALLENGE_CONFIG.keys())}"
        )

    # --- 1. Read + size check ---
    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large. Max {settings.MAX_IMAGE_SIZE_MB}MB.")

    # --- 2. Decode ---
    try:
        image_bgr = decode_image(file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(e)}")

    # --- 3. Face analysis (landmarks + pose) ---
    face_analysis = face_analysis_service.analyze(image_bgr)

    if not face_analysis.face_detected:
        return VerificationResult(
            verified=False,
            challenge_type=challenge_type,
            liveness_score=0.0,
            liveness_passed=False,
            pose_matched=False,
            face_detected=False,
            confidence=0.0,
            message="❌ No face detected. Please ensure your face is clearly visible.",
        )

    # --- 4. Liveness check ---
    liveness_result = liveness_service.check(image_bgr, face_analysis.face_bbox)

    # --- 5. Challenge / pose matching ---
    match_result = challenge_matcher.match(face_analysis, challenge_type)

    # --- 6. Final decision ---
    verified = liveness_result.is_live and match_result.matched
    overall_confidence = (liveness_result.score + match_result.confidence) / 2.0

    if verified:
        message = f"✅ Verified! Challenge '{challenge_type}' passed."
    elif not liveness_result.is_live:
        message = f"❌ Liveness check failed (score={liveness_result.score:.2f}). Ensure you are a real person, not a photo."
    else:
        message = f"❌ Pose not matched. {match_result.reason}"

    pose_angles = PoseAngles(
        yaw=round(face_analysis.yaw, 2),
        pitch=round(face_analysis.pitch, 2),
        roll=round(face_analysis.roll, 2),
    )

    return VerificationResult(
        verified=verified,
        challenge_type=challenge_type,
        liveness_score=round(liveness_result.score, 3),
        liveness_passed=liveness_result.is_live,
        pose_matched=match_result.matched,
        pose_angles=pose_angles,
        face_detected=True,
        confidence=round(overall_confidence, 3),
        message=message,
        details={
            "smile_score": round(face_analysis.smile_score, 3),
            "challenge_measured": round(match_result.measured_value, 2),
            "challenge_required_range": list(match_result.required_range),
            "liveness_real_prob": round(liveness_result.is_real_prob, 3),
            "liveness_spoof_prob": round(liveness_result.is_spoof_prob, 3),
            "match_reason": match_result.reason,
        },
    )
