from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from enum import Enum


class ChallengeType(str, Enum):
    look_left = "look_left"
    look_right = "look_right"
    look_up = "look_up"
    look_down = "look_down"
    smile = "smile"


class ChallengeResponse(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    challenge_id: str = Field(..., min_length=1)
    challenge_type: ChallengeType
    instruction: str = Field(..., min_length=1)
    expires_in_seconds: int = Field(120, ge=1)


class PoseAngles(BaseModel):
    yaw: float = Field(..., description="Left/right rotation in degrees")
    pitch: float = Field(..., description="Up/down rotation in degrees")
    roll: float = Field(..., description="Tilt rotation in degrees")


class VerificationDetails(BaseModel):
    smile_score: float = Field(..., ge=0.0, le=1.0)
    challenge_measured: float
    challenge_required_range: list[float]
    liveness_real_prob: float = Field(..., ge=0.0, le=1.0)
    liveness_spoof_prob: float = Field(..., ge=0.0, le=1.0)
    match_reason: str


class VerificationResult(BaseModel):
    verified: bool
    challenge_type: str
    liveness_score: float = Field(..., ge=0.0, le=1.0)
    liveness_passed: bool
    pose_matched: bool
    pose_angles: Optional[PoseAngles] = None
    face_detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    message: str
    details: Optional[VerificationDetails] = None


class ModelStatus(BaseModel):
    face_landmarker: bool
    liveness: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    models: ModelStatus
    uptime_seconds: float
