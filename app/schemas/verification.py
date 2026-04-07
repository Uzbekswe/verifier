from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class ChallengeType(str, Enum):
    look_left = "look_left"
    look_right = "look_right"
    look_up = "look_up"
    look_down = "look_down"
    smile = "smile"


class ChallengeResponse(BaseModel):
    challenge_id: str
    challenge_type: ChallengeType
    instruction: str
    expires_in_seconds: int = 120


class PoseAngles(BaseModel):
    yaw: float = Field(..., description="Left/right rotation in degrees")
    pitch: float = Field(..., description="Up/down rotation in degrees")
    roll: float = Field(..., description="Tilt rotation in degrees")


class VerificationResult(BaseModel):
    verified: bool
    challenge_type: str
    liveness_score: float
    liveness_passed: bool
    pose_matched: bool
    pose_angles: Optional[PoseAngles] = None
    face_detected: bool
    confidence: float
    message: str
    details: dict = {}


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
