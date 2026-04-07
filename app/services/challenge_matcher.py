"""
Challenge matching: given pose angles + smile score + challenge type,
decide if the user successfully completed the required gesture.
"""

from dataclasses import dataclass
from typing import Tuple
from app.services.face_analysis import FaceAnalysis
from app.core.config import settings


@dataclass
class ChallengeMatchResult:
    matched: bool
    challenge_type: str
    measured_value: float   # the relevant metric (angle or smile score)
    required_range: Tuple[float, float]
    confidence: float
    reason: str


# Challenge definitions: (axis, direction, threshold)
CHALLENGE_CONFIG = {
    "look_left": {
        "axis": "yaw",
        "direction": "negative",   # yaw < -threshold
        "threshold": settings.YAW_THRESHOLD,
        "instruction": "Turn your head to the LEFT",
        "emoji": "👈",
    },
    "look_right": {
        "axis": "yaw",
        "direction": "positive",   # yaw > +threshold
        "threshold": settings.YAW_THRESHOLD,
        "instruction": "Turn your head to the RIGHT",
        "emoji": "👉",
    },
    "look_up": {
        "axis": "pitch",
        "direction": "negative",
        "threshold": settings.PITCH_THRESHOLD,
        "instruction": "Tilt your head UP",
        "emoji": "☝️",
    },
    "look_down": {
        "axis": "pitch",
        "direction": "positive",
        "threshold": settings.PITCH_THRESHOLD,
        "instruction": "Tilt your head DOWN",
        "emoji": "👇",
    },
    "smile": {
        "axis": "smile",
        "direction": "positive",
        "threshold": 0.35,
        "instruction": "Give us a big SMILE 😄",
        "emoji": "😄",
    },
}


class ChallengeMatcher:

    def match(self, analysis: FaceAnalysis, challenge_type: str) -> ChallengeMatchResult:
        """
        Check whether the face in the image satisfies the given challenge.
        """
        if not analysis.face_detected:
            return ChallengeMatchResult(
                matched=False,
                challenge_type=challenge_type,
                measured_value=0.0,
                required_range=(0.0, 0.0),
                confidence=0.0,
                reason="No face detected in the image",
            )

        if challenge_type not in CHALLENGE_CONFIG:
            return ChallengeMatchResult(
                matched=False,
                challenge_type=challenge_type,
                measured_value=0.0,
                required_range=(0.0, 0.0),
                confidence=0.0,
                reason=f"Unknown challenge type: {challenge_type}",
            )

        config = CHALLENGE_CONFIG[challenge_type]
        axis = config["axis"]
        direction = config["direction"]
        threshold = config["threshold"]
        margin = settings.POSE_MARGIN

        # Get the measured value for this axis
        if axis == "yaw":
            measured = analysis.yaw
        elif axis == "pitch":
            measured = analysis.pitch
        elif axis == "smile":
            measured = analysis.smile_score
        else:
            measured = 0.0

        # Check match condition
        if direction == "positive":
            matched = measured > threshold
            required_range = (threshold, 90.0)
            # Confidence: how far past threshold (normalized 0-1)
            if matched:
                confidence = min(1.0, (measured - threshold) / (threshold + margin))
            else:
                confidence = max(0.0, measured / threshold)
        else:  # negative direction
            matched = measured < -threshold
            required_range = (-90.0, -threshold)
            if matched:
                confidence = min(1.0, (-measured - threshold) / (threshold + margin))
            else:
                confidence = max(0.0, -measured / threshold)

        if matched:
            reason = f"✅ Challenge passed: {axis}={measured:.1f}° (required {'>' if direction=='positive' else '<'} {threshold if direction=='positive' else -threshold})"
        else:
            reason = f"❌ Challenge failed: {axis}={measured:.1f}° (required {'>' if direction=='positive' else '<'} {threshold if direction=='positive' else -threshold:.0f})"

        return ChallengeMatchResult(
            matched=matched,
            challenge_type=challenge_type,
            measured_value=measured,
            required_range=required_range,
            confidence=confidence,
            reason=reason,
        )

    def get_instruction(self, challenge_type: str) -> str:
        config = CHALLENGE_CONFIG.get(challenge_type, {})
        emoji = config.get("emoji", "")
        instruction = config.get("instruction", challenge_type)
        return f"{emoji} {instruction}"


challenge_matcher = ChallengeMatcher()
