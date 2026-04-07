from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    APP_NAME: str = "Photo Verification API"
    VERSION: str = "1.0.0"
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

    # Pose challenge thresholds (degrees)
    YAW_THRESHOLD: float = 15.0       # head turn left/right
    PITCH_THRESHOLD: float = 12.0     # head tilt up/down
    POSE_MARGIN: float = 5.0          # acceptable margin beyond threshold

    # Liveness
    LIVENESS_THRESHOLD: float = 0.6   # min score to be considered live

    # Supported challenges
    CHALLENGES: list = ["look_left", "look_right", "look_up", "look_down", "smile"]

settings = Settings()
