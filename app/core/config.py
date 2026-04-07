from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Photo Verification API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Image limits
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_BYTES: int = 10 * 1024 * 1024
    MAX_IMAGE_DIMENSION: int = 4096

    # Pose challenge thresholds (degrees)
    YAW_THRESHOLD: float = 15.0
    PITCH_THRESHOLD: float = 12.0
    POSE_MARGIN: float = 5.0

    # Liveness
    LIVENESS_THRESHOLD: float = 0.6

    # Supported challenges
    CHALLENGES: list = ["look_left", "look_right", "look_up", "look_down", "smile"]

    # Concurrency
    CV_THREAD_POOL_SIZE: int = 4

    # Rate limiting
    RATE_LIMIT_STORAGE_URI: str = "memory://"
    RATE_LIMIT_GLOBAL: str = "200/minute"
    RATE_LIMIT_SUBMIT: str = "10/minute"
    RATE_LIMIT_CHALLENGE: str = "30/minute"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" | "text"

    # CORS
    CORS_ORIGINS: list = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
