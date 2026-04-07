"""
Prometheus metrics setup.
Auto-instruments all FastAPI routes and exposes domain-specific counters/histograms.
"""
import logging

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# --- Domain metrics ---

verification_attempts = Counter(
    "verification_attempts_total",
    "Total number of verification attempts",
    ["challenge_type", "result"],  # result: verified | liveness_failed | pose_failed | no_face
)

cv_processing_seconds = Histogram(
    "cv_processing_seconds",
    "Time spent in CV pipeline stages",
    ["stage"],  # decode | face_analysis | liveness | total
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

liveness_score_histogram = Histogram(
    "liveness_score_distribution",
    "Distribution of liveness anti-spoof scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def setup_metrics(app):
    """Attach prometheus-fastapi-instrumentator to the app."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator(
            should_group_status_codes=False,
            should_ignore_untemplated=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/metrics", "/api/v1/health"],
        ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

        logger.info("Prometheus metrics enabled at /metrics")
    except ImportError:
        logger.warning("prometheus-fastapi-instrumentator not installed — metrics disabled")
