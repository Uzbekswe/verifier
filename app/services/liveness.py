"""
Liveness (anti-spoofing) detection.
Uses DeepFace's built-in anti_spoof which wraps MinivisionAI's 
Silent-Face-Anti-Spoofing (MiniFASNet) — pretrained, no training needed.
Model weights auto-download from GitHub on first run (~4MB).
"""

import cv2
import numpy as np
import logging
from dataclasses import dataclass
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class LivenessResult:
    is_live: bool
    score: float          # 0.0 = definite spoof, 1.0 = definite real
    is_real_prob: float
    is_spoof_prob: float


class LivenessService:
    def __init__(self):
        self._loaded = False

    def load(self):
        """Pre-warm DeepFace anti-spoof models by running a dummy inference."""
        if not self._loaded:
            try:
                import numpy as np
                from deepface import DeepFace
                # Run a tiny dummy image through extract_faces to trigger
                # MiniFASNet weight download and model initialization.
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                DeepFace.extract_faces(
                    img_path=dummy,
                    anti_spoofing=True,
                    detector_backend="opencv",
                )
                self._loaded = True
                logger.info("Liveness models ready (DeepFace/MiniFASNet)")
            except Exception as e:
                # Model download may fail on first cold start — it will retry at inference time.
                logger.warning(f"Liveness model pre-warm failed: {e}. Will retry at inference time.")
                self._loaded = True  # avoid retrying load() on every request
        return self

    def check(self, image_bgr: np.ndarray, face_bbox=None) -> LivenessResult:
        """
        Run liveness detection on a face image.
        Crops to face region if bbox provided.
        Returns LivenessResult with score and boolean.
        """
        try:
            from deepface import DeepFace

            # Crop to face if we have a bbox to reduce background noise
            if face_bbox is not None:
                x, y, w, h = face_bbox
                pad = int(max(w, h) * 0.15)  # 15% padding
                h_img, w_img = image_bgr.shape[:2]
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_img, x + w + pad)
                y2 = min(h_img, y + h + pad)
                face_img = image_bgr[y1:y2, x1:x2]
            else:
                face_img = image_bgr

            if face_img.size == 0:
                face_img = image_bgr

            # DeepFace anti_spoof returns a list of dicts per face
            result = DeepFace.extract_faces(
                img_path=face_img,
                anti_spoofing=True,
                detector_backend="opencv",
            )

            if not result:
                return LivenessResult(is_live=False, score=0.0, is_real_prob=0.0, is_spoof_prob=1.0)

            face_data = result[0]

            # DeepFace returns is_real and antispoof_score
            antispoof_score = face_data.get("antispoof_score", 0.0)

            # antispoof_score: higher = more likely real
            score = float(antispoof_score)
            is_live = score >= settings.LIVENESS_THRESHOLD

            return LivenessResult(
                is_live=is_live,
                score=score,
                is_real_prob=score,
                is_spoof_prob=1.0 - score,
            )

        except Exception as e:
            logger.warning(f"Liveness check error: {e}. Falling back to texture analysis.")
            return self._fallback_texture_check(image_bgr)

    def _fallback_texture_check(self, image_bgr: np.ndarray) -> LivenessResult:
        """
        Lightweight fallback: LBP texture variance analysis.
        Real faces have complex texture; printed/screen spoofs are flatter.
        Not production-grade but works for demo.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Laplacian variance = sharpness / texture richness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Real face in good lighting typically > 80; screen replay often < 40
        score = float(np.clip(laplacian_var / 200.0, 0.0, 1.0))
        is_live = score > 0.4

        return LivenessResult(
            is_live=is_live,
            score=score,
            is_real_prob=score,
            is_spoof_prob=1.0 - score,
        )


# Singleton
liveness_service = LivenessService()
