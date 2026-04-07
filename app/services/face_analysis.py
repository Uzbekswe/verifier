"""
Face detection + 478-landmark extraction + head pose estimation.
Uses MediaPipe FaceLandmarker Tasks API (pretrained, no training needed).
Head pose via PnP (Perspective-n-Point) solving with OpenCV.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark
from dataclasses import dataclass
from typing import Optional, Tuple
import logging
import os
import traceback
import urllib.request

logger = logging.getLogger(__name__)

# Model path
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/face_landmarker.task")
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        logger.info("Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        logger.info("Model downloaded successfully.")

# 3D reference face model points (canonical face geometry)
FACE_3D_POINTS = np.array([
    [0.0,    0.0,    0.0],    # nose tip (landmark 1)
    [0.0,   -330.0, -65.0],   # chin (landmark 152)
    [-225.0, 170.0, -135.0],  # left eye corner (landmark 263)
    [225.0,  170.0, -135.0],  # right eye corner (landmark 33)
    [-150.0, -150.0, -125.0], # left mouth corner (landmark 287)
    [150.0,  -150.0, -125.0], # right mouth corner (landmark 57)
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]

# Smile detection landmarks
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_MOUTH_IDX = 61
RIGHT_MOUTH_IDX = 291
LEFT_CHEEK_IDX = 116
RIGHT_CHEEK_IDX = 345


@dataclass
class FaceAnalysis:
    face_detected: bool
    yaw: float = 0.0        # left(-) / right(+)
    pitch: float = 0.0      # up(-) / down(+)
    roll: float = 0.0
    smile_score: float = 0.0
    landmarks: Optional[np.ndarray] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h


class FaceAnalysisService:
    def __init__(self):
        self._detector = None
        self._loaded = False

    def load(self):
        """Lazy-load MediaPipe FaceLandmarker."""
        if not self._loaded:
            _ensure_model()
            model_path = os.path.abspath(_MODEL_PATH)
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._detector = mp_vision.FaceLandmarker.create_from_options(options)
            self._loaded = True
            logger.info("MediaPipe FaceLandmarker loaded")
        return self

    def analyze(self, image_bgr: np.ndarray) -> FaceAnalysis:
        """
        Run full face analysis pipeline on a BGR image.
        Returns FaceAnalysis with pose angles and smile score.
        """
        try:
            if not self._loaded:
                self.load()

            h, w = image_bgr.shape[:2]
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = np.ascontiguousarray(image_rgb)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            result = self._detector.detect(mp_image)

            if not result.face_landmarks:
                return FaceAnalysis(face_detected=False)

            face_landmarks = result.face_landmarks[0]

            # Convert normalized landmarks → pixel coords
            lm_array = np.array([
                [lm.x * w, lm.y * h, lm.z]
                for lm in face_landmarks
            ], dtype=np.float64)

            # --- Head Pose via PnP ---
            yaw, pitch, roll = self._estimate_pose(lm_array, w, h)

            # --- Smile score ---
            smile_score = self._compute_smile_score(lm_array, w, h)

            # --- Face bounding box ---
            xs = lm_array[:, 0]
            ys = lm_array[:, 1]
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()), int(ys.max())
            bbox = (x1, y1, x2 - x1, y2 - y1)

            return FaceAnalysis(
                face_detected=True,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                smile_score=smile_score,
                landmarks=lm_array,
                face_bbox=bbox,
            )
        except Exception as e:
            logger.error(f"Face analysis failed: {e}\n{traceback.format_exc()}")
            return FaceAnalysis(face_detected=False)

    def _estimate_pose(self, lm_array: np.ndarray, img_w: int, img_h: int):
        """
        Solve PnP to get rotation angles.
        Uses 6 stable landmark points mapped to 3D canonical model.
        """
        image_points = np.array([
            lm_array[idx, :2] for idx in FACE_LANDMARK_INDICES
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1         ]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            FACE_3D_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        proj_matrix = np.hstack([rotation_mat, np.zeros((3, 1))])
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

        pitch = float(euler_angles[0])
        yaw   = float(euler_angles[1])
        roll  = float(euler_angles[2])

        return yaw, pitch, roll

    def _compute_smile_score(self, lm_array: np.ndarray, img_w: int, img_h: int) -> float:
        """
        Smile detection using mouth aspect ratio (MAR).
        """
        try:
            upper_lip = lm_array[UPPER_LIP_IDX, :2]
            lower_lip = lm_array[LOWER_LIP_IDX, :2]
            left_mouth = lm_array[LEFT_MOUTH_IDX, :2]
            right_mouth = lm_array[RIGHT_MOUTH_IDX, :2]
            left_cheek = lm_array[LEFT_CHEEK_IDX, :2]
            right_cheek = lm_array[RIGHT_CHEEK_IDX, :2]

            mouth_height = np.linalg.norm(lower_lip - upper_lip)
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            face_height = np.linalg.norm(right_cheek - left_cheek)

            if mouth_width < 1e-6 or face_height < 1e-6:
                return 0.0

            mar = mouth_height / mouth_width
            mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
            left_corner_elevation = mouth_center_y - left_mouth[1]
            right_corner_elevation = mouth_center_y - right_mouth[1]
            corner_score = (left_corner_elevation + right_corner_elevation) / (2 * face_height)

            smile_score = (mar * 2.0) + (corner_score * 3.0)
            return float(np.clip(smile_score, 0.0, 1.0))
        except Exception:
            return 0.0


# Module-level singleton
face_analysis_service = FaceAnalysisService()
