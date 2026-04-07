"""
Pytest fixtures for the photo verification API.
Uses dependency_overrides so tests run without loading 300MB of model weights.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.face_analysis import FaceAnalysisService, FaceAnalysis
from app.services.liveness import LivenessService, LivenessResult
from app.core.dependencies import run_in_cv_executor


class _MockFaceService(FaceAnalysisService):
    def __init__(self, result: FaceAnalysis):
        super().__init__()
        self._loaded = True
        self._result = result

    def analyze(self, image_bgr):
        return self._result


class _MockLivenessService(LivenessService):
    def __init__(self, result: LivenessResult):
        super().__init__()
        self._loaded = True
        self._result = result

    def check(self, image_bgr, face_bbox=None):
        return self._result


@pytest.fixture
def verified_face():
    return FaceAnalysis(
        face_detected=True,
        yaw=-20.0,
        pitch=0.0,
        roll=0.0,
        smile_score=0.0,
        face_bbox=(100, 100, 200, 200),
    )


@pytest.fixture
def live_result():
    return LivenessResult(is_live=True, score=0.9, is_real_prob=0.9, is_spoof_prob=0.1)


@pytest.fixture
def client(verified_face, live_result, monkeypatch):
    import app.services.face_analysis as fa_module
    import app.services.liveness as lv_module

    monkeypatch.setattr(fa_module, "face_analysis_service", _MockFaceService(verified_face))
    monkeypatch.setattr(lv_module, "liveness_service", _MockLivenessService(live_result))

    # Make run_in_cv_executor run synchronously in tests
    import app.core.dependencies as dep_module
    import asyncio

    async def sync_executor(fn, *args):
        return fn(*args)

    monkeypatch.setattr(dep_module, "run_in_cv_executor", sync_executor)

    app_instance = create_app()
    return TestClient(app_instance)
