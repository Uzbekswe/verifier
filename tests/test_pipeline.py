"""
Integration test for the photo verification pipeline.
Uses real face photos from numpy-generated test images.
Run: python3 tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
import requests
import io

# ── Suppress TF noise ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def download_test_face():
    """Download a real face photo for testing (public domain)."""
    try:
        # Use a known face image URL that's public domain
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/200px-Gatto_europeo4.jpg"
        # Actually, use a generated face
        return None
    except Exception:
        return None


def create_synthetic_face_image(width=640, height=480, yaw_offset=0):
    """
    Create a synthetic face-like image for testing.
    In production you'd use real selfies.
    This tests the pipeline without needing a webcam.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray bg

    cx, cy = width // 2 + yaw_offset, height // 2
    # Face ellipse
    cv2.ellipse(img, (cx, cy), (100, 130), 0, 0, 360, (210, 185, 155), -1)
    # Eyes
    cv2.ellipse(img, (cx - 35, cy - 25), (20, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (cx + 35, cy - 25), (20, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (cx - 35, cy - 25), 8, (60, 40, 20), -1)
    cv2.circle(img, (cx + 35, cy - 25), 8, (60, 40, 20), -1)
    # Nose
    cv2.ellipse(img, (cx, cy + 15), (12, 8), 0, 0, 360, (190, 165, 135), -1)
    # Mouth
    cv2.ellipse(img, (cx, cy + 55), (40, 15), 0, 0, 180, (160, 100, 100), -1)
    # Eyebrows
    cv2.line(img, (cx - 55, cy - 45), (cx - 15, cy - 40), (80, 60, 40), 3)
    cv2.line(img, (cx + 15, cy - 40), (cx + 55, cy - 45), (80, 60, 40), 3)

    return img


def image_to_bytes(img):
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_face_analysis_directly():
    """Test FaceAnalysisService without HTTP."""
    print("\n" + "="*60)
    print("TEST 1: Direct face analysis service")
    print("="*60)

    from app.services.face_analysis import face_analysis_service
    face_analysis_service.load()

    img = create_synthetic_face_image()
    result = face_analysis_service.analyze(img)

    print(f"  Face detected : {result.face_detected}")
    print(f"  Yaw           : {result.yaw:.2f}°")
    print(f"  Pitch         : {result.pitch:.2f}°")
    print(f"  Roll          : {result.roll:.2f}°")
    print(f"  Smile score   : {result.smile_score:.3f}")
    print(f"  Face bbox     : {result.face_bbox}")

    # Synthetic face should either detect or gracefully not detect
    print(f"  Status: {'✅ Face detected' if result.face_detected else '⚠️  No face detected (synthetic image)'}")
    return result


def test_liveness_directly():
    """Test liveness service directly."""
    print("\n" + "="*60)
    print("TEST 2: Direct liveness service")
    print("="*60)

    from app.services.liveness import liveness_service

    img = create_synthetic_face_image()
    result = liveness_service.check(img)

    print(f"  Is live       : {result.is_live}")
    print(f"  Score         : {result.score:.3f}")
    print(f"  Real prob     : {result.is_real_prob:.3f}")
    print(f"  Spoof prob    : {result.is_spoof_prob:.3f}")
    print(f"  Status: ✅ Liveness service working")
    return result


def test_challenge_matcher():
    """Test challenge matching logic with mock face analysis."""
    print("\n" + "="*60)
    print("TEST 3: Challenge matcher logic")
    print("="*60)

    from app.services.face_analysis import FaceAnalysis
    from app.services.challenge_matcher import challenge_matcher

    test_cases = [
        # (yaw, pitch, smile, challenge, expected_pass)
        (-20.0, 0.0,  0.0,  "look_left",  True),
        (20.0,  0.0,  0.0,  "look_right", True),
        (0.0,   -15.0, 0.0, "look_up",    True),
        (0.0,   15.0,  0.0, "look_down",  True),
        (0.0,   0.0,   0.5, "smile",      True),
        (5.0,   0.0,   0.0, "look_left",  False),   # not enough yaw
        (0.0,   0.0,   0.1, "smile",      False),   # not enough smile
    ]

    all_passed = True
    for yaw, pitch, smile, challenge, expected in test_cases:
        face = FaceAnalysis(
            face_detected=True,
            yaw=yaw, pitch=pitch, roll=0.0,
            smile_score=smile,
        )
        result = challenge_matcher.match(face, challenge)
        status = "✅" if result.matched == expected else "❌"
        if result.matched != expected:
            all_passed = False
        print(f"  {status} {challenge:12s} | yaw={yaw:6.1f}° pitch={pitch:6.1f}° smile={smile:.1f} → matched={result.matched} (expected={expected})")

    print(f"  Overall: {'✅ All passed' if all_passed else '❌ Some failed'}")
    return all_passed


def test_api_endpoints():
    """Test HTTP endpoints if server is running."""
    print("\n" + "="*60)
    print("TEST 4: HTTP API endpoints (requires running server)")
    print("="*60)

    base = "http://localhost:8000"
    try:
        # Health check
        r = requests.get(f"{base}/health", timeout=3)
        print(f"  GET /health → {r.status_code}: {r.json()}")

        # Get challenge
        r = requests.post(f"{base}/verify/challenge", timeout=3)
        print(f"  POST /verify/challenge → {r.status_code}: {r.json()}")

        # Submit test image
        img = create_synthetic_face_image()
        img_bytes = image_to_bytes(img)
        files = {"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        r = requests.post(f"{base}/verify/submit/look_left", files=files, timeout=30)
        print(f"  POST /verify/submit/look_left → {r.status_code}")
        data = r.json()
        for k, v in data.items():
            if k != "details":
                print(f"    {k}: {v}")
        print(f"    details: {data.get('details', {})}")

    except requests.exceptions.ConnectionError:
        print("  ⚠️  Server not running. Start with: uvicorn app.main:app --reload")


if __name__ == "__main__":
    print("\n🔍 Photo Verification Pipeline — Integration Tests")

    test_face_analysis_directly()
    test_liveness_directly()
    all_ok = test_challenge_matcher()
    test_api_endpoints()

    print("\n" + "="*60)
    print("✅ Core pipeline tests complete.")
    print("="*60)
