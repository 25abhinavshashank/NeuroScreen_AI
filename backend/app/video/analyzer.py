from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_FRAMES_ANALYZED = 150
SAMPLE_EVERY_N_FRAMES = 3

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def _ensure_model_downloaded() -> None:
    """Download the face landmarker model on first use if it isn't
    already present (e.g. a fresh clone that didn't commit the binary
    model file), rather than failing outright."""

    if MODEL_PATH.exists():
        return

    import urllib.request

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = MODEL_PATH.with_suffix(".task.download")
    urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, tmp_path)
    tmp_path.rename(MODEL_PATH)

# Standard MediaPipe Face Mesh landmark indices (478-point topology
# with iris refinement, the default for the face_landmarker task) --
# stable indices used throughout the open-source face-mesh ecosystem
# for gaze/blink detection, not something this project invented.
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263
_LEFT_IRIS = [468, 469, 470, 471]
_RIGHT_IRIS = [473, 474, 475, 476]

# 6-point eye contours used for the eye-aspect-ratio (EAR) blink test
# (Soukupova & Cech, 2016).
_LEFT_EYE_EAR_POINTS = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE_EAR_POINTS = [362, 385, 387, 263, 373, 380]

EAR_BLINK_THRESHOLD = 0.21


def _eye_aspect_ratio(points: list[tuple[float, float]]) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)."""
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
    vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
    horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
    if horizontal < 1e-6:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def _euler_angles_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Extract (yaw, pitch, roll) degrees from a 4x4 (or 3x3) rotation
    matrix, as returned directly by FaceLandmarker's
    facial_transformation_matrixes -- no manual solvePnP needed since
    the task graph already solves the 3D face model fit internally."""

    rotation_matrix = np.asarray(matrix)[:3, :3]

    sy = float(np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        roll = 0.0

    return float(yaw), float(pitch), float(roll)


def _gaze_is_toward_camera(
    landmarks_px: dict[int, tuple[float, float]],
    yaw: float | None,
    pitch: float | None,
) -> bool | None:
    """Coarse gaze-direction proxy: primarily head yaw/pitch, refined
    by iris position within the eye socket when available. Returns
    None if there isn't enough information to estimate."""

    if yaw is None or pitch is None:
        return None

    if abs(yaw) > 20 or abs(pitch) > 20:
        return False

    required = [*_LEFT_IRIS, *_RIGHT_IRIS, _LEFT_EYE_OUTER, _LEFT_EYE_INNER, _RIGHT_EYE_INNER, _RIGHT_EYE_OUTER]
    if not all(idx in landmarks_px for idx in required):
        return True

    left_iris_x = statistics.fmean(landmarks_px[i][0] for i in _LEFT_IRIS)
    left_span = (landmarks_px[_LEFT_EYE_INNER][0], landmarks_px[_LEFT_EYE_OUTER][0])
    left_center_ratio = (left_iris_x - min(left_span)) / max(abs(left_span[1] - left_span[0]), 1e-6)

    right_iris_x = statistics.fmean(landmarks_px[i][0] for i in _RIGHT_IRIS)
    right_span = (landmarks_px[_RIGHT_EYE_INNER][0], landmarks_px[_RIGHT_EYE_OUTER][0])
    right_center_ratio = (right_iris_x - min(right_span)) / max(abs(right_span[1] - right_span[0]), 1e-6)

    avg_ratio = (left_center_ratio + right_center_ratio) / 2.0
    return 0.30 <= avg_ratio <= 0.70


def analyze_video(video_path: str) -> dict[str, Any]:
    """Extract descriptive face/gaze/head-pose observation metrics
    from a video file. Pure signal extraction -- returns numbers, not
    an autism assessment; see schemas/video.py's disclaimer for why
    this is never fused into the screening prediction."""

    from mediapipe import Image, ImageFormat
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    _ensure_model_downloaded()

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Could not open video file (unsupported or corrupt format).")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (total_frames / fps) if fps > 0 else 0.0

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )

    frames_analyzed = 0
    frames_with_face = 0
    gaze_samples: list[bool] = []
    yaw_samples: list[float] = []
    pitch_samples: list[float] = []
    blink_count = 0
    was_blinking = False

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while frames_analyzed < MAX_FRAMES_ANALYZED:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            if frame_index % SAMPLE_EVERY_N_FRAMES != 0:
                continue

            frames_analyzed += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_index / fps) * 1000) if fps > 0 else frame_index * 33

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.face_landmarks:
                was_blinking = False
                continue

            frames_with_face += 1
            height, width = frame.shape[:2]
            face_landmarks = result.face_landmarks[0]

            landmarks_px = {
                i: (lm.x * width, lm.y * height)
                for i, lm in enumerate(face_landmarks)
            }

            yaw = pitch = None
            if result.facial_transformation_matrixes:
                yaw, pitch, _roll = _euler_angles_from_matrix(
                    result.facial_transformation_matrixes[0]
                )
                yaw_samples.append(yaw)
                pitch_samples.append(pitch)

            gaze_toward_camera = _gaze_is_toward_camera(landmarks_px, yaw, pitch)
            if gaze_toward_camera is not None:
                gaze_samples.append(gaze_toward_camera)

            try:
                left_ear = _eye_aspect_ratio([landmarks_px[i] for i in _LEFT_EYE_EAR_POINTS])
                right_ear = _eye_aspect_ratio([landmarks_px[i] for i in _RIGHT_EYE_EAR_POINTS])
                avg_ear = (left_ear + right_ear) / 2.0

                is_blinking = avg_ear < EAR_BLINK_THRESHOLD
                if is_blinking and not was_blinking:
                    blink_count += 1
                was_blinking = is_blinking
            except KeyError:
                pass

    capture.release()

    face_detected_ratio = (frames_with_face / frames_analyzed) if frames_analyzed else 0.0
    avg_gaze_ratio = (sum(gaze_samples) / len(gaze_samples)) if gaze_samples else None
    head_pose_variability = (
        float(statistics.pstdev(yaw_samples) + statistics.pstdev(pitch_samples))
        if len(yaw_samples) >= 2
        else None
    )
    blink_rate_per_minute = (
        blink_count / (duration_seconds / 60.0) if duration_seconds > 0 else None
    )

    return {
        "frames_analyzed": frames_analyzed,
        "frames_with_face_detected": frames_with_face,
        "face_detected_ratio": round(face_detected_ratio, 4),
        "avg_gaze_toward_camera_ratio": round(avg_gaze_ratio, 4) if avg_gaze_ratio is not None else None,
        "head_pose_variability": round(head_pose_variability, 2) if head_pose_variability is not None else None,
        "blink_rate_per_minute": round(blink_rate_per_minute, 2) if blink_rate_per_minute is not None else None,
        "duration_seconds": round(duration_seconds, 2),
    }
