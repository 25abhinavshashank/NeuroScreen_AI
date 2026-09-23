from __future__ import annotations

import numpy as np

from backend.app.video.analyzer import (
    _euler_angles_from_matrix,
    _eye_aspect_ratio,
    _gaze_is_toward_camera,
)


def test_eye_aspect_ratio_open_eye_is_larger_than_closed():
    open_eye = [(0, 0), (1, -2), (2, -2), (3, 0), (2, 2), (1, 2)]
    closed_eye = [(0, 0), (1, -0.2), (2, -0.2), (3, 0), (2, 0.2), (1, 0.2)]

    open_ear = _eye_aspect_ratio(open_eye)
    closed_ear = _eye_aspect_ratio(closed_eye)

    assert open_ear > closed_ear
    assert closed_ear < 0.21  # below the module's blink threshold
    assert open_ear >= 0.21


def test_eye_aspect_ratio_zero_width_does_not_crash():
    degenerate = [(0, 0), (0, -1), (0, -1), (0, 0), (0, 1), (0, 1)]
    assert _eye_aspect_ratio(degenerate) == 0.0


def test_euler_angles_identity_matrix_is_zero():
    identity = np.eye(4)
    yaw, pitch, roll = _euler_angles_from_matrix(identity)
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6
    assert abs(roll) < 1e-6


def test_euler_angles_90_degree_yaw_rotation():
    # Rotation of 90 degrees around the Y axis (yaw).
    theta = np.radians(90)
    rotation_y = np.array(
        [
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ]
    )
    yaw, _pitch, _roll = _euler_angles_from_matrix(rotation_y)
    assert abs(abs(yaw) - 90.0) < 1.0


def test_gaze_toward_camera_none_without_pose():
    assert _gaze_is_toward_camera({}, None, None) is None


def test_gaze_not_toward_camera_when_head_turned():
    assert _gaze_is_toward_camera({}, yaw=45.0, pitch=0.0) is False


def test_gaze_toward_camera_when_head_frontal_and_no_iris_data():
    # No iris landmarks supplied -> falls back to head-orientation-only.
    assert _gaze_is_toward_camera({}, yaw=5.0, pitch=5.0) is True
