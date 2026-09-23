from __future__ import annotations

from pydantic import BaseModel, Field


class VideoObservationResult(BaseModel):
    """Descriptive behavioral-observation metrics extracted from video
    via face landmark tracking (MediaPipe Face Mesh). These are NOT an
    autism prediction and are not combined with the screening model's
    output -- see `disclaimer`. There is no validated dataset linking
    these signals to ASD outcomes; treat them as raw observations for
    a professional to consider, not a score to interpret on their own.
    """

    model_config = {"extra": "allow"}

    frames_analyzed: int
    frames_with_face_detected: int
    face_detected_ratio: float = Field(ge=0.0, le=1.0)

    avg_gaze_toward_camera_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    head_pose_variability: float | None = None
    blink_rate_per_minute: float | None = None

    duration_seconds: float
    disclaimer: str = (
        "These are descriptive facial/gaze-tracking observations only, computed "
        "locally from the uploaded video. They are NOT an autism assessment and "
        "are not combined with the screening result above -- there is no "
        "validated connection between these signals and ASD outcomes."
    )


class VideoAsdPredictionResult(BaseModel):
    """Output of the experimental clip-level video ASD classifier
    (scripts/train_video_asd.py, trained on the real Zenodo ADOS
    pose-video dataset). Measured test accuracy is 59.1% -- barely
    above the 50% chance baseline for this balanced two-class task --
    so `disclaimer` is not boilerplate here: this prediction is not
    reliable and must never be presented as a diagnosis or combined
    with the questionnaire-based screening result.
    """

    predicted_label: str
    probability_asd: float = Field(ge=0.0, le=1.0)
    probability_td: float = Field(ge=0.0, le=1.0)
    segments_analyzed: int
    frames_analyzed: int
    model_version: str
    test_accuracy: float
    test_f1: float
    disclaimer: str


class VideoAnalysisError(BaseModel):
    error: str
    detail: str
