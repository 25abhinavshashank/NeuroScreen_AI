from __future__ import annotations

"""Inference for the video-based ASD classifier trained by
scripts/train_video_asd.py. Real test-set numbers (60 epochs, 160
subjects / 9,680 held-out clips): 59.1% accuracy, 60.3% F1 -- barely
above the 50% chance baseline for this balanced binary task, and the
model overfits hard (100% train accuracy by epoch 36). This is NOT a
validated predictor; every result carries that disclaimer verbatim.

A single ~5-frame training clip is very little signal, so at inference
time an uploaded video (which may be much longer than 5 frames) is
split into non-overlapping ~5-frame windows and the per-window
probabilities are averaged -- soft-voting across more of the footage
rather than trusting any single clip, which is the closest analogue
this project can build to how the source paper likely aggregates
clip-level predictions into a subject-level one."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend.app.ml.video_asd.feature_extractor import (
    FEATURE_DIM,
    build_vgg16_feature_model,
    extract_video_features,
    get_device,
)
from backend.app.ml.video_asd.model import VideoASDClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = PROJECT_ROOT / "models" / "artifacts" / "neuro_screen_video_asd_lstm_v1.0.0.pt"

WINDOW_FRAMES = 5  # matches the ~5-6 frame average clip length seen in training
MAX_TOTAL_FRAMES = 120  # cap extraction to ~2 minutes of 1fps sampling

TEST_ACCURACY = 0.5911
TEST_F1 = 0.6026
DISCLAIMER = (
    "Experimental research model, NOT a validated screening or diagnostic tool. "
    f"Measured test accuracy was {TEST_ACCURACY:.1%} on held-out subjects "
    "(a coin flip is 50% for this balanced two-class task) -- the model overfits "
    "the training data heavily and this prediction should not be trusted or acted "
    "on. Shown for research transparency only; it is never combined with the "
    "questionnaire-based screening result above."
)

_vgg_model: torch.nn.Module | None = None
_classifier: VideoASDClassifier | None = None
_label_to_idx: dict[str, int] | None = None
_device: torch.device | None = None


def _ensure_loaded() -> None:
    global _vgg_model, _classifier, _label_to_idx, _device

    if _classifier is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Video ASD model weights not found at {MODEL_PATH}. Run scripts/train_video_asd.py first."
        )

    _device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=_device, weights_only=False)

    _label_to_idx = checkpoint["label_to_idx"]
    classifier = VideoASDClassifier(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    ).to(_device)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    _classifier = classifier

    _vgg_model = build_vgg16_feature_model(_device)


def _windows(features: np.ndarray, window_frames: int) -> list[np.ndarray]:
    n = features.shape[0]
    return [features[i : i + window_frames] for i in range(0, n, window_frames)]


@torch.no_grad()
def predict_video_asd(video_path: str) -> dict:
    """Runs the trained clip-level LSTM over windows of the uploaded
    video's frames and averages the per-window ASD probability.
    Returns a plain dict matching VideoASDPredictionResult's fields."""

    _ensure_loaded()
    assert _classifier is not None and _vgg_model is not None and _label_to_idx is not None and _device is not None

    features = extract_video_features(video_path, _vgg_model, _device, max_frames=MAX_TOTAL_FRAMES)
    windows = [w for w in _windows(features, WINDOW_FRAMES) if len(w) > 0]
    if not windows:
        raise ValueError("No usable frames extracted from the uploaded video.")

    asd_idx = _label_to_idx["ASD"]

    probs_asd: list[float] = []
    for window in windows:
        x = torch.from_numpy(window).float().unsqueeze(0).to(_device)
        lengths = torch.tensor([window.shape[0]])
        logits = _classifier(x, lengths)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probs_asd.append(float(probs[asd_idx]))

    avg_prob_asd = float(np.mean(probs_asd))
    predicted_label = "ASD" if avg_prob_asd >= 0.5 else "TD"

    return {
        "predicted_label": predicted_label,
        "probability_asd": avg_prob_asd,
        "probability_td": 1.0 - avg_prob_asd,
        "segments_analyzed": len(windows),
        "frames_analyzed": int(features.shape[0]),
        "model_version": "v1.0.0",
        "test_accuracy": TEST_ACCURACY,
        "test_f1": TEST_F1,
        "disclaimer": DISCLAIMER,
    }
