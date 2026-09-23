from __future__ import annotations

"""VGG16 frame-feature extraction, ported from the reference
implementation (AutismBrainBehavior/Video-Neural-Network-ASD-screening,
Natraj et al. 2024, PLOS ONE) to PyTorch since that repo's
tensorflow-gpu==1.15 stack has no native Windows GPU support on a
modern CUDA driver (TensorFlow dropped native Windows GPU support
after 2.10 -- it now requires WSL2). Architecture and preprocessing
match the original exactly: 1 frame/second, resized to 224x224,
ImageNet-normalized, VGG16 conv features (the "hi-dim", include_top=
False variant -- 512x7x7 = 25088-dim per frame, not the 4096-dim FC
features).

Decoding uses PyAV, not cv2.VideoCapture: the real Zenodo archive's
clips are encoded with the old mpeg4 (not h264) codec, which the
pip opencv-python wheel's bundled FFMPEG cannot open on this machine
(isOpened() is False, 0 frames read, no error raised) -- confirmed by
probing a sample clip directly. PyAV bundles its own full-codec ffmpeg
and decodes these files without issue."""

from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from torchvision import transforms

FEATURE_DIM = 25088  # 512 * 7 * 7, matching the reference repo's hi-dim VGG16 features
_PREPROCESS = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vgg16_feature_model(device: torch.device) -> torch.nn.Module:
    """VGG16 with ImageNet weights, classifier head removed -- output
    is the flattened 512x7x7 conv-block feature map per frame."""

    vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(vgg.features, vgg.avgpool)
    feature_extractor.eval()
    feature_extractor.to(device)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    return feature_extractor


@torch.no_grad()
def extract_video_features(
    video_path: str,
    model: torch.nn.Module,
    device: torch.device,
    max_frames: int = 600,
) -> np.ndarray:
    """Extract ~1 frame/second from a video, run each through VGG16.
    Returns an (n_frames, 25088) array. Caches nothing -- caching is
    the caller's responsibility (see build_feature_cache.py)."""

    container = av.open(video_path)
    if not container.streams.video:
        container.close()
        raise ValueError(f"No video stream in: {video_path}")

    stream = container.streams.video[0]
    fps = float(stream.average_rate) if stream.average_rate else 30.0
    frame_interval = max(1, round(fps))  # ~1 sampled frame per second of footage

    features: list[np.ndarray] = []
    try:
        for i, frame in enumerate(container.decode(video=0)):
            if i % frame_interval != 0:
                continue
            frame_rgb = frame.to_ndarray(format="rgb24")
            tensor = _PREPROCESS(frame_rgb).unsqueeze(0).to(device)
            output = model(tensor)
            features.append(output.flatten().cpu().numpy())
            if len(features) >= max_frames:
                break
    finally:
        container.close()

    if not features:
        raise ValueError(f"No frames could be extracted from: {video_path}")

    return np.stack(features)


def extract_and_cache(
    video_path: str,
    cache_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    max_frames: int = 600,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)

    features = extract_video_features(video_path, model, device, max_frames=max_frames)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, features)
    return features
