from __future__ import annotations

"""Scans the Zenodo ADOS pose-video dataset directory structure and
builds (video_path, label) pairs. The exact folder layout is verified
against the actual downloaded archive in scripts/train_video_asd.py
rather than assumed -- this module supports the two most likely
layouts (split-then-class and class-then-split) and fails loudly if
neither matches, rather than silently mis-labeling samples."""

from dataclasses import dataclass
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class VideoSample:
    path: Path
    label: str  # e.g. "ASD" or "TD" -- exact class folder name, not normalized here
    split: str  # "train" or "test"


def _find_videos(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]


def _classify_split(dir_name: str) -> str | None:
    """Substring match so real-world names like "training_set" /
    "testing_set" (the actual Zenodo archive's folder names) match,
    not just the bare "train"/"test"."""
    name = dir_name.lower()
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return None


def scan_dataset(root: Path) -> list[VideoSample]:
    """Try split/class/*.mp4 first, then class/split/*.mp4, then a
    flat class/*.mp4 (single split, caller does the train/test split)."""

    samples: list[VideoSample] = []

    # Layout A: root/{train,test}/{class}/*.mp4
    for split_dir in root.iterdir():
        if not split_dir.is_dir():
            continue
        split = _classify_split(split_dir.name)
        if split is None:
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for video_path in _find_videos(class_dir):
                samples.append(VideoSample(path=video_path, label=class_dir.name, split=split))

    if samples:
        return samples

    # Layout B: root/{class}/{train,test}/*.mp4
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for split_dir in class_dir.iterdir():
            if not split_dir.is_dir():
                continue
            split = _classify_split(split_dir.name)
            if split is None:
                continue
            for video_path in _find_videos(split_dir):
                samples.append(VideoSample(path=video_path, label=class_dir.name, split=split))

    if samples:
        return samples

    # Layout C: flat root/{class}/*.mp4, no pre-defined split.
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for video_path in _find_videos(class_dir):
            samples.append(VideoSample(path=video_path, label=class_dir.name, split="unassigned"))

    return samples
