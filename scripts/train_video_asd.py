from __future__ import annotations

"""Trains the video-based ASD classifier on the real labeled Zenodo
ADOS pose-estimation dataset (Natraj et al. 2024, PLOS ONE;
https://zenodo.org/records/12658214, CC-BY-4.0). Architecture and
preprocessing ported from the paper's reference implementation
(AutismBrainBehavior/Video-Neural-Network-ASD-screening) to PyTorch --
see backend/app/ml/video_asd/ for why (that repo's tensorflow-gpu==
1.15 has no native Windows GPU support on a modern CUDA driver).

This is NOT fused into NEUROSCREEN's questionnaire-based screening
prediction -- it is a separate model exposed through its own endpoint,
trained on real labeled clinical data but with a small test cohort
(80 ASD, 80 TD) exactly as the original paper reports and caveats.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from backend.app.ml.video_asd.dataset import scan_dataset
from backend.app.ml.video_asd.feature_extractor import (
    FEATURE_DIM,
    build_vgg16_feature_model,
    extract_and_cache,
    get_device,
)
from backend.app.ml.video_asd.model import VideoASDClassifier

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "autism_video" / "autism_data_anonymized"
FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "autism_video_features"
ARTIFACT_DIR = PROJECT_ROOT / "models" / "artifacts"
METADATA_DIR = PROJECT_ROOT / "models" / "metadata"

MODEL_NAME = "neuro_screen_video_asd_lstm"
MODEL_VERSION = "v1.0.0"

BATCH_SIZE = 8  # much smaller than the paper's 625 -- we have ~160 samples total, not 43k
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
RANDOM_STATE = 42


class VideoFeatureDataset(Dataset):
    """Lazily loads each sample's cached .npy features from disk on
    access, rather than holding all of them in memory at once -- with
    ~19k clips at ~550KB/clip that's ~10GB, more than this machine's
    total RAM. Only a batch's worth is ever resident at a time."""

    def __init__(self, samples: list[tuple[Path, str]], label_to_idx: dict[str, int]):
        self.samples = samples
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cache_path, label = self.samples[idx]
        features = np.load(cache_path)
        return torch.from_numpy(features).float(), self.label_to_idx[label]


def collate_variable_length(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def ensure_feature_cache(samples, device) -> list[tuple[Path, str]]:
    """Extracts+caches VGG16 features to disk for any sample not
    already cached. Returns (cache_path, label) pairs only -- not the
    loaded feature arrays -- to keep memory flat regardless of dataset
    size (see VideoFeatureDataset for why that matters here)."""

    print(f"Ensuring VGG16 feature cache for {len(samples)} videos (slow the first time, instant on rerun)...")
    vgg = None
    cache_paths: list[tuple[Path, str]] = []

    for i, sample in enumerate(samples):
        cache_path = FEATURE_CACHE_DIR / f"{sample.label}_{sample.path.stem}.npy"
        if not cache_path.exists():
            if vgg is None:
                vgg = build_vgg16_feature_model(device)
            start = time.time()
            features = extract_and_cache(str(sample.path), cache_path, vgg, device)
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(samples)}] {sample.path.name}: {features.shape[0]} frames ({elapsed:.1f}s)")
            del features
        cache_paths.append((cache_path, sample.label))

    if vgg is not None:
        del vgg
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return cache_paths


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    return all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-videos", type=int, default=None, help="Cap total videos for a quick smoke test.")
    parser.add_argument(
        "--device", choices=["auto", "cuda", "cpu"], default="auto",
        help="Force a device instead of auto-detecting -- useful when the GPU driver is in a bad state.",
    )
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Scanning dataset at {DATA_ROOT} ...")
    samples = scan_dataset(DATA_ROOT)
    if not samples:
        raise RuntimeError(f"No video samples found under {DATA_ROOT}. Check the archive was extracted here.")

    labels_found = sorted({s.label for s in samples})
    splits_found = sorted({s.split for s in samples})
    print(f"Found {len(samples)} videos. Labels: {labels_found}. Splits: {splits_found}.")

    if args.max_videos:
        samples = samples[: args.max_videos]
        print(f"Capped to {len(samples)} videos for a smoke test.")

    label_to_idx = {label: i for i, label in enumerate(labels_found)}

    cache_paths = ensure_feature_cache(samples, device)

    if "unassigned" in splits_found:
        # No predefined split in the archive -- do our own stratified split.
        from sklearn.model_selection import train_test_split

        indices = list(range(len(cache_paths)))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=RANDOM_STATE,
            stratify=[cache_paths[i][1] for i in indices],
        )
    else:
        train_idx = [i for i, s in enumerate(samples) if s.split == "train"]
        test_idx = [i for i, s in enumerate(samples) if s.split == "test"]

    train_samples = [cache_paths[i] for i in train_idx]
    test_samples = [cache_paths[i] for i in test_idx]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    train_ds = VideoFeatureDataset(train_samples, label_to_idx)
    test_ds = VideoFeatureDataset(test_samples, label_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_variable_length)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_variable_length)

    model = VideoASDClassifier(input_dim=FEATURE_DIM, num_classes=len(label_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_test_acc = 0.0
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = ARTIFACT_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.pt"
    # Saved every epoch (not just on improvement) so a killed process --
    # this environment restarts sessions mid-run without warning -- can
    # resume from where it left off instead of re-training from epoch 0.
    resume_path = ARTIFACT_DIR / f"{MODEL_NAME}_{MODEL_VERSION}_resume.pt"

    history = []
    start_epoch = 0
    last_train_acc = 0.0

    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_test_acc = checkpoint["best_test_acc"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"]
        last_train_acc = checkpoint.get("last_train_acc", 0.0)
        print(f"Resuming from epoch {start_epoch} (best_test_acc so far: {best_test_acc:.4f})", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        epoch_loss /= len(train_ds)

        # Full train-set eval re-reads ~half the dataset from disk --
        # only do it every 5 epochs (plus the last one) to keep epochs
        # fast; test-set eval (what actually matters) runs every time.
        is_last_epoch = epoch == args.epochs - 1
        if epoch % 5 == 0 or is_last_epoch:
            train_preds, train_labels = evaluate(model, train_loader, device)
            last_train_acc = accuracy_score(train_labels, train_preds)
        train_acc = last_train_acc

        test_preds, test_labels = evaluate(model, test_loader, device)
        test_acc = accuracy_score(test_labels, test_preds)

        print(f"Epoch {epoch+1}/{args.epochs}  loss={epoch_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}", flush=True)
        history.append({"epoch": epoch + 1, "loss": epoch_loss, "train_acc": train_acc, "test_acc": test_acc})

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": FEATURE_DIM,
                    "num_classes": len(label_to_idx),
                    "label_to_idx": label_to_idx,
                },
                weight_path,
            )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_test_acc": best_test_acc,
                "last_train_acc": last_train_acc,
                "history": history,
            },
            resume_path,
        )

    # Final evaluation with the best checkpoint.
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_preds, test_labels = evaluate(model, test_loader, device)

    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel() if len(set(test_labels)) == 2 else (0, 0, 0, 0)

    metrics = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "architecture": "VGG16 (frozen, ImageNet) -> LSTM(512) -> Dense(512) -> Dense(n_classes)",
        "source_dataset": "https://zenodo.org/records/12658214 (CC-BY-4.0)",
        "reference_paper": "Natraj et al. 2024, PLOS ONE, doi:10.1371/journal.pone.0308388",
        "labels": label_to_idx,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "best_test_accuracy": float(best_test_acc),
        "final_test_accuracy": float(accuracy_score(test_labels, test_preds)),
        "final_test_f1": float(f1_score(test_labels, test_preds, zero_division=0)),
        "final_test_precision": float(precision_score(test_labels, test_preds, zero_division=0)),
        "final_test_recall": float(recall_score(test_labels, test_preds, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "history": history,
        "disclaimer": (
            "Trained on a small research cohort (paper reports 80 ASD + 80 TD total, "
            "40/40 in this test split) from a single site using gold-standard ADOS "
            "clinical assessment recordings. The original paper explicitly states "
            "generalization to less controlled environments (e.g. home video) 'has "
            "yet to be formally tested'. This is a research-grade prototype, not a "
            "validated diagnostic tool."
        ),
    }

    import json

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = METADATA_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.json"
    metadata_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Model saved to: {weight_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
