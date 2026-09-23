from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from backend.app.core.config import (
    ARTIFACT_DIR, MODEL_METADATA_DIR, REGISTRY_PATH,
    MODEL_NAME, MODEL_VERSION, MODEL_STATUS, RANDOM_STATE, TEST_SIZE,
    CV_FOLDS, FEATURE_NAMES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_COLUMN, EXCLUDED_FEATURES,
)
from backend.app.ml.data.loader import load_dataset, clean_features, split_features_target, build_preprocessor

ARTIFACT_PATH = ARTIFACT_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.joblib"
METADATA_PATH = MODEL_METADATA_DIR / f"{MODEL_NAME}_{MODEL_VERSION}.json"


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])


def score_rule_audit(X: pd.DataFrame, y: pd.Series) -> dict:
    scores = X[[f"A{i}_Score" for i in range(1, 11)]].sum(axis=1, min_count=1)
    best = None
    for threshold in range(0, 11):
        predicted = (scores >= threshold).astype(int)
        accuracy = float((predicted == y).mean())
        candidate = {"threshold": threshold, "accuracy": accuracy, "errors": int((predicted != y).sum())}
        if best is None or (candidate["errors"], -candidate["accuracy"]) < (best["errors"], -best["accuracy"]):
            best = candidate
    return {"note": "Dataset-label reconstruction audit; not a clinical rule.", "best_threshold": best, "score_missing_rows": int(scores.isna().sum())}


def cv_evaluate(X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc", "pr_auc": "average_precision"}
    result = cross_validate(build_pipeline(), X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1)
    return {
        metric: {"mean": float(np.mean(result[f"test_{metric}"])), "std": float(np.std(result[f"test_{metric}"])), "folds": [float(v) for v in result[f"test_{metric}"]]}
        for metric in scoring
    }


def holdout_evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probability = model.predict_proba(X_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "sensitivity": float(recall_score(y_test, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "brier_score": float(brier_score_loss(y_test, probability)),
        "log_loss": float(log_loss(y_test, probability, labels=[0, 1])),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
    }


def build_ood_baseline(model: Pipeline, X: pd.DataFrame) -> dict:
    transformed = model.named_steps["preprocessor"].transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    center = np.median(dense, axis=0)
    mad = np.median(np.abs(dense - center), axis=0)
    safe_mad = np.where(mad < 1e-8, 1.0, mad)
    distances = np.mean(np.abs((dense - center) / safe_mad), axis=1)
    return {
        "ood_center": center,
        "ood_mad": safe_mad,
        "ood_threshold": float(np.percentile(distances, 99.0)),
        "training_distance_min": float(distances.min()),
        "training_distance_median": float(np.median(distances)),
        "training_distance_mean": float(distances.mean()),
        "training_distance_max": float(distances.max()),
        "method": "mean absolute robust standardized distance; 99th percentile of training distances",
    }


def main() -> None:
    print("=" * 72)
    print("NEUROSCREEN AI v2 MODEL TRAINING")
    print("=" * 72)

    raw = load_dataset()
    data = clean_features(raw)
    X, y = split_features_target(data)

    if list(X.columns) != FEATURE_NAMES:
        raise RuntimeError(f"Feature schema mismatch: {list(X.columns)}")

    print(f"Dataset: {len(X)} rows, {len(X.columns)} approved features")
    print(f"Excluded: {EXCLUDED_FEATURES}")
    print(f"Target distribution: {y.value_counts().sort_index().to_dict()}")

    rule_audit = score_rule_audit(X, y)
    print(f"Label reconstruction audit: {rule_audit['best_threshold']}")

    cv_metrics = cv_evaluate(X, y)
    print(f"CV accuracy: {cv_metrics['accuracy']['mean']:.4f} +/- {cv_metrics['accuracy']['std']:.4f}")
    print(f"CV ROC-AUC: {cv_metrics['roc_auc']['mean']:.4f} +/- {cv_metrics['roc_auc']['std']:.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    model = build_pipeline()
    model.fit(X_train, y_train)
    holdout = holdout_evaluate(model, X_test, y_test)
    print(f"Holdout accuracy: {holdout['accuracy']:.4f}")
    print(f"Holdout ROC-AUC: {holdout['roc_auc']:.4f}")

    # Refit the selected pipeline on all available labeled data for deployment.
    deployment_model = build_pipeline()
    deployment_model.fit(X, y)
    preprocessor = deployment_model.named_steps["preprocessor"]
    classifier = deployment_model.named_steps["classifier"]
    transformed = preprocessor.transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    ood = build_ood_baseline(deployment_model, X)

    transformed_names = preprocessor.get_feature_names_out().tolist()
    if len(transformed_names) != dense.shape[1]:
        raise RuntimeError("Transformed feature names do not match transformed matrix.")

    artifact = {
        "model": classifier,
        "preprocessor": preprocessor,
        "pipeline": deployment_model,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_status": MODEL_STATUS,
        "algorithm": "LogisticRegression",
        "feature_names": FEATURE_NAMES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "transformed_feature_names": transformed_names,
        "transformed_feature_count": len(transformed_names),
        "training_samples": len(X),
        "holdout_samples": len(X_test),
        "cv_metrics": cv_metrics,
        "holdout_metrics": holdout,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "shap_background": dense,
        **ood,
    }

    joblib.dump(artifact, ARTIFACT_PATH)

    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "status": MODEL_STATUS,
        "algorithm": "LogisticRegression",
        "feature_names": FEATURE_NAMES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "excluded_features": EXCLUDED_FEATURES,
        "training_samples": len(X),
        "holdout_samples": len(X_test),
        "cv_folds": CV_FOLDS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cross_validation": cv_metrics,
        "holdout": holdout,
        "label_reconstruction_audit": rule_audit,
        "dataset_sha256": dataset_sha256(PROJECT_ROOT / "data" / "raw" / "autism_child" / "Autism-Child-Data.arff"),
        "artifact": str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Research/educational screening prototype. Dataset metrics do not represent clinical diagnostic accuracy.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Merge into the multi-model registry rather than overwrite it --
    # must not clobber the toddler model's entry (see train_toddler.py).
    registry = {"active_models": {}}
    if REGISTRY_PATH.exists():
        existing = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if "active_models" in existing:
            registry = existing
        elif "active_model" in existing:
            registry = {"active_models": {"child": existing["active_model"]}}

    registry["active_models"]["child"] = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": MODEL_STATUS,
        "artifact": str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "age_range_years": [4.0, 11.0],
    }
    registry["updated_at"] = metadata["created_at"]
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print("\nDeployment artifact:", ARTIFACT_PATH)
    print("Metadata:", METADATA_PATH)
    print("Registry:", REGISTRY_PATH)
    print("\nIMPORTANT: near-perfect results on this dataset should not be interpreted as clinical accuracy.")


if __name__ == "__main__":
    main()
