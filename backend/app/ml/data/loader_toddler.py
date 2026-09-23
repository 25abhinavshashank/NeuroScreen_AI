from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.core.config import (
    TODDLER_DATASET_PATH,
    TODDLER_TARGET_COLUMN,
    TODDLER_SCORE_COLUMNS,
    TODDLER_NUMERIC_FEATURES,
    TODDLER_CATEGORICAL_FEATURES,
    TODDLER_FEATURE_NAMES,
)


def load_dataset(path: Path = TODDLER_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Toddler dataset not found: {path}")
    return pd.read_csv(path)


def _clean_missing(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return None if value in {"", "?", "nan", "NaN"} else value
    return value


def _normalize_test_taker(value):
    value = _clean_missing(value)
    if value is None:
        return None
    # Source data has inconsistent casing for the same category
    # ("Health Care Professional" vs "Health care professional").
    mapping = {
        "family member": "Family member",
        "health care professional": "Health care professional",
        "self": "Self",
        "others": "Others",
    }
    return mapping.get(str(value).lower(), str(value))


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    required = [*TODDLER_FEATURE_NAMES, TODDLER_TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in TODDLER_SCORE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].dropna().loc[~data[column].dropna().isin([0, 1])]
        if not invalid.empty:
            raise ValueError(f"{column} contains values other than 0/1: {invalid.unique().tolist()}")

    data["Age_Mons"] = pd.to_numeric(data["Age_Mons"], errors="coerce")

    for column in TODDLER_CATEGORICAL_FEATURES:
        data[column] = data[column].map(_clean_missing)

    data["Who completed the test"] = data["Who completed the test"].map(_normalize_test_taker)

    data[TODDLER_TARGET_COLUMN] = (
        data[TODDLER_TARGET_COLUMN].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    )
    if data[TODDLER_TARGET_COLUMN].isna().any():
        raise ValueError("Target contains invalid or missing values.")
    return data


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[TODDLER_FEATURE_NAMES].copy(), df[TODDLER_TARGET_COLUMN].astype(int).copy()


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, TODDLER_NUMERIC_FEATURES),
        ("categorical", categorical, TODDLER_CATEGORICAL_FEATURES),
    ], remainder="drop")
