from __future__ import annotations

from pathlib import Path
from typing import Tuple

import arff
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from backend.app.core.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    SCORE_COLUMNS,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        dataset = arff.load(file)
    columns = [attribute[0] for attribute in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=columns)


def _clean_missing(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return None if value in {"", "?", "nan", "NaN"} else value
    return value


def _normalize_relation(value):
    value = _clean_missing(value)
    if value is None:
        return None
    mapping = {
        "self": "Self",
        "parent": "Parent",
        "relative": "Relative",
        "health care professional": "Health care professional",
    }
    return mapping.get(str(value).lower(), str(value))


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    required = [*FEATURE_NAMES, TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in SCORE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].dropna().loc[~data[column].dropna().isin([0, 1])]
        if not invalid.empty:
            raise ValueError(f"{column} contains values other than 0/1: {invalid.unique().tolist()}")

    data["age"] = pd.to_numeric(data["age"], errors="coerce")

    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].map(_clean_missing)

    data["relation"] = data["relation"].map(_normalize_relation)

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains invalid or missing values.")
    return data


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_NAMES].copy(), df[TARGET_COLUMN].astype(int).copy()


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
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
