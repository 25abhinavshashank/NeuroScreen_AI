
from pathlib import Path

import arff
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "autism_child"
    / "Autism-Child-Data.arff"
)

TARGET_COLUMN = "Class/ASD"

SCORE_COLUMNS = [
    f"A{i}_Score"
    for i in range(1, 11)
]

NUMERIC_FEATURES = [
    *SCORE_COLUMNS,
    "age",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation",
]

FEATURE_COLUMNS = [
    *NUMERIC_FEATURES,
    *CATEGORICAL_FEATURES,
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        dataset = arff.load(file)

    columns = [
        attribute[0]
        for attribute in dataset["attributes"]
    ]

    return pd.DataFrame(
        dataset["data"],
        columns=columns,
    )


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = df.copy()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in SCORE_COLUMNS:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["age"] = pd.to_numeric(
        dataframe["age"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        dataframe[column] = (
            dataframe[column]
            .replace(
                {
                    "?": None,
                    "": None,
                }
            )
        )

        dataframe[column] = (
            dataframe[column]
            .where(
                dataframe[column].notna(),
                None,
            )
        )

    # --------------------------------------------------------
    # Normalize relation
    # --------------------------------------------------------

    dataframe["relation"] = (
        dataframe["relation"]
        .apply(
            lambda value:
            value.strip().lower()
            if isinstance(value, str)
            else value
        )
    )

    return dataframe


# ============================================================
# TARGET
# ============================================================

def prepare_target(
    df: pd.DataFrame,
) -> pd.Series:

    target = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(
            {
                "YES": 1,
                "NO": 0,
            }
        )
    )

    if target.isna().any():

        raise ValueError(
            "Invalid or missing target values detected."
        )

    return target.astype(int)


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# BASE MODEL
# ============================================================

def build_base_model():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# CALIBRATED MODEL
# ============================================================

def build_calibrated_model():

    base_model = build_base_model()

    calibration_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=calibration_cv,
    )

    return calibrated_model


# ============================================================
# CALIBRATION METRICS
# ============================================================

def evaluate_probabilities(
    y_true,
    probabilities,
    model_name,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    loss = log_loss(
        y_true,
        probabilities,
    )

    accuracy = (
        predictions == y_true
    ).mean()

    print(
        f"\n{model_name}"
    )

    print(
        "-" * 60
    )

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Brier Score : {brier:.6f}"
    )

    print(
        f"Log Loss    : {loss:.6f}"
    )

    print(
        f"Min Prob    : {probabilities.min():.6f}"
    )

    print(
        f"Max Prob    : {probabilities.max():.6f}"
    )

    # --------------------------------------------------------
    # Calibration bins
    # --------------------------------------------------------

    fraction_positive, mean_probability = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=5,
            strategy="uniform",
        )
    )

    print(
        "\nCalibration bins:"
    )

    print(
        "Mean predicted probability | "
        "Observed positive fraction"
    )

    for predicted, observed in zip(
        mean_probability,
        fraction_positive,
    ):

        print(
            f"{predicted:.4f}"
            f"                       | "
            f"{observed:.4f}"
        )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "brier_score": brier,
        "log_loss": loss,
    }


# ============================================================
# MAIN
# ============================================================

def run_calibration():

    print("=" * 70)

    print(
        "NEUROSCREEN AI — "
        "PROBABILITY CALIBRATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    dataframe = load_dataset()

    dataframe = clean_dataset(
        dataframe
    )

    X = dataframe[
        FEATURE_COLUMNS
    ].copy()

    y = prepare_target(
        dataframe
    )

    # --------------------------------------------------------
    # Strict holdout split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    print(
        "\nThe test set remains completely "
        "untouched during calibration."
    )

    # ========================================================
    # UNCALIBRATED MODEL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BASE LOGISTIC REGRESSION"
    )

    print(
        "=" * 70
    )

    base_model = build_base_model()

    base_model.fit(
        X_train,
        y_train,
    )

    base_probabilities = (
        base_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    base_results = evaluate_probabilities(
        y_test,
        base_probabilities,
        "Uncalibrated Logistic Regression",
    )

    # ========================================================
    # CALIBRATED MODEL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CALIBRATED LOGISTIC REGRESSION"
    )

    print(
        "=" * 70
    )

    calibrated_model = build_calibrated_model()

    calibrated_model.fit(
        X_train,
        y_train,
    )

    calibrated_probabilities = (
        calibrated_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    calibrated_results = evaluate_probabilities(
        y_test,
        calibrated_probabilities,
        "Calibrated Logistic Regression",
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    results = pd.DataFrame(
        [
            base_results,
            calibrated_results,
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CALIBRATION COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_path = (
        PROJECT_ROOT
        / "models"
        / "metadata"
        / "calibration_results.csv"
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print(
        f"\nResults saved to:"
        f"\n{results_path}"
    )

    # ========================================================
    # FINAL NOTE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CALIBRATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nCalibration improves the reliability "
        "assessment of predicted probabilities."
    )

    print(
        "A probability should NOT be interpreted "
        "as clinical certainty or diagnosis."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_calibration()