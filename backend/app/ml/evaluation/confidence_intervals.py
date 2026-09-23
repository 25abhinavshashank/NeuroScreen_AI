
from pathlib import Path

import arff
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "models"
    / "metadata"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "confidence_intervals.csv"
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

N_BOOTSTRAPS = 5000
CONFIDENCE_LEVEL = 0.95


# ============================================================
# DATASET LOADING
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
# DATA CLEANING
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = df.copy()

    # --------------------------------------------------------
    # Numeric features
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
    # Categorical features
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
# TARGET PREPARATION
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
# MODEL
# ============================================================

def build_model():

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
# SPECIFICITY
# ============================================================

def calculate_specificity(
    y_true,
    y_pred,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_negative = np.sum(
        (y_true == 0) &
        (y_pred == 0)
    )

    false_positive = np.sum(
        (y_true == 0) &
        (y_pred == 1)
    )

    denominator = (
        true_negative +
        false_positive
    )

    if denominator == 0:
        return np.nan

    return (
        true_negative /
        denominator
    )


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability,
):

    metrics = {}

    metrics["accuracy"] = accuracy_score(
        y_true,
        y_pred,
    )

    metrics["precision"] = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    metrics["sensitivity"] = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    metrics["specificity"] = calculate_specificity(
        y_true,
        y_pred,
    )

    metrics["f1"] = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    metrics["balanced_accuracy"] = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    # --------------------------------------------------------
    # ROC-AUC requires both classes
    # --------------------------------------------------------

    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_probability,
        )

        metrics["pr_auc"] = (
            average_precision_score(
                y_true,
                y_probability,
            )
        )

    else:

        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    metrics["brier_score"] = (
        brier_score_loss(
            y_true,
            y_probability,
        )
    )

    return metrics


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_confidence_interval(
    y_true,
    y_pred,
    y_probability,
    metric_name,
    n_bootstraps=N_BOOTSTRAPS,
    confidence_level=CONFIDENCE_LEVEL,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_probability = np.asarray(
        y_probability
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    bootstrap_values = []

    n_samples = len(y_true)

    for _ in range(n_bootstraps):

        indices = rng.integers(
            low=0,
            high=n_samples,
            size=n_samples,
        )

        sample_y_true = (
            y_true[indices]
        )

        sample_y_pred = (
            y_pred[indices]
        )

        sample_probability = (
            y_probability[indices]
        )

        # ROC-AUC and PR-AUC cannot be
        # calculated if a bootstrap sample
        # contains only one class.

        if metric_name in {
            "roc_auc",
            "pr_auc",
        }:

            if len(
                np.unique(
                    sample_y_true
                )
            ) < 2:

                continue

        metrics = calculate_metrics(
            sample_y_true,
            sample_y_pred,
            sample_probability,
        )

        value = metrics[
            metric_name
        ]

        if not np.isnan(value):

            bootstrap_values.append(
                value
            )

    bootstrap_values = np.asarray(
        bootstrap_values
    )

    if len(bootstrap_values) == 0:

        return {
            "estimate": np.nan,
            "lower": np.nan,
            "upper": np.nan,
            "valid_bootstrap_samples": 0,
        }

    alpha = (
        1 -
        confidence_level
    )

    lower_percentile = (
        100 *
        (alpha / 2)
    )

    upper_percentile = (
        100 *
        (1 - alpha / 2)
    )

    lower = np.percentile(
        bootstrap_values,
        lower_percentile,
    )

    upper = np.percentile(
        bootstrap_values,
        upper_percentile,
    )

    original_metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    return {
        "estimate": original_metrics[
            metric_name
        ],
        "lower": lower,
        "upper": upper,
        "valid_bootstrap_samples": len(
            bootstrap_values
        ),
    }


# ============================================================
# MAIN
# ============================================================

def run_confidence_interval_analysis():

    print("=" * 70)

    print(
        "NEUROSCREEN AI — "
        "BOOTSTRAP CONFIDENCE INTERVALS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
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
    # Same strict holdout split
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
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Train Logistic Regression
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

    print(
        "✓ Model trained."
    )

    # --------------------------------------------------------
    # Generate test predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_probability = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Point estimates
    # --------------------------------------------------------

    point_metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability,
    )

    print(
        "\nPoint Estimates:"
    )

    print(
        "-" * 70
    )

    for metric_name, value in point_metrics.items():

        print(
            f"{metric_name:20s}: "
            f"{value:.6f}"
        )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        f"BOOTSTRAP ANALYSIS "
        f"({N_BOOTSTRAPS:,} RESAMPLES)"
    )

    print(
        "=" * 70
    )

    metrics_to_analyze = [
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]

    results = []

    for metric_name in metrics_to_analyze:

        print(
            f"\nCalculating "
            f"{metric_name}..."
        )

        result = (
            bootstrap_confidence_interval(
                y_true=y_test,
                y_pred=y_pred,
                y_probability=y_probability,
                metric_name=metric_name,
            )
        )

        results.append(
            {
                "metric": metric_name,
                "estimate": result[
                    "estimate"
                ],
                "ci_lower": result[
                    "lower"
                ],
                "ci_upper": result[
                    "upper"
                ],
                "confidence_level": CONFIDENCE_LEVEL,
                "bootstrap_samples": result[
                    "valid_bootstrap_samples"
                ],
            }
        )

    results_df = pd.DataFrame(
        results
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "95% BOOTSTRAP CONFIDENCE INTERVALS"
    )

    print(
        "=" * 70
    )

    for _, row in results_df.iterrows():

        print(
            f"\n{row['metric']}"
        )

        print(
            f"  Estimate : "
            f"{row['estimate']:.4f}"
        )

        print(
            f"  95% CI   : "
            f"[{row['ci_lower']:.4f}, "
            f"{row['ci_upper']:.4f}]"
        )

        print(
            f"  Valid bootstrap samples: "
            f"{int(row['bootstrap_samples'])}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "CONFIDENCE INTERVAL ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_PATH}"
    )

    # ========================================================
    # INTERPRETATION NOTE
    # ========================================================

    print(
        "\nNOTE:"
    )

    print(
        "These confidence intervals quantify uncertainty "
        "around performance on this particular holdout sample."
    )

    print(
        "They do NOT establish clinical validity, "
        "diagnostic accuracy, or population-level performance."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_confidence_interval_analysis()