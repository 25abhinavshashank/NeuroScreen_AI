
from pathlib import Path

import arff
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "autism_child"
    / "Autism-Child-Data.arff"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Class/ASD"

RANDOM_STATE = 42
N_SPLITS = 5

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


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
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

    data = df.copy()

    for column in SCORE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["age"] = pd.to_numeric(
        data["age"],
        errors="coerce",
    )

    for column in CATEGORICAL_FEATURES:

        data[column] = (
            data[column]
            .astype("object")
            .apply(
                lambda value:
                str(value).strip()
                if pd.notna(value)
                else None
            )
        )

    data["relation"] = (
        data["relation"]
        .apply(
            lambda value:
            value.lower()
            if isinstance(value, str)
            else value
        )
        .replace(
            {
                "self": "Self",
                "parent": "Parent",
                "relative": "Relative",
                "health care professional":
                    "Health care professional",
            }
        )
    )

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
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

    return data


# ============================================================
# QUESTIONNAIRE SCORE ANALYSIS
# ============================================================

def analyze_question_scores(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("A1-A10 QUESTIONNAIRE SCORE ANALYSIS")
    print("=" * 70)

    print(
        "\nMean response by target class:"
    )

    summary = (
        df.groupby(TARGET_COLUMN)[SCORE_COLUMNS]
        .mean()
        .T
    )

    summary.columns = [
        "NO",
        "YES",
    ]

    summary["difference"] = (
        summary["YES"] - summary["NO"]
    )

    print(
        summary.round(3).to_string()
    )


# ============================================================
# TOTAL QUESTIONNAIRE SCORE
# ============================================================

def analyze_total_score(
    df: pd.DataFrame,
) -> None:

    data = df.copy()

    data["calculated_score"] = (
        data[SCORE_COLUMNS]
        .sum(axis=1)
    )

    print("\n" + "=" * 70)
    print("QUESTIONNAIRE TOTAL SCORE ANALYSIS")
    print("=" * 70)

    summary = (
        data.groupby(TARGET_COLUMN)
        ["calculated_score"]
        .agg(
            [
                "count",
                "min",
                "max",
                "mean",
                "std",
            ]
        )
    )

    print(
        summary.round(3).to_string()
    )

    print(
        "\nScore range by target:"
    )

    for target in [0, 1]:

        scores = data.loc[
            data[TARGET_COLUMN] == target,
            "calculated_score",
        ]

        label = (
            "NO"
            if target == 0
            else "YES"
        )

        print(
            f"  {label}: "
            f"{scores.min():.0f} "
            f"to "
            f"{scores.max():.0f}"
        )


# ============================================================
# TARGET SEPARABILITY
# ============================================================

def analyze_score_separability(
    df: pd.DataFrame,
) -> None:

    data = df.copy()

    data["calculated_score"] = (
        data[SCORE_COLUMNS]
        .sum(axis=1)
    )

    no_scores = data.loc[
        data[TARGET_COLUMN] == 0,
        "calculated_score",
    ]

    yes_scores = data.loc[
        data[TARGET_COLUMN] == 1,
        "calculated_score",
    ]

    max_no = no_scores.max()
    min_yes = yes_scores.min()

    print("\n" + "=" * 70)
    print("TARGET SEPARABILITY ANALYSIS")
    print("=" * 70)

    print(
        f"\nMaximum NO score : {max_no}"
    )

    print(
        f"Minimum YES score: {min_yes}"
    )

    if max_no < min_yes:

        print(
            "\n⚠ PERFECT SCORE SEPARATION DETECTED"
        )

        print(
            "The questionnaire total score completely "
            "separates the two target classes in this dataset."
        )

    else:

        print(
            "\nNo perfect separation based on total score."
        )


# ============================================================
# LOGISTIC REGRESSION PIPELINE
# ============================================================

def build_logistic_pipeline():

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

    preprocessor = ColumnTransformer(
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

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# OUT-OF-FOLD PREDICTION
# ============================================================

def evaluate_out_of_fold(
    X: pd.DataFrame,
    y: pd.Series,
) -> None:

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD LOGISTIC REGRESSION ANALYSIS")
    print("=" * 70)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipeline = build_logistic_pipeline()

    predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )

    probabilities = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    cm = confusion_matrix(
        y,
        predictions,
    )

    print("\nConfusion matrix:")
    print(cm)

    print(
        "\nClassification report:"
    )

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "NO",
                "YES",
            ],
            digits=4,
        )
    )

    auc = roc_auc_score(
        y,
        probabilities,
    )

    print(
        f"Out-of-fold ROC-AUC: {auc:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def run_metrics_audit():

    print("=" * 70)
    print("NEUROSCREEN AI — MODEL METRICS & SEPARABILITY AUDIT")
    print("=" * 70)

    dataframe = load_dataset()

    print(
        f"\nDataset: "
        f"{len(dataframe)} rows"
    )

    dataframe = clean_dataset(
        dataframe
    )

    X = dataframe[
        FEATURE_COLUMNS
    ].copy()

    y = dataframe[
        TARGET_COLUMN
    ].copy()

    print(
        f"Model features: "
        f"{X.shape[1]}"
    )

    analyze_question_scores(
        dataframe
    )

    analyze_total_score(
        dataframe
    )

    analyze_score_separability(
        dataframe
    )

    evaluate_out_of_fold(
        X,
        y,
    )

    print("\n" + "=" * 70)
    print("METRICS AUDIT COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_metrics_audit()