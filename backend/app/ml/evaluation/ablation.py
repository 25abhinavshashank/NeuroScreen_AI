
from pathlib import Path

import arff
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
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


# ============================================================
# FEATURE GROUPS
# ============================================================

QUESTIONNAIRE_FEATURES = [
    f"A{i}_Score"
    for i in range(1, 11)
]

CONTEXT_FEATURES = [
    "age",
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation",
]

ALL_FEATURES = [
    *QUESTIONNAIRE_FEATURES,
    *CONTEXT_FEATURES,
]


# ============================================================
# LOAD DATASET
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

    # ----------------------------
    # Questionnaire scores
    # ----------------------------

    for column in QUESTIONNAIRE_FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ----------------------------
    # Age
    # ----------------------------

    data["age"] = pd.to_numeric(
        data["age"],
        errors="coerce",
    )

    # ----------------------------
    # Categorical variables
    # ----------------------------

    for column in [
        "gender",
        "ethnicity",
        "jundice",
        "austim",
        "contry_of_res",
        "used_app_before",
        "relation",
    ]:

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

    # Normalize relation
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

    # ----------------------------
    # Target
    # ----------------------------

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

    if data[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Target contains invalid or missing values."
        )

    return data


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    features,
) -> ColumnTransformer:

    numeric_features = [
        feature
        for feature in features
        if feature in QUESTIONNAIRE_FEATURES
        or feature == "age"
    ]

    categorical_features = [
        feature
        for feature in features
        if feature in CONTEXT_FEATURES
        and feature != "age"
    ]

    transformers = []

    # Numeric pipeline
    if numeric_features:

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

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    # Categorical pipeline
    if categorical_features:

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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODEL
# ============================================================

def build_pipeline(
    features,
) -> Pipeline:

    preprocessor = build_preprocessor(
        features
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
# FEATURE GROUP EXPERIMENT
# ============================================================

def evaluate_feature_group(
    name: str,
    features: list,
    dataframe: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    print("\nFeatures:")

    for feature in features:
        print(f"  ✓ {feature}")

    X = dataframe[
        features
    ].copy()

    y = dataframe[
        TARGET_COLUMN
    ].copy()

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline(
        features
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    result = {
        "feature_group": name,
        "features": len(features),
        "accuracy": scores[
            "test_accuracy"
        ].mean(),
        "precision": scores[
            "test_precision"
        ].mean(),
        "recall": scores[
            "test_recall"
        ].mean(),
        "f1": scores[
            "test_f1"
        ].mean(),
        "roc_auc": scores[
            "test_roc_auc"
        ].mean(),
    }

    print("\nResults:")

    for metric, value in result.items():

        if metric in {
            "feature_group",
        }:
            continue

        if metric == "features":
            print(
                f"{metric:12s}: "
                f"{value}"
            )
        else:
            print(
                f"{metric:12s}: "
                f"{value:.4f}"
            )

    return result


# ============================================================
# DETERMINISTIC QUESTIONNAIRE RULE
# ============================================================

def evaluate_questionnaire_rule(
    dataframe: pd.DataFrame,
):

    print("\n" + "=" * 70)
    print("EXPERIMENT: DETERMINISTIC QUESTIONNAIRE RULE")
    print("=" * 70)

    calculated_score = (
        dataframe[
            QUESTIONNAIRE_FEATURES
        ]
        .sum(axis=1)
    )

    predictions = (
        calculated_score >= 7
    ).astype(int)

    actual = dataframe[
        TARGET_COLUMN
    ]

    accuracy = (
        predictions == actual
    ).mean()

    print(
        "\nRule:"
    )

    print(
        "  score >= 7 → YES"
    )

    print(
        "  score < 7  → NO"
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Correct predictions: "
        f"{(predictions == actual).sum()}"
    )

    print(
        f"Incorrect predictions: "
        f"{(predictions != actual).sum()}"
    )

    return {
        "feature_group":
            "Questionnaire rule",
        "features":
            10,
        "accuracy":
            accuracy,
        "precision":
            None,
        "recall":
            None,
        "f1":
            None,
        "roc_auc":
            None,
    }


# ============================================================
# REPORT
# ============================================================

def print_comparison(
    results: list,
):

    print("\n" + "=" * 70)
    print("ABLATION STUDY — FINAL COMPARISON")
    print("=" * 70)

    table = pd.DataFrame(
        results
    )

    numeric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    table[numeric_columns] = (
        table[numeric_columns]
        .round(4)
    )

    print(
        table.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def run_ablation():

    print("=" * 70)
    print("NEUROSCREEN AI — FEATURE ABLATION STUDY")
    print("=" * 70)

    dataframe = load_dataset()

    print(
        f"\nDataset: "
        f"{len(dataframe)} rows"
    )

    dataframe = clean_dataset(
        dataframe
    )

    results = []

    # --------------------------------------------------------
    # Experiment A
    # --------------------------------------------------------

    questionnaire_result = (
        evaluate_feature_group(
            "Questionnaire only",
            QUESTIONNAIRE_FEATURES,
            dataframe,
        )
    )

    results.append(
        questionnaire_result
    )

    # --------------------------------------------------------
    # Experiment B
    # --------------------------------------------------------

    context_result = (
        evaluate_feature_group(
            "Context only",
            CONTEXT_FEATURES,
            dataframe,
        )
    )

    results.append(
        context_result
    )

    # --------------------------------------------------------
    # Experiment C
    # --------------------------------------------------------

    combined_result = (
        evaluate_feature_group(
            "Questionnaire + Context",
            ALL_FEATURES,
            dataframe,
        )
    )

    results.append(
        combined_result
    )

    # --------------------------------------------------------
    # Experiment D
    # --------------------------------------------------------

    rule_result = (
        evaluate_questionnaire_rule(
            dataframe
        )
    )

    results.append(
        rule_result
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print_comparison(
        results
    )

    print(
        "\nAblation study completed successfully."
    )


if __name__ == "__main__":
    run_ablation()