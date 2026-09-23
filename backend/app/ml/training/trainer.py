
from pathlib import Path

import arff
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate


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
# FEATURE POLICY
# ============================================================

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
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the original UCI ARFF dataset."""

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
# DATA CLEANING
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw values without fitting anything to the data."""

    data = df.copy()

    # Numeric questionnaire scores
    for column in SCORE_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # Age
    data["age"] = pd.to_numeric(
        data["age"],
        errors="coerce",
    )

    # Categorical values
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

    # Normalize relation values
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

    # Target
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
# FEATURE / TARGET SEPARATION
# ============================================================

def split_features_target(
    df: pd.DataFrame,
):
    """
    Select only the approved 18 model features.

    result and age_desc are deliberately excluded.
    """

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing transformer."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
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
# MODEL DEFINITIONS
# ============================================================

def build_models():
    """Return candidate classification models."""

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),

        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(model) -> Pipeline:
    """
    Combine preprocessing and model.

    The preprocessor is fitted independently inside
    every cross-validation training fold.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# CROSS-VALIDATION
# ============================================================

def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
):
    """Evaluate every candidate model using stratified CV."""

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    models = build_models()

    results = []

    for model_name, model in models.items():

        print(
            f"\nEvaluating: {model_name}"
        )

        pipeline = build_pipeline(model)

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
            "model": model_name,
            "accuracy_mean":
                scores["test_accuracy"].mean(),
            "accuracy_std":
                scores["test_accuracy"].std(),
            "precision_mean":
                scores["test_precision"].mean(),
            "recall_mean":
                scores["test_recall"].mean(),
            "f1_mean":
                scores["test_f1"].mean(),
            "roc_auc_mean":
                scores["test_roc_auc"].mean(),
            "roc_auc_std":
                scores["test_roc_auc"].std(),
        }

        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# RESULT REPORT
# ============================================================

def print_results(results: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    display_columns = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
        "roc_auc_std",
    ]

    report = results[display_columns].copy()

    numeric_columns = [
        column
        for column in display_columns
        if column != "model"
    ]

    report[numeric_columns] = report[
        numeric_columns
    ].round(4)

    print(report.to_string(index=False))

    print("\n" + "=" * 70)

    best_model = results.loc[
        results["roc_auc_mean"].idxmax()
    ]

    print(
        "Best model by mean ROC-AUC:"
    )

    print(
        f"  {best_model['model']}"
    )

    print(
        f"  Mean ROC-AUC: "
        f"{best_model['roc_auc_mean']:.4f}"
    )

    print(
        f"  ROC-AUC std: "
        f"{best_model['roc_auc_std']:.4f}"
    )

    print("=" * 70)


# ============================================================
# MAIN TEST
# ============================================================

def run_training_evaluation() -> None:

    print("=" * 70)
    print("NEUROSCREEN AI — MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataframe = load_dataset()

    print(
        f"\nRaw dataset: "
        f"{dataframe.shape[0]} rows × "
        f"{dataframe.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    dataframe = clean_dataset(dataframe)

    # --------------------------------------------------------
    # Select features
    # --------------------------------------------------------

    X, y = split_features_target(dataframe)

    print(
        f"Model feature matrix: {X.shape}"
    )

    print(
        f"Number of approved features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Feature policy
    # --------------------------------------------------------

    print("\nFeature policy:")
    print("-" * 50)

    for feature in FEATURE_COLUMNS:
        print(f"  ✓ {feature}")

    print("\nExplicitly excluded:")
    print("  ✗ result   -> derived from A1-A10")
    print("  ✗ age_desc -> constant field")
    print("  ✗ Class/ASD -> target")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print(
        "\nStarting 5-fold stratified "
        "cross-validation..."
    )

    results = evaluate_models(
        X,
        y,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_results(results)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    run_training_evaluation()