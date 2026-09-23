
from pathlib import Path

import arff
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
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
from sklearn.svm import SVC


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
    / "holdout_results.csv"
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

EXCLUDED_COLUMNS = [
    "result",
    "age_desc",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the UCI Autism Child dataset.
    """

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

    dataframe = pd.DataFrame(
        dataset["data"],
        columns=columns,
    )

    return dataframe


# ============================================================
# DATA CLEANING
# ============================================================

def clean_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean feature representations without fitting
    any statistical transformation.

    Actual imputation/scaling/encoding happens later
    inside the sklearn pipeline.
    """

    dataframe = df.copy()

    # --------------------------------------------------------
    # Questionnaire scores
    # --------------------------------------------------------

    for column in SCORE_COLUMNS:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Age
    # --------------------------------------------------------

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
    # Normalize relation capitalization
    # --------------------------------------------------------

    if "relation" in dataframe.columns:

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
    """
    Convert YES/NO target values into 1/0.
    """

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
            "Target contains invalid or missing values."
        )

    return target.astype(int)


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    IMPORTANT:
    Imputation, scaling, and one-hot encoding are fitted
    ONLY on the training data because this preprocessor is
    inside the sklearn Pipeline.
    """

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

    return preprocessor


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models() -> dict:
    """
    Define candidate models.

    SVM probability estimates are enabled so that
    ROC-AUC, PR-AUC, and Brier score can be calculated.
    """

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ============================================================
# PIPELINE CREATION
# ============================================================

def build_pipeline(model) -> Pipeline:
    """
    Combine preprocessing and model into one pipeline.

    This prevents preprocessing leakage.
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
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict:
    """
    Calculate comprehensive binary classification metrics.
    """

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = confusion.ravel()

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    if (tn + fp) > 0:

        specificity = (
            tn / (tn + fp)
        )

    else:

        specificity = np.nan

    # --------------------------------------------------------
    # Standard metrics
    # --------------------------------------------------------

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall_sensitivity": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),

        "brier_score": brier_score_loss(
            y_true,
            y_probability,
        ),

        "true_negatives": tn,

        "false_positives": fp,

        "false_negatives": fn,

        "true_positives": tp,
    }

    return metrics


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:

    print("\n" + "=" * 70)

    print(
        f"HOLDOUT EVALUATION — {name}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    pipeline = build_pipeline(
        model
    )

    # --------------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------------

    print(
        "\nFitting model on training data..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "✓ Training completed."
    )

    # --------------------------------------------------------
    # Predict test labels
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    # --------------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------------

    y_probability = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_probability,
    )

    # --------------------------------------------------------
    # Print confusion matrix
    # --------------------------------------------------------

    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "NO",
                "YES",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print(
        "Holdout Metrics:"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision          : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Sensitivity/Recall : "
        f"{metrics['recall_sensitivity']:.4f}"
    )

    print(
        f"Specificity        : "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"F1 Score           : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Balanced Accuracy  : "
        f"{metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC              : "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier Score        : "
        f"{metrics['brier_score']:.4f}"
    )

    # --------------------------------------------------------
    # Attach model name
    # --------------------------------------------------------

    metrics["model"] = name

    return metrics


# ============================================================
# MAIN EVALUATION
# ============================================================

def run_holdout_evaluation():

    print("=" * 70)

    print(
        "NEUROSCREEN AI — "
        "STRICT HOLDOUT EVALUATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataframe = load_dataset()

    print(
        f"\nDataset shape: "
        f"{dataframe.shape[0]} rows × "
        f"{dataframe.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Clean dataset
    # --------------------------------------------------------

    dataframe = clean_dataset(
        dataframe
    )

    # --------------------------------------------------------
    # Select approved features ONLY
    # --------------------------------------------------------

    X = dataframe[
        FEATURE_COLUMNS
    ].copy()

    y = prepare_target(
        dataframe
    )

    print(
        f"\nApproved model features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "\nExcluded features:"
    )

    for column in EXCLUDED_COLUMNS:

        print(
            f"  - {column}"
        )

    print(
        f"  - {TARGET_COLUMN} "
        "(target)"
    )

    # --------------------------------------------------------
    # Stratified train/test split
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
        "\n" + "=" * 70
    )

    print(
        "STRICT TRAIN / TEST SPLIT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples : "
        f"{len(X_test)}"
    )

    print(
        f"\nTraining YES: "
        f"{int(y_train.sum())}"
    )

    print(
        f"Training NO : "
        f"{int((y_train == 0).sum())}"
    )

    print(
        f"\nTesting YES: "
        f"{int(y_test.sum())}"
    )

    print(
        f"Testing NO : "
        f"{int((y_test == 0).sum())}"
    )

    # --------------------------------------------------------
    # Important methodological statement
    # --------------------------------------------------------

    print(
        "\nIMPORTANT:"
    )

    print(
        "The test set will NOT be used to fit "
        "preprocessing, select features, tune "
        "hyperparameters, or determine thresholds."
    )

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    models = build_models()

    results = []

    # --------------------------------------------------------
    # Evaluate every candidate
    # --------------------------------------------------------

    for name, model in models.items():

        result = evaluate_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    columns_order = [
        "model",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    ]

    results_df = results_df[
        columns_order
    ]

    # --------------------------------------------------------
    # Sort by ROC-AUC
    # --------------------------------------------------------

    results_df = results_df.sort_values(
        by="roc_auc",
        ascending=False,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL HOLDOUT COMPARISON"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall_sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = results_df.iloc[0]

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST HOLDOUT MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"\nModel   : "
        f"{best_model['model']}"
    )

    print(
        f"ROC-AUC : "
        f"{best_model['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC  : "
        f"{best_model['pr_auc']:.4f}"
    )

    print(
        f"F1      : "
        f"{best_model['f1']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{best_model['accuracy']:.4f}"
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_PATH}"
    )

    # --------------------------------------------------------
    # Final methodological warning
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION NOTE"
    )

    print(
        "=" * 70
    )

    print(
        "\nThese results describe performance on "
        "this specific UCI dataset and holdout split."
    )

    print(
        "They must NOT be interpreted as clinical "
        "diagnostic accuracy or real-world performance."
    )

    print(
        "\nSTRICT HOLDOUT EVALUATION COMPLETED"
    )

    print(
        "=" * 70
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_holdout_evaluation()