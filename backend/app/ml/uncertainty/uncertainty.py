
from pathlib import Path

import arff
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
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
    / "uncertainty_results.csv"
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

def load_dataset():

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

def clean_dataset(df):

    dataframe = df.copy()

    # Numeric columns
    for column in SCORE_COLUMNS:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["age"] = pd.to_numeric(
        dataframe["age"],
        errors="coerce",
    )

    # Categorical columns
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

        dataframe[column] = dataframe[column].where(
            dataframe[column].notna(),
            None,
        )

    # Normalize relation values
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

def prepare_target(df):

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
            "Invalid target values detected."
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
        ]
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
# ENTROPY
# ============================================================

def binary_entropy(probability):

    probability = np.asarray(
        probability,
        dtype=float,
    )

    epsilon = 1e-12

    probability = np.clip(
        probability,
        epsilon,
        1 - epsilon,
    )

    entropy = -(
        probability * np.log2(probability)
        +
        (1 - probability)
        * np.log2(1 - probability)
    )

    return entropy


# ============================================================
# MARGIN
# ============================================================

def prediction_margin(probability):

    probability = np.asarray(
        probability,
        dtype=float,
    )

    return np.abs(
        probability - 0.5
    ) * 2


# ============================================================
# UNCERTAINTY SCORE
# ============================================================

def calculate_uncertainty(probability):

    """
    Returns normalized uncertainty in [0, 1].

    0 = very confident
    1 = maximally uncertain

    For binary classification:

        p = 0.0 or 1.0 -> uncertainty = 0
        p = 0.5         -> uncertainty = 1
    """

    entropy = binary_entropy(
        probability
    )

    return entropy


# ============================================================
# CONFIDENCE SCORE
# ============================================================

def calculate_confidence(probability):

    """
    Simple confidence score based on distance
    from the decision boundary.

    0 = exactly on the 0.5 boundary
    1 = maximally separated from the boundary
    """

    return prediction_margin(
        probability
    )


# ============================================================
# UNCERTAINTY CATEGORIES
# ============================================================

def classify_uncertainty(
    uncertainty,
):

    if uncertainty < 0.25:

        return "LOW"

    if uncertainty < 0.60:

        return "MODERATE"

    return "HIGH"


# ============================================================
# MAIN
# ============================================================

def run_uncertainty_analysis():

    print("=" * 70)

    print(
        "NEUROSCREEN AI — "
        "PREDICTIVE UNCERTAINTY ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load and clean
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
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
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
    # Train model
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
    # Predictions
    # --------------------------------------------------------

    probabilities = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Calculate uncertainty
    # --------------------------------------------------------

    uncertainty = (
        calculate_uncertainty(
            probabilities
        )
    )

    confidence = (
        calculate_confidence(
            probabilities
        )
    )

    categories = [
        classify_uncertainty(
            value
        )
        for value in uncertainty
    ]

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "prediction": predictions,
            "probability_yes": probabilities,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "uncertainty_category": categories,
        }
    )

    # Preserve original test index
    results.insert(
        0,
        "sample_index",
        X_test.index,
    )

    # --------------------------------------------------------
    # Print individual predictions
    # --------------------------------------------------------

    print(
        "\nPrediction-level uncertainty:"
    )

    print(
        "-" * 70
    )

    for _, row in results.iterrows():

        print(
            f"Sample {int(row['sample_index']):3d} | "
            f"Actual={int(row['actual'])} | "
            f"Pred={int(row['prediction'])} | "
            f"P(YES)={row['probability_yes']:.4f} | "
            f"Confidence={row['confidence']:.4f} | "
            f"Uncertainty={row['uncertainty']:.4f} | "
            f"{row['uncertainty_category']}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "UNCERTAINTY SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Mean probability : "
        f"{probabilities.mean():.4f}"
    )

    print(
        f"Mean confidence  : "
        f"{confidence.mean():.4f}"
    )

    print(
        f"Mean uncertainty  : "
        f"{uncertainty.mean():.4f}"
    )

    print(
        f"Minimum uncertainty: "
        f"{uncertainty.min():.4f}"
    )

    print(
        f"Maximum uncertainty: "
        f"{uncertainty.max():.4f}"
    )

    print(
        "\nUncertainty categories:"
    )

    category_counts = (
        results[
            "uncertainty_category"
        ]
        .value_counts()
    )

    for category in [
        "LOW",
        "MODERATE",
        "HIGH",
    ]:

        print(
            f"  {category:8s}: "
            f"{category_counts.get(category, 0)}"
        )

    # --------------------------------------------------------
    # Closest predictions to decision boundary
    # --------------------------------------------------------

    print(
        "\nMost uncertain predictions:"
    )

    print(
        "-" * 70
    )

    most_uncertain = (
        results
        .sort_values(
            "uncertainty",
            ascending=False,
        )
        .head(10)
    )

    for _, row in most_uncertain.iterrows():

        print(
            f"Sample {int(row['sample_index']):3d} | "
            f"P(YES)={row['probability_yes']:.4f} | "
            f"Uncertainty={row['uncertainty']:.4f} | "
            f"Actual={int(row['actual'])}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "UNCERTAINTY ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_PATH}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Uncertainty represents model uncertainty "
        "under this dataset/model setup."
    )

    print(
        "It is NOT equivalent to clinical certainty "
        "or diagnostic certainty."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_uncertainty_analysis()