
from pathlib import Path

import arff
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    / "ood_results.csv"
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

# Training-distribution threshold.
# 99th percentile means only the most unusual
# 1% of training observations are flagged.
OOD_PERCENTILE = 99.0


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

    for column in SCORE_COLUMNS:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe["age"] = pd.to_numeric(
        dataframe["age"],
        errors="coerce",
    )

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
# DENSE FEATURE REPRESENTATION
# ============================================================

def transform_features(
    preprocessor,
    X,
):

    transformed = (
        preprocessor
        .transform(X)
    )

    if hasattr(
        transformed,
        "toarray",
    ):

        transformed = (
            transformed.toarray()
        )

    return np.asarray(
        transformed,
        dtype=float,
    )


# ============================================================
# ROBUST CENTER AND SCALE
# ============================================================

def calculate_training_statistics(
    X_train_transformed,
):

    # Median is used as a robust center.
    center = np.median(
        X_train_transformed,
        axis=0,
    )

    # Median absolute deviation.
    absolute_deviation = np.abs(
        X_train_transformed
        - center
    )

    mad = np.median(
        absolute_deviation,
        axis=0,
    )

    # Prevent division by zero for
    # constant/one-hot features.
    mad = np.where(
        mad < 1e-9,
        1.0,
        mad,
    )

    return center, mad


# ============================================================
# ROBUST DISTANCE
# ============================================================

def calculate_distance(
    X,
    center,
    mad,
):

    standardized_distance = (
        np.abs(
            X - center
        )
        / mad
    )

    # Mean absolute standardized deviation.
    distance = (
        standardized_distance
        .mean(axis=1)
    )

    return distance


# ============================================================
# MAIN
# ============================================================

def run_ood_analysis():

    print("=" * 70)

    print(
        "NEUROSCREEN AI — "
        "OUT-OF-DISTRIBUTION ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
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
    # Same split
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
    # Fit preprocessing ONLY on training data
    # --------------------------------------------------------

    preprocessor = build_preprocessor()

    X_train_transformed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_test_transformed = (
        preprocessor.transform(
            X_test
        )
    )

    if hasattr(
        X_train_transformed,
        "toarray",
    ):

        X_train_transformed = (
            X_train_transformed.toarray()
        )

    if hasattr(
        X_test_transformed,
        "toarray",
    ):

        X_test_transformed = (
            X_test_transformed.toarray()
        )

    X_train_transformed = np.asarray(
        X_train_transformed,
        dtype=float,
    )

    X_test_transformed = np.asarray(
        X_test_transformed,
        dtype=float,
    )

    print(
        "\nTransformed training shape:"
        f" {X_train_transformed.shape}"
    )

    print(
        "Transformed testing shape:"
        f" "
        f"{X_test_transformed.shape}"
    )

    # --------------------------------------------------------
    # Training distribution statistics
    # --------------------------------------------------------

    center, mad = (
        calculate_training_statistics(
            X_train_transformed
        )
    )

    training_distances = (
        calculate_distance(
            X_train_transformed,
            center,
            mad,
        )
    )

    test_distances = (
        calculate_distance(
            X_test_transformed,
            center,
            mad,
        )
    )

    # --------------------------------------------------------
    # OOD threshold
    # --------------------------------------------------------

    threshold = np.percentile(
        training_distances,
        OOD_PERCENTILE,
    )

    print(
        "\nTraining distance statistics:"
    )

    print(
        f"Minimum : "
        f"{training_distances.min():.4f}"
    )

    print(
        f"Median  : "
        f"{np.median(training_distances):.4f}"
    )

    print(
        f"Mean    : "
        f"{training_distances.mean():.4f}"
    )

    print(
        f"Maximum : "
        f"{training_distances.max():.4f}"
    )

    print(
        f"\nOOD threshold "
        f"({OOD_PERCENTILE:.0f}th training percentile): "
        f"{threshold:.4f}"
    )

    # --------------------------------------------------------
    # OOD flags
    # --------------------------------------------------------

    ood_flags = (
        test_distances
        > threshold
    )

    # --------------------------------------------------------
    # Train prediction model
    # --------------------------------------------------------

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )

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
    # Results
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "sample_index": X_test.index,
            "actual": y_test.to_numpy(),
            "prediction": predictions,
            "probability_yes": probabilities,
            "distance_from_training_distribution": (
                test_distances
            ),
            "ood_threshold": threshold,
            "ood_score": (
                test_distances
                / threshold
            ),
            "ood_status": np.where(
                ood_flags,
                "OOD_SUSPECTED",
                "IN_DISTRIBUTION",
            ),
        }
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST-SET OOD ANALYSIS"
    )

    print(
        "=" * 70
    )

    for _, row in results.iterrows():

        print(
            f"Sample {int(row['sample_index']):3d} | "
            f"P(YES)={row['probability_yes']:.4f} | "
            f"Distance={row['distance_from_training_distribution']:.4f} | "
            f"OOD Score={row['ood_score']:.4f} | "
            f"{row['ood_status']}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    ood_count = int(
        ood_flags.sum()
    )

    in_distribution_count = (
        len(ood_flags)
        - ood_count
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "OOD SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"IN_DISTRIBUTION : "
        f"{in_distribution_count}"
    )

    print(
        f"OOD_SUSPECTED   : "
        f"{ood_count}"
    )

    print(
        f"OOD percentage  : "
        f"{ood_count / len(ood_flags):.2%}"
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
        "OOD ANALYSIS COMPLETE"
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
        "OOD detection identifies inputs that are "
        "unusual relative to the training distribution."
    )

    print(
        "It does NOT establish whether an input is "
        "clinically valid or invalid."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_ood_analysis()