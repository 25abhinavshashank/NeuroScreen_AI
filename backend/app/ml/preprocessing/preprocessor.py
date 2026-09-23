from pathlib import Path

import arff
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


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
# COLUMN DEFINITIONS
# ============================================================

# Questionnaire features
SCORE_COLUMNS = [
    f"A{i}_Score"
    for i in range(1, 11)
]


# Numerical features
NUMERIC_FEATURES = [
    *SCORE_COLUMNS,
    "age",
]


# Categorical features
CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation",
]


# Target
TARGET_COLUMN = "Class/ASD"


# Explicitly excluded columns:
#
# result   -> exactly equal to A1 + A2 + ... + A10
# age_desc -> constant value: "4-11 years"
#
# Class/ASD -> target, therefore never part of X


# ============================================================
# LOAD RAW DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the original UCI ARFF dataset.

    This function does not modify the dataset.
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
# CLEAN FEATURES
# ============================================================

def clean_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform deterministic data cleaning.

    IMPORTANT:
    No statistics are learned here.

    Median imputation, categorical imputation, scaling,
    and one-hot encoding are performed by the sklearn
    preprocessing pipeline.

    This is important for preventing preprocessing leakage.
    """

    data = df.copy()

    # --------------------------------------------------------
    # Convert A1-A10 to numeric
    # --------------------------------------------------------

    for column in SCORE_COLUMNS:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Convert age to numeric
    # --------------------------------------------------------

    data["age"] = pd.to_numeric(
        data["age"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Normalize categorical columns
    #
    # Missing values are converted to None.
    # We do NOT fill them here.
    #
    # The SimpleImputer inside the sklearn pipeline will
    # learn the replacement values from training data only.
    # --------------------------------------------------------

    categorical_columns = [
        "gender",
        "ethnicity",
        "jundice",
        "austim",
        "contry_of_res",
        "used_app_before",
        "relation",
    ]

    for column in categorical_columns:

        data[column] = (
            data[column]
            .astype("object")
            .apply(
                lambda value: (
                    str(value).strip()
                    if pd.notna(value)
                    else None
                )
            )
        )

    # --------------------------------------------------------
    # Normalize relation values
    #
    # Dataset contains both:
    #
    #     Self
    #     self
    #
    # We normalize them to one category.
    # --------------------------------------------------------

    data["relation"] = (
        data["relation"]
        .apply(
            lambda value: (
                value.lower()
                if isinstance(value, str)
                else value
            )
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

    # --------------------------------------------------------
    # Normalize target
    # --------------------------------------------------------

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
# FEATURE / TARGET SEPARATION
# ============================================================

def split_features_target(
    df: pd.DataFrame,
):
    """
    Separate model features X and target y.

    IMPORTANT:

    result
        Excluded because it is exactly the sum of A1-A10.

    age_desc
        Excluded because it is constant.

    Class/ASD
        Used only as the prediction target.
    """

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X = df[features].copy()

    y = df[TARGET_COLUMN].copy()

    return X, y


# ============================================================
# NUMERIC PREPROCESSING PIPELINE
# ============================================================

def build_numeric_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for numeric features.

    Median is learned ONLY when .fit() is called.

    StandardScaler parameters are also learned ONLY during fit.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


# ============================================================
# CATEGORICAL PREPROCESSING PIPELINE
# ============================================================

def build_categorical_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for categorical features.

    Most-frequent imputation and category mappings are learned
    only during fit.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
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


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build the complete preprocessing transformer.

    The returned transformer is NOT fitted.

    During training, it must be fitted only on training data.

    Example:

        preprocessor.fit(X_train)

    Then:

        X_train_transformed =
            preprocessor.transform(X_train)

        X_test_transformed =
            preprocessor.transform(X_test)
    """

    numeric_pipeline = (
        build_numeric_pipeline()
    )

    categorical_pipeline = (
        build_categorical_pipeline()
    )

    transformer = ColumnTransformer(
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

    return transformer


# ============================================================
# GET TRANSFORMED FEATURE NAMES
# ============================================================

def get_feature_names(
    transformer: ColumnTransformer,
):
    """
    Return feature names after preprocessing.

    Useful for:
        - SHAP
        - feature importance
        - model explainability
        - debugging
    """

    return transformer.get_feature_names_out()


# ============================================================
# PREPROCESSOR SMOKE TEST
# ============================================================

def run_preprocessor_test() -> None:

    print("=" * 70)
    print("NEUROSCREEN AI — PREPROCESSOR TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
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

    dataframe = clean_features(
        dataframe
    )

    # --------------------------------------------------------
    # Separate X / y
    # --------------------------------------------------------

    X, y = split_features_target(
        dataframe
    )

    print(
        f"Feature matrix before transformation: "
        f"{X.shape}"
    )

    print(
        f"Target shape: "
        f"{y.shape}"
    )

    # --------------------------------------------------------
    # Show features
    # --------------------------------------------------------

    print("\nFeatures being used:")

    for column in X.columns:
        print(f"  ✓ {column}")

    # --------------------------------------------------------
    # Show excluded features
    # --------------------------------------------------------

    print("\nFeatures explicitly excluded:")

    print(
        "  ✗ result"
        "  -> derived from A1-A10"
    )

    print(
        "  ✗ age_desc"
        "  -> constant field"
    )

    print(
        "  ✗ Class/ASD"
        "  -> prediction target"
    )

    # --------------------------------------------------------
    # Build transformer
    # --------------------------------------------------------

    print(
        "\nBuilding preprocessing transformer..."
    )

    transformer = build_preprocessor()

    # --------------------------------------------------------
    # Smoke test
    #
    # NOTE:
    # This fit_transform() is ONLY for testing that the
    # transformer works.
    #
    # During actual model training we will put this
    # transformer INSIDE the model pipeline and let
    # cross-validation fit it separately on each training fold.
    # --------------------------------------------------------

    transformed = transformer.fit_transform(
        X
    )

    print(
        f"\nTransformed feature matrix: "
        f"{transformed.shape}"
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = get_feature_names(
        transformer
    )

    print(
        f"Generated feature names: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Display first few transformed features
    # --------------------------------------------------------

    print(
        "\nFirst 20 transformed feature names:"
    )

    for name in feature_names[:20]:
        print(f"  {name}")

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    if X.shape[1] != 18:
        raise ValueError(
            f"Expected 18 raw features, "
            f"but found {X.shape[1]}"
        )

    if len(y) != len(X):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if y.isna().any():
        raise ValueError(
            "Target contains missing values after cleaning."
        )

    print(
        "\nPreprocessor test completed successfully."
    )

    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_preprocessor_test()