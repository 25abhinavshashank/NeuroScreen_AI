
from pathlib import Path

import arff
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
    / "abstention_results.csv"
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
# DATASET
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
# CLEANING
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
# UNCERTAINTY
# ============================================================

def calculate_uncertainty(probability):

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
# RISK-COVERAGE ANALYSIS
# ============================================================

def calculate_risk_coverage(
    y_true,
    y_pred,
    uncertainty,
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    uncertainty = np.asarray(uncertainty)

    n_samples = len(y_true)

    # --------------------------------------------------------
    # Sort from most confident -> most uncertain
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        uncertainty
    )

    sorted_y_true = y_true[
        sorted_indices
    ]

    sorted_y_pred = y_pred[
        sorted_indices
    ]

    sorted_uncertainty = uncertainty[
        sorted_indices
    ]

    results = []

    # --------------------------------------------------------
    # Evaluate every possible coverage level
    # --------------------------------------------------------

    for accepted_count in range(
        1,
        n_samples + 1,
    ):

        accepted_true = (
            sorted_y_true[
                :accepted_count
            ]
        )

        accepted_pred = (
            sorted_y_pred[
                :accepted_count
            ]
        )

        errors = np.sum(
            accepted_true
            != accepted_pred
        )

        accuracy = accuracy_score(
            accepted_true,
            accepted_pred,
        )

        risk = (
            errors /
            accepted_count
        )

        coverage = (
            accepted_count /
            n_samples
        )

        abstained_count = (
            n_samples -
            accepted_count
        )

        # The next sample would be the
        # most uncertain sample currently
        # excluded from the accepted set.

        if accepted_count < n_samples:

            uncertainty_threshold = (
                sorted_uncertainty[
                    accepted_count
                ]
            )

        else:

            uncertainty_threshold = (
                sorted_uncertainty[
                    -1
                ]
            )

        results.append(
            {
                "accepted_count": accepted_count,
                "abstained_count": abstained_count,
                "coverage": coverage,
                "risk": risk,
                "selective_accuracy": accuracy,
                "uncertainty_threshold": (
                    uncertainty_threshold
                ),
            }
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# SELECT SPECIFIC COVERAGE LEVELS
# ============================================================

def summarize_coverage_levels(
    risk_coverage,
    coverage_levels,
):

    summaries = []

    for requested_coverage in coverage_levels:

        row = (
            risk_coverage
            .iloc[
                (
                    risk_coverage[
                        "coverage"
                    ]
                    -
                    requested_coverage
                )
                .abs()
                .argmin()
            ]
        )

        summaries.append(
            {
                "requested_coverage": (
                    requested_coverage
                ),
                "actual_coverage": (
                    row["coverage"]
                ),
                "accepted_count": (
                    row["accepted_count"]
                ),
                "abstained_count": (
                    row["abstained_count"]
                ),
                "risk": row["risk"],
                "selective_accuracy": (
                    row[
                        "selective_accuracy"
                    ]
                ),
                "uncertainty_threshold": (
                    row[
                        "uncertainty_threshold"
                    ]
                ),
            }
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# MAIN
# ============================================================

def run_abstention_analysis():

    print("=" * 70)

    print(
        "NEUROSCREEN AI — "
        "RISK-COVERAGE / ABSTENTION ANALYSIS"
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
    # Same fixed split
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
    # Train
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

    uncertainty = (
        calculate_uncertainty(
            probabilities
        )
    )

    # --------------------------------------------------------
    # Standard performance
    # --------------------------------------------------------

    standard_accuracy = (
        accuracy_score(
            y_test,
            predictions,
        )
    )

    standard_errors = np.sum(
        y_test.to_numpy()
        != predictions
    )

    print(
        "\nStandard evaluation:"
    )

    print(
        f"Accuracy: "
        f"{standard_accuracy:.4f}"
    )

    print(
        f"Errors  : "
        f"{standard_errors}"
    )

    # ========================================================
    # RISK-COVERAGE
    # ========================================================

    risk_coverage = (
        calculate_risk_coverage(
            y_true=y_test.to_numpy(),
            y_pred=predictions,
            uncertainty=uncertainty,
        )
    )

    # --------------------------------------------------------
    # Print selected coverage levels
    # --------------------------------------------------------

    coverage_levels = [
        1.00,
        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.60,
        0.50,
    ]

    summary = summarize_coverage_levels(
        risk_coverage,
        coverage_levels,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RISK-COVERAGE SUMMARY"
    )

    print(
        "=" * 70
    )

    for _, row in summary.iterrows():

        print(
            f"\nRequested coverage : "
            f"{row['requested_coverage']:.0%}"
        )

        print(
            f"Actual coverage    : "
            f"{row['actual_coverage']:.2%}"
        )

        print(
            f"Accepted cases     : "
            f"{int(row['accepted_count'])}"
        )

        print(
            f"Abstained cases    : "
            f"{int(row['abstained_count'])}"
        )

        print(
            f"Selective accuracy : "
            f"{row['selective_accuracy']:.4f}"
        )

        print(
            f"Selective risk     : "
            f"{row['risk']:.4f}"
        )

        print(
            f"Uncertainty cutoff : "
            f"{row['uncertainty_threshold']:.4f}"
        )

    # ========================================================
    # FIND ZERO-RISK COVERAGE
    # ========================================================

    zero_risk = risk_coverage[
        risk_coverage["risk"] == 0
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "ZERO-RISK COVERAGE ANALYSIS"
    )

    print(
        "=" * 70
    )

    if len(zero_risk) > 0:

        maximum_zero_risk = (
            zero_risk
            .sort_values(
                "coverage",
                ascending=False,
            )
            .iloc[0]
        )

        print(
            "\nHighest observed "
            "zero-error coverage:"
        )

        print(
            f"Coverage: "
            f"{maximum_zero_risk['coverage']:.2%}"
        )

        print(
            f"Accepted: "
            f"{int(maximum_zero_risk['accepted_count'])}"
        )

        print(
            f"Abstained: "
            f"{int(maximum_zero_risk['abstained_count'])}"
        )

    else:

        print(
            "\nNo zero-risk coverage "
            "was observed."
        )

    # ========================================================
    # SAVE
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the complete curve
    risk_coverage.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # Save summary separately
    summary_path = (
        RESULTS_DIR
        / "abstention_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ABSTENTION ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nFull risk-coverage curve:"
        f"\n{RESULTS_PATH}"
    )

    print(
        f"\nCoverage summary:"
        f"\n{summary_path}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Abstention is an experimental selective-prediction "
        "mechanism for this dataset."
    )

    print(
        "It does NOT determine when a child should receive "
        "clinical care or diagnosis."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_abstention_analysis()