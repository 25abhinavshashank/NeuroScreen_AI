from pathlib import Path

import arff
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "autism_child"
    / "Autism-Child-Data.arff"
)

RESULTS_DIR = PROJECT_ROOT / "models" / "metadata"

GLOBAL_RESULTS_PATH = (
    RESULTS_DIR / "shap_feature_importance.csv"
)

LOCAL_RESULTS_PATH = (
    RESULTS_DIR / "shap_local_explanations.csv"
)

UNCERTAIN_RESULTS_PATH = (
    RESULTS_DIR / "shap_uncertain_cases.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

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

# Same uncertainty policy used previously
HIGH_UNCERTAINTY_THRESHOLD = 0.60


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    """
    Load the official UCI ARFF dataset.
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

    return pd.DataFrame(
        dataset["data"],
        columns=columns,
    )


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataset(df):
    """
    Clean numeric and categorical fields.
    """

    dataframe = df.copy()

    # --------------------------------------------------------
    # Numeric
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
    # Categorical
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

        dataframe[column] = dataframe[
            column
        ].where(
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
    """
    Convert YES/NO target into 1/0.
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
            "Invalid target values detected."
        )

    return target.astype(int)


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():

    numeric_pipeline = Pipeline(
        [
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
        [
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
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
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

    return LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )


# ============================================================
# FEATURE NAME HELPERS
# ============================================================

def get_transformed_feature_names(
    preprocessor,
):
    """
    Return the names generated by ColumnTransformer.
    """

    names = (
        preprocessor
        .get_feature_names_out()
    )

    cleaned_names = []

    for name in names:

        # Remove:
        #
        # numeric__
        # categorical__
        #
        if "__" in name:

            name = name.split(
                "__",
                1,
            )[1]

        cleaned_names.append(name)

    return cleaned_names


def map_to_original_feature(
    transformed_feature_name,
):
    """
    Map one-hot encoded feature names back
    to their original source feature.
    """

    for category in CATEGORICAL_FEATURES:

        if transformed_feature_name.startswith(
            f"{category}_"
        ):

            return category

    return transformed_feature_name


# ============================================================
# UNCERTAINTY
# ============================================================

def calculate_uncertainty(
    probability_yes,
):
    """
    Normalized binary entropy.

    0.0 = very decisive probability
    1.0 = probability close to 0.5
    """

    probability_yes = np.clip(
        probability_yes,
        1e-12,
        1 - 1e-12,
    )

    entropy = -(
        probability_yes
        * np.log2(probability_yes)
        +
        (1 - probability_yes)
        * np.log2(1 - probability_yes)
    )

    return entropy


def classify_uncertainty(
    uncertainty,
):

    if uncertainty < 0.25:
        return "LOW"

    if uncertainty < 0.60:
        return "MODERATE"

    return "HIGH"


# ============================================================
# AGGREGATE SHAP VALUES
# ============================================================

def aggregate_sample_shap_values(
    shap_values,
    transformed_feature_names,
):
    """
    Aggregate one-hot SHAP values back to original features.

    Example:

        gender_m
        gender_f
        gender_unknown

    become:

        gender
    """

    aggregated = {}

    for value, transformed_name in zip(
        shap_values,
        transformed_feature_names,
    ):

        original_feature = (
            map_to_original_feature(
                transformed_name
            )
        )

        aggregated.setdefault(
            original_feature,
            0.0,
        )

        aggregated[
            original_feature
        ] += float(value)

    return aggregated


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

def calculate_global_importance(
    shap_values,
    transformed_feature_names,
):
    """
    Calculate mean absolute SHAP importance
    at the original feature level.
    """

    global_importance = {}

    for column_index, transformed_name in enumerate(
        transformed_feature_names
    ):

        original_feature = (
            map_to_original_feature(
                transformed_name
            )
        )

        mean_absolute_value = np.mean(
            np.abs(
                shap_values[:, column_index]
            )
        )

        global_importance.setdefault(
            original_feature,
            0.0,
        )

        global_importance[
            original_feature
        ] += float(
            mean_absolute_value
        )

    return global_importance


# ============================================================
# MAIN
# ============================================================

def run_shap_analysis():

    print("=" * 70)
    print("NEUROSCREEN AI — SHAP EXPLAINABILITY")
    print("=" * 70)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

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

    print(
        f"\nDataset shape: "
        f"{dataframe.shape}"
    )

    print(
        f"Features used: "
        f"{len(FEATURE_COLUMNS)}"
    )

    # ========================================================
    # 2. TRAIN / TEST SPLIT
    # ========================================================

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

    # ========================================================
    # 3. PREPROCESS
    # ========================================================

    preprocessor = (
        build_preprocessor()
    )

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

    X_train_transformed = np.asarray(
        X_train_transformed,
        dtype=float,
    )

    X_test_transformed = np.asarray(
        X_test_transformed,
        dtype=float,
    )

    print(
        "\nTransformed training shape:",
        X_train_transformed.shape,
    )

    print(
        "Transformed testing shape:",
        X_test_transformed.shape,
    )

    # ========================================================
    # 4. TRAIN MODEL
    # ========================================================

    model = build_model()

    model.fit(
        X_train_transformed,
        y_train,
    )

    print(
        "\nLogistic Regression trained."
    )

    # ========================================================
    # 5. PREDICTIONS
    # ========================================================

    probabilities = (
        model.predict_proba(
            X_test_transformed
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    uncertainties = np.array(
        [
            calculate_uncertainty(
                probability
            )
            for probability in probabilities
        ]
    )

    uncertainty_levels = np.array(
        [
            classify_uncertainty(
                uncertainty
            )
            for uncertainty in uncertainties
        ]
    )

    # ========================================================
    # 6. SHAP
    # ========================================================

    print(
        "\nCreating SHAP masker..."
    )

    # Use ALL training samples.
    masker = shap.maskers.Independent(
        X_train_transformed,
        max_samples=len(
            X_train_transformed
        ),
    )

    print(
        f"SHAP background samples: "
        f"{len(X_train_transformed)}"
    )

    explainer = shap.LinearExplainer(
        model,
        masker,
    )

    shap_values = (
        explainer.shap_values(
            X_test_transformed
        )
    )

    # Compatibility with SHAP versions
    if isinstance(
        shap_values,
        list,
    ):

        shap_values = shap_values[-1]

    shap_values = np.asarray(
        shap_values
    )

    print(
        "SHAP matrix shape:",
        shap_values.shape,
    )

    # ========================================================
    # 7. FEATURE NAMES
    # ========================================================

    transformed_feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    print(
        "Transformed features:",
        len(
            transformed_feature_names
        ),
    )

    # ========================================================
    # 8. GLOBAL IMPORTANCE
    # ========================================================

    global_importance = (
        calculate_global_importance(
            shap_values,
            transformed_feature_names,
        )
    )

    global_df = pd.DataFrame(
        [
            {
                "feature": feature,
                "mean_absolute_shap": importance,
            }
            for feature, importance
            in global_importance.items()
        ]
    )

    global_df = (
        global_df
        .sort_values(
            "mean_absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # Add ranking

    global_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(global_df) + 1,
        ),
    )

    # ========================================================
    # 9. LOCAL EXPLANATIONS
    # ========================================================

    local_rows = []

    for row_position in range(
        len(X_test)
    ):

        sample_index = X_test.index[
            row_position
        ]

        sample_shap = (
            shap_values[
                row_position
            ]
        )

        aggregated = (
            aggregate_sample_shap_values(
                sample_shap,
                transformed_feature_names,
            )
        )

        # ----------------------------------------------------
        # Positive / negative contributions
        # ----------------------------------------------------

        sorted_features = sorted(
            aggregated.items(),
            key=lambda item: abs(
                item[1]
            ),
            reverse=True,
        )

        top_positive = [
            (
                feature,
                value,
            )
            for feature, value
            in sorted_features
            if value > 0
        ][:5]

        top_negative = [
            (
                feature,
                value,
            )
            for feature, value
            in sorted_features
            if value < 0
        ][:5]

        positive_text = "; ".join(
            [
                f"{feature}={value:.6f}"
                for feature, value
                in top_positive
            ]
        )

        negative_text = "; ".join(
            [
                f"{feature}={value:.6f}"
                for feature, value
                in top_negative
            ]
        )

        actual = int(
            y_test.iloc[
                row_position
            ]
        )

        prediction = int(
            predictions[
                row_position
            ]
        )

        probability = float(
            probabilities[
                row_position
            ]
        )

        uncertainty = float(
            uncertainties[
                row_position
            ]
        )

        uncertainty_level = (
            uncertainty_levels[
                row_position
            ]
        )

        local_rows.append(
            {
                "sample_index": int(
                    sample_index
                ),
                "actual": actual,
                "prediction": prediction,
                "probability_yes": probability,
                "uncertainty": uncertainty,
                "uncertainty_level": (
                    uncertainty_level
                ),
                "correct": (
                    actual == prediction
                ),
                "top_positive_features": (
                    positive_text
                ),
                "top_negative_features": (
                    negative_text
                ),
            }
        )

    local_df = pd.DataFrame(
        local_rows
    )

    # ========================================================
    # 10. UNCERTAIN / ERROR CASES
    # ========================================================

    uncertain_df = (
        local_df[
            (
                local_df[
                    "uncertainty"
                ]
                >= HIGH_UNCERTAINTY_THRESHOLD
            )
            |
            (
                local_df[
                    "correct"
                ]
                == False
            )
        ]
        .sort_values(
            "uncertainty",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # 11. PRINT GLOBAL RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "GLOBAL SHAP FEATURE IMPORTANCE"
    )

    print(
        "=" * 70
    )

    for _, row in global_df.iterrows():

        print(
            f"{int(row['rank']):2d}. "
            f"{row['feature']:20s} "
            f"{row['mean_absolute_shap']:.6f}"
        )

    # ========================================================
    # 12. PRINT UNCERTAIN CASES
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "UNCERTAIN / ERROR CASES"
    )

    print(
        "=" * 70
    )

    if len(uncertain_df) == 0:

        print(
            "No high-uncertainty or "
            "incorrect cases found."
        )

    else:

        for _, row in uncertain_df.iterrows():

            print(
                f"\nSample {int(row['sample_index'])}"
            )

            print(
                f"Actual       : "
                f"{int(row['actual'])}"
            )

            print(
                f"Prediction   : "
                f"{int(row['prediction'])}"
            )

            print(
                f"P(YES)       : "
                f"{row['probability_yes']:.4f}"
            )

            print(
                f"Uncertainty  : "
                f"{row['uncertainty']:.4f}"
            )

            print(
                f"Level        : "
                f"{row['uncertainty_level']}"
            )

            print(
                f"Correct      : "
                f"{row['correct']}"
            )

            print(
                "Positive SHAP:"
            )

            print(
                f"  {row['top_positive_features']}"
            )

            print(
                "Negative SHAP:"
            )

            print(
                f"  {row['top_negative_features']}"
            )

    # ========================================================
    # 13. SAVE
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_df.to_csv(
        GLOBAL_RESULTS_PATH,
        index=False,
    )

    local_df.to_csv(
        LOCAL_RESULTS_PATH,
        index=False,
    )

    uncertain_df.to_csv(
        UNCERTAIN_RESULTS_PATH,
        index=False,
    )

    # ========================================================
    # 14. SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SHAP ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nGlobal results:"
    )

    print(
        GLOBAL_RESULTS_PATH
    )

    print(
        "\nLocal explanations:"
    )

    print(
        LOCAL_RESULTS_PATH
    )

    print(
        "\nUncertain/error cases:"
    )

    print(
        UNCERTAIN_RESULTS_PATH
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "SHAP values describe how the trained "
        "model uses features."
    )

    print(
        "They are not clinical explanations, "
        "causal evidence, or diagnostic conclusions."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_shap_analysis()