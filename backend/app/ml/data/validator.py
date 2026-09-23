
from pathlib import Path

import arff
import pandas as pd


# ============================================================
# PROJECT PATHS
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
# DATASET SCHEMA
# ============================================================

SCORE_COLUMNS = [
    f"A{i}_Score"
    for i in range(1, 11)
]

TARGET_COLUMN = "Class/ASD"

EXCLUDED_COLUMNS = [
    "result",
    "age_desc",
]


EXPECTED_COLUMNS = [
    *SCORE_COLUMNS,
    "age",
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "result",
    "age_desc",
    "relation",
    TARGET_COLUMN,
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the original UCI ARFF dataset.
    """

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

    df = pd.DataFrame(
        dataset["data"],
        columns=columns,
    )

    return df


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_schema(df: pd.DataFrame) -> None:
    """
    Verify that the raw dataset contains the expected columns.
    """

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    if unexpected_columns:
        print(
            f"WARNING: Unexpected columns found: "
            f"{unexpected_columns}"
        )

    print("Schema validation: PASSED")


# ============================================================
# TARGET VALIDATION
# ============================================================

def validate_target(df: pd.DataFrame) -> None:
    """
    Validate the Class/ASD target column.
    """

    target = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    allowed_values = {"YES", "NO"}

    invalid_values = set(target.unique()) - allowed_values

    if invalid_values:
        raise ValueError(
            f"Invalid target values found: {invalid_values}"
        )

    if target.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    print("Target validation: PASSED")

    print("\nTarget distribution:")
    print(target.value_counts())


# ============================================================
# SCORE VALIDATION
# ============================================================

def validate_questionnaire_scores(
    df: pd.DataFrame,
) -> None:
    """
    Verify A1-A10 contain only binary questionnaire values.
    """

    for column in SCORE_COLUMNS:

        values = set(
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .dropna()
            .unique()
        )

        invalid_values = values - {0, 1}

        if invalid_values:
            raise ValueError(
                f"{column} contains invalid values: "
                f"{invalid_values}"
            )

    print("A1-A10 score validation: PASSED")


# ============================================================
# MISSING VALUE REPORT
# ============================================================

def report_missing_values(
    df: pd.DataFrame,
) -> None:
    """
    Report missing values without modifying the dataset.
    """

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    print("\nMissing-value report:")
    print("-" * 50)

    if missing.empty:
        print("No missing values found.")
        return

    for column, count in missing.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"{column:20s} "
            f"{count:4d} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# DUPLICATE REPORT
# ============================================================

def report_duplicates(
    df: pd.DataFrame,
) -> None:
    """
    Report exact duplicate rows.

    We report them here rather than automatically deleting them.
    """

    duplicate_count = df.duplicated().sum()

    print("\nDuplicate-row report:")
    print("-" * 50)

    print(
        f"Exact duplicate rows: "
        f"{duplicate_count}"
    )


# ============================================================
# EXCLUDED FEATURE REPORT
# ============================================================

def report_excluded_features() -> None:

    print("\nExcluded features:")
    print("-" * 50)

    for column in EXCLUDED_COLUMNS:

        if column == "result":
            reason = (
                "Derived directly from A1-A10; "
                "excluded to avoid redundant/derived information."
            )

        elif column == "age_desc":
            reason = (
                "Constant dataset field; provides no "
                "predictive information."
            )

        else:
            reason = "Excluded by feature policy."

        print(
            f"{column:12s} -> {reason}"
        )


# ============================================================
# COMPLETE AUDIT
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
) -> None:

    print("=" * 70)
    print("NEUROSCREEN AI — DATASET VALIDATION")
    print("=" * 70)

    print(
        f"\nDataset shape: "
        f"{df.shape[0]} rows × {df.shape[1]} columns"
    )

    validate_schema(df)

    validate_target(df)

    validate_questionnaire_scores(df)

    report_missing_values(df)

    report_duplicates(df)

    report_excluded_features()

    print("\n" + "=" * 70)
    print("DATASET VALIDATION COMPLETE")
    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    dataframe = load_dataset()

    validate_dataset(dataframe)