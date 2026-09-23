from pathlib import Path

import arff
import pandas as pd

from sklearn.model_selection import train_test_split


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

TARGET_COLUMN = "Class/ASD"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the UCI Autism Child dataset from the ARFF file.
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
# EXACT DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n" + "=" * 70)
    print("EXACT DUPLICATE ANALYSIS")
    print("=" * 70)

    duplicate_count = df.duplicated().sum()

    print(
        f"\nExact duplicate rows: "
        f"{duplicate_count}"
    )

    duplicate_rows = df[
        df.duplicated(
            keep=False
        )
    ].copy()

    if duplicate_rows.empty:

        print(
            "✓ No exact duplicate rows found."
        )

    else:

        print(
            "\nRows participating in duplication:"
        )

        print(
            duplicate_rows.to_string(
                index=True
            )
        )

    return duplicate_rows


# ============================================================
# DUPLICATE TARGET CONSISTENCY
# ============================================================

def check_duplicate_target_consistency(
    duplicate_rows: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("DUPLICATE TARGET CONSISTENCY")
    print("=" * 70)

    if duplicate_rows.empty:

        print(
            "\nNo duplicates to inspect."
        )

        return

    feature_columns = [
        column
        for column in duplicate_rows.columns
        if column != TARGET_COLUMN
    ]

    grouped = (
        duplicate_rows
        .groupby(feature_columns)[TARGET_COLUMN]
        .nunique()
    )

    inconsistent = grouped[
        grouped > 1
    ]

    if inconsistent.empty:

        print(
            "\n✓ All exact duplicates have "
            "consistent target labels."
        )

    else:

        print(
            "\n⚠ Inconsistent duplicate labels detected."
        )

        print(
            inconsistent
        )


# ============================================================
# ROW KEY CREATION
# ============================================================

def create_row_keys(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> set[str]:
    """
    Convert every feature row into a deterministic string key.

    This allows us to check whether identical observations
    appear in both the training and testing datasets.

    Missing values are explicitly represented as <MISSING>.
    """

    return set(
        dataframe[feature_columns]
        .apply(
            lambda row: "||".join(
                (
                    "<MISSING>"
                    if pd.isna(value)
                    else str(value).strip()
                )
                for value in row
            ),
            axis=1,
        )
    )


# ============================================================
# TRAIN / TEST DUPLICATE CONTAMINATION
# ============================================================

def check_cross_split_duplicates(
    df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("TRAIN / TEST DUPLICATE CONTAMINATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Convert target labels to numerical values
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reproduce the project's stratified split
    # --------------------------------------------------------

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    train_df = df.loc[
        train_indices
    ].copy()

    test_df = df.loc[
        test_indices
    ].copy()

    print(
        f"\nTraining rows: {len(train_df)}"
    )

    print(
        f"Testing rows : {len(test_df)}"
    )

    # --------------------------------------------------------
    # Exclude target column
    #
    # We want to determine whether the same FEATURE
    # observation occurs in both train and test.
    # --------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != TARGET_COLUMN
    ]

    # --------------------------------------------------------
    # Create deterministic row keys
    # --------------------------------------------------------

    train_keys = create_row_keys(
        train_df,
        feature_columns,
    )

    test_keys = create_row_keys(
        test_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Find overlap
    # --------------------------------------------------------

    overlap = (
        train_keys
        .intersection(
            test_keys
        )
    )

    print(
        f"\nIdentical feature rows appearing "
        f"in BOTH train and test: "
        f"{len(overlap)}"
    )

    # --------------------------------------------------------
    # Report result
    # --------------------------------------------------------

    if overlap:

        print(
            "\n⚠ DUPLICATE CONTAMINATION DETECTED"
        )

        print(
            "Identical observations occur "
            "in both training and testing."
        )

        print(
            "\nThese observations can make "
            "test performance look artificially strong."
        )

    else:

        print(
            "\n✓ No identical feature rows "
            "cross the train/test boundary."
        )


# ============================================================
# MAIN AUDIT
# ============================================================

def run_duplicate_audit():

    print("=" * 70)

    print(
        "NEUROSCREEN AI — "
        "DUPLICATE CONTAMINATION AUDIT"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataframe = load_dataset()

    print(
        f"\nDataset: "
        f"{len(dataframe)} rows × "
        f"{len(dataframe.columns)} columns"
    )

    # --------------------------------------------------------
    # Exact duplicate analysis
    # --------------------------------------------------------

    duplicate_rows = (
        analyze_duplicates(
            dataframe
        )
    )

    # --------------------------------------------------------
    # Check duplicate target consistency
    # --------------------------------------------------------

    check_duplicate_target_consistency(
        duplicate_rows
    )

    # --------------------------------------------------------
    # Check train/test contamination
    # --------------------------------------------------------

    check_cross_split_duplicates(
        dataframe
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DUPLICATE AUDIT COMPLETED")
    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_duplicate_audit()