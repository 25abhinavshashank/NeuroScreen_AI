
from pathlib import Path

import arff
import pandas as pd

from sklearn.model_selection import train_test_split


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
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Class/ASD"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the original ARFF dataset."""

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

    dataframe = pd.DataFrame(
        dataset["data"],
        columns=columns
    )

    return dataframe


# ============================================================
# TARGET CLEANING
# ============================================================

def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Class/ASD into binary target."""

    data = df.copy()

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
# STRATIFIED SPLIT
# ============================================================

def create_train_test_split(
    df: pd.DataFrame,
):
    """
    Create a stratified train/test split.

    Stratification preserves approximately the same
    class distribution in both sets.
    """

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# DISTRIBUTION REPORT
# ============================================================

def print_class_distribution(
    name: str,
    y: pd.Series,
) -> None:
    """Print class counts and percentages."""

    counts = y.value_counts().sort_index()
    percentages = (
        y.value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(f"\n{name} class distribution:")
    print("-" * 50)

    for class_value in counts.index:
        label = "NO" if class_value == 0 else "YES"

        print(
            f"{label:5s} : "
            f"{counts[class_value]:3d} "
            f"({percentages[class_value]:.2f}%)"
        )


# ============================================================
# SPLIT VALIDATION
# ============================================================

def validate_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Validate the generated train/test split."""

    print("\nSplit validation:")
    print("-" * 50)

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    if len(X_train) + len(X_test) != 292:
        raise ValueError(
            "Train + test samples do not equal dataset size."
        )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            "Training or testing set is empty."
        )

    if len(y_train) != len(X_train):
        raise ValueError(
            "Training feature/target sizes do not match."
        )

    if len(y_test) != len(X_test):
        raise ValueError(
            "Testing feature/target sizes do not match."
        )

    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())

    if train_classes != {0, 1}:
        raise ValueError(
            f"Training set missing a class: {train_classes}"
        )

    if test_classes != {0, 1}:
        raise ValueError(
            f"Testing set missing a class: {test_classes}"
        )

    print("✓ Sample counts are valid")
    print("✓ Feature/target sizes match")
    print("✓ Both classes exist in training set")
    print("✓ Both classes exist in testing set")


# ============================================================
# MAIN TEST
# ============================================================

def run_split_test() -> None:

    print("=" * 70)
    print("NEUROSCREEN AI — DATA SPLITTING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    dataframe = load_dataset()

    print(
        f"\nOriginal dataset: "
        f"{dataframe.shape[0]} rows × "
        f"{dataframe.shape[1]} columns"
    )

    # --------------------------------------------------------
    # Prepare target
    # --------------------------------------------------------

    dataframe = prepare_target(dataframe)

    # --------------------------------------------------------
    # Original distribution
    # --------------------------------------------------------

    print_class_distribution(
        "Original dataset",
        dataframe[TARGET_COLUMN],
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print("\nCreating stratified train/test split...")

    X_train, X_test, y_train, y_test = (
        create_train_test_split(dataframe)
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nSplit results:")
    print("-" * 50)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")

    # --------------------------------------------------------
    # Distributions
    # --------------------------------------------------------

    print_class_distribution(
        "Training set",
        y_train,
    )

    print_class_distribution(
        "Testing set",
        y_test,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_split(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # --------------------------------------------------------
    # Leakage reminder
    # --------------------------------------------------------

    print("\nLeakage policy:")
    print("-" * 50)
    print("✓ Dataset split happens BEFORE preprocessing.")
    print("✓ Test data is NOT used to fit preprocessing.")
    print("✓ Stratification preserves target proportions.")
    print("✓ Random state is fixed for reproducibility.")

    print("\n" + "=" * 70)
    print("DATA SPLITTING TEST COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_split_test()