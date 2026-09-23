
from pathlib import Path
import arff
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "autism_child"
    / "Autism-Child-Data.arff"
)


# ---------------------------------------------------------
# Load ARFF
# ---------------------------------------------------------

def load_arff(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        dataset = arff.load(file)

    columns = [attribute[0] for attribute in dataset["attributes"]]
    rows = dataset["data"]

    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------
# Dataset audit
# ---------------------------------------------------------

def audit_dataset(df: pd.DataFrame) -> None:

    print("\n" + "=" * 70)
    print("NEUROSCREEN AI — CHILD ASD DATASET AUDIT")
    print("=" * 70)

    print("\n[1] Dataset shape")
    print("-" * 70)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\n[2] Column names")
    print("-" * 70)

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    print("\n[3] Data types")
    print("-" * 70)
    print(df.dtypes)

    print("\n[4] Missing values")
    print("-" * 70)

    missing = df.isna().sum()

    missing_report = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percentage": (
                missing / len(df) * 100
            ).round(2),
        }
    )

    print(
        missing_report[
            missing_report["missing_count"] > 0
        ]
    )

    print("\n[5] Duplicate rows")
    print("-" * 70)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    print("\n[6] Unique values")
    print("-" * 70)

    for column in df.columns:
        unique_values = df[column].dropna().unique()

        print(f"\n{column}")
        print(f"Unique values ({len(unique_values)}):")

        if len(unique_values) <= 20:
            print(unique_values)

    print("\n[7] Possible target columns")
    print("-" * 70)

    target_candidates = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in [
                "class",
                "asd",
                "target",
                "result",
                "diagnosis",
            ]
        )
    ]

    for column in target_candidates:
        print(
            f"\n{column}:"
        )
        print(df[column].value_counts(dropna=False))

    print("\n[8] Screening Score inspection")
    print("-" * 70)

    screening_columns = [
        column
        for column in df.columns
        if "score" in column.lower()
    ]

    if screening_columns:
        for column in screening_columns:
            print(f"\n{column}")
            print(df[column].describe())
            print(df[column].value_counts(dropna=False))
    else:
        print("No column containing 'score' was found.")

    print("\n[9] First 10 rows")
    print("-" * 70)
    print(df.head(10).to_string())

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    dataframe = load_arff(DATASET_PATH)

    audit_dataset(dataframe)