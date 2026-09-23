
from pathlib import Path

import arff
import numpy as np
import pandas as pd

print("LEAKAGE SCRIPT STARTED")


PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "autism_child"
    / "Autism-Child-Data.arff"
)


SCORE_COLUMNS = [
    f"A{i}_Score"
    for i in range(1, 11)
]


def load_dataset() -> pd.DataFrame:
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = arff.load(file)

    columns = [attribute[0] for attribute in dataset["attributes"]]

    return pd.DataFrame(
        dataset["data"],
        columns=columns,
    )


def audit_result_leakage(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("RESULT / TARGET LEAKAGE AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # Convert questionnaire scores to numeric
    # ---------------------------------------------------------

    scores = df[SCORE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    result = pd.to_numeric(
        df["result"],
        errors="coerce",
    )

    target = (
        df["Class/ASD"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map({
            "YES": 1,
            "NO": 0,
        })
    )

    # ---------------------------------------------------------
    # Calculate questionnaire score manually
    # ---------------------------------------------------------

    calculated_score = scores.sum(axis=1)

    comparison = pd.DataFrame({
        "calculated_score": calculated_score,
        "result": result,
    })

    comparison["difference"] = (
        comparison["calculated_score"]
        - comparison["result"]
    )

    # ---------------------------------------------------------
    # Exact equality
    # ---------------------------------------------------------

    exact_matches = (
        comparison["difference"] == 0
    ).sum()

    total_valid = comparison.dropna().shape[0]

    print("\n[1] Result vs A1-A10 calculated score")
    print("-" * 70)

    print(
        f"Valid rows       : {total_valid}"
    )

    print(
        f"Exact matches    : {exact_matches}"
    )

    if total_valid:
        print(
            f"Match percentage : "
            f"{exact_matches / total_valid * 100:.2f}%"
        )

    # ---------------------------------------------------------
    # Differences
    # ---------------------------------------------------------

    print("\n[2] Score differences")
    print("-" * 70)

    print(
        comparison["difference"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # ---------------------------------------------------------
    # Correlation
    # ---------------------------------------------------------

    print("\n[3] Correlation")
    print("-" * 70)

    valid = comparison.dropna()

    if len(valid) > 1:
        correlation = valid[
            ["calculated_score", "result"]
        ].corr().iloc[0, 1]

        print(
            f"Correlation between calculated score "
            f"and result: {correlation:.6f}"
        )

    # ---------------------------------------------------------
    # Result vs target
    # ---------------------------------------------------------

    target_comparison = pd.DataFrame({
        "result": result,
        "target": target,
    }).dropna()

    print("\n[4] Result distribution by target")
    print("-" * 70)

    print(
        target_comparison
        .groupby("target")["result"]
        .describe()
    )

    # ---------------------------------------------------------
    # Target relationship
    # ---------------------------------------------------------

    if len(target_comparison) > 1:
        target_correlation = target_comparison[
            ["result", "target"]
        ].corr().iloc[0, 1]

        print(
            f"\nResult-target correlation: "
            f"{target_correlation:.6f}"
        )

    # ---------------------------------------------------------
    # Potential leakage decision
    # ---------------------------------------------------------

    print("\n[5] Leakage decision")
    print("-" * 70)

    if (
        total_valid > 0
        and exact_matches == total_valid
    ):
        print(
            "RESULT IS DERIVED FROM A1-A10."
        )
        print(
            "Do NOT use 'result' as an independent "
            "model feature."
        )
    else:
        print(
            "Result is not exactly identical to the "
            "sum of A1-A10."
        )
        print(
            "Further investigation is required before "
            "using it as a feature."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    dataframe = load_dataset()

    audit_result_leakage(dataframe)