
from dataclasses import dataclass
from enum import Enum


class DecisionStatus(str, Enum):
    SCREENING_RESULT = "SCREENING_RESULT"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    OOD_REVIEW = "OOD_REVIEW"


@dataclass
class DecisionResult:
    status: DecisionStatus
    prediction: int
    probability_yes: float
    confidence: float
    uncertainty: float
    uncertainty_level: str
    ood_status: str
    reason: str


def calculate_confidence(probability_yes: float) -> float:
    """
    Convert probability distance from 0.5 into a confidence-like score.

    This is a model-derived confidence measure.
    It is NOT clinical certainty.
    """
    return abs(probability_yes - 0.5) * 2.0


def calculate_entropy_uncertainty(probability_yes: float) -> float:
    """
    Normalized binary entropy.

    0.0 -> very certain model probability
    1.0 -> probability close to 0.5
    """
    import math

    p = min(max(probability_yes, 1e-12), 1 - 1e-12)

    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    return entropy


def classify_uncertainty(uncertainty: float) -> str:
    """
    Same uncertainty categories used in uncertainty.py.
    """
    if uncertainty < 0.25:
        return "LOW"

    if uncertainty < 0.60:
        return "MODERATE"

    return "HIGH"


def make_decision(
    probability_yes: float,
    ood_status: str = "IN_DISTRIBUTION",
) -> DecisionResult:

    prediction = int(probability_yes >= 0.5)

    confidence = calculate_confidence(probability_yes)
    uncertainty = calculate_entropy_uncertainty(probability_yes)
    uncertainty_level = classify_uncertainty(uncertainty)

    # ---------------------------------------------------------
    # RULE 1: OOD takes priority
    # ---------------------------------------------------------

    if ood_status == "OOD_SUSPECTED":
        return DecisionResult(
            status=DecisionStatus.OOD_REVIEW,
            prediction=prediction,
            probability_yes=probability_yes,
            confidence=confidence,
            uncertainty=uncertainty,
            uncertainty_level=uncertainty_level,
            ood_status=ood_status,
            reason=(
                "The input appears unusual relative to the model's "
                "training distribution. Manual review is recommended."
            ),
        )

    # ---------------------------------------------------------
    # RULE 2: High uncertainty → human review
    # ---------------------------------------------------------

    if uncertainty_level == "HIGH":
        return DecisionResult(
            status=DecisionStatus.HUMAN_REVIEW,
            prediction=prediction,
            probability_yes=probability_yes,
            confidence=confidence,
            uncertainty=uncertainty,
            uncertainty_level=uncertainty_level,
            ood_status=ood_status,
            reason=(
                "The model is highly uncertain about this prediction. "
                "The result should not be treated as a definitive screening output."
            ),
        )

    # ---------------------------------------------------------
    # RULE 3: Otherwise return model screening result
    # ---------------------------------------------------------

    return DecisionResult(
        status=DecisionStatus.SCREENING_RESULT,
        prediction=prediction,
        probability_yes=probability_yes,
        confidence=confidence,
        uncertainty=uncertainty,
        uncertainty_level=uncertainty_level,
        ood_status=ood_status,
        reason=(
            "The input is within the observed training distribution "
            "and the model uncertainty is not high."
        ),
    )


def run_demo():
    print("=" * 70)
    print("NEUROSCREEN AI — UNIFIED DECISION ENGINE")
    print("=" * 70)

    test_probabilities = [
        0.01,
        0.18,
        0.35,
        0.49,
        0.62,
        0.85,
        0.99,
    ]

    for probability in test_probabilities:

        result = make_decision(
            probability_yes=probability,
            ood_status="IN_DISTRIBUTION",
        )

        print("\n" + "-" * 70)

        print(f"P(YES)          : {result.probability_yes:.4f}")
        print(f"Prediction       : {result.prediction}")
        print(f"Confidence       : {result.confidence:.4f}")
        print(f"Uncertainty      : {result.uncertainty:.4f}")
        print(f"Uncertainty level: {result.uncertainty_level}")
        print(f"OOD status       : {result.ood_status}")
        print(f"Decision         : {result.status.value}")
        print(f"Reason           : {result.reason}")

    # Explicit OOD test
    print("\n" + "=" * 70)
    print("OOD PRIORITY TEST")
    print("=" * 70)

    result = make_decision(
        probability_yes=0.95,
        ood_status="OOD_SUSPECTED",
    )

    print(f"Decision: {result.status.value}")
    print(f"Reason  : {result.reason}")

    print("\n" + "=" * 70)
    print("DECISION ENGINE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()