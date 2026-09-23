from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import json
import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import (
    TODDLER_FEATURE_NAMES,
    TODDLER_NUMERIC_FEATURES,
    TODDLER_CATEGORICAL_FEATURES,
    TODDLER_SCORE_COLUMNS,
    TODDLER_AGE_MIN_YEARS,
    TODDLER_AGE_MAX_YEARS,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "models"
    / "registry"
    / "registry.json"
)


# ============================================================
# APPROVED FEATURE SCHEMA
# ============================================================

APPROVED_FEATURES = [
    "A1_Score",
    "A2_Score",
    "A3_Score",
    "A4_Score",
    "A5_Score",
    "A6_Score",
    "A7_Score",
    "A8_Score",
    "A9_Score",
    "A10_Score",
    "age",
    "gender",
    "ethnicity",
    "jundice",
    "austim",
    "contry_of_res",
    "used_app_before",
    "relation",
]


# ============================================================
# EXCLUDED DATASET FEATURES
# ============================================================

EXCLUDED_FEATURES = [
    "result",
    "age_desc",
]

# The training dataset (data/raw/autism_child/Autism-Child-Data.arff) contains
# only integer ages 4-11 (confirmed by direct inspection: min=4.0, max=11.0,
# no infant/toddler examples at all). The generic OOD/Mahalanobis distance
# check does not reliably catch an out-of-range age on its own -- age is 1
# of 86 post-encoding features, so it gets diluted by the rest. This is an
# explicit, separate check for exactly that known boundary.
TRAINING_AGE_MIN_YEARS = 4.0
TRAINING_AGE_MAX_YEARS = 11.0


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_numeric(value: Any) -> float:
    """
    Convert a numeric input to float.

    Missing values become NaN.
    """

    if value is None:
        return np.nan

    if isinstance(value, str):

        value = value.strip()

        if value == "":
            return np.nan

    try:
        return float(value)

    except (TypeError, ValueError) as exc:

        raise ValueError(
            f"Invalid numeric value: {value}"
        ) from exc


def normalize_gender(value: Any) -> Optional[str]:
    """
    Normalize gender values to the representation
    used by the training data.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    value_lower = value.lower()

    if value_lower in {"m", "male"}:
        return "m"

    if value_lower in {"f", "female"}:
        return "f"

    return value


def normalize_binary_string(
    value: Any,
) -> Optional[str]:
    """
    Normalize yes/no categorical values.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    value_lower = value.lower()

    if value_lower in {
        "yes",
        "y",
        "true",
        "1",
    }:
        return "yes"

    if value_lower in {
        "no",
        "n",
        "false",
        "0",
    }:
        return "no"

    return value


def normalize_relation(
    value: Any,
) -> Optional[str]:
    """
    Normalize relation values.

    The original dataset contains values such as:
    Parent, Self, Relative,
    Health care professional, and self.
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() == "self":
        return "Self"

    if value.lower() == "parent":
        return "Parent"

    if value.lower() == "relative":
        return "Relative"

    if value.lower() == "health care professional":
        return "Health care professional"

    return value


# ============================================================
# UNCERTAINTY
# ============================================================

def calculate_confidence(
    probability: float,
) -> float:
    """
    Binary confidence.

    Probability 0.5 -> confidence 0.
    Probability 0 or 1 -> confidence 1.
    """

    return float(
        abs(
            probability - 0.5
        )
        * 2.0
    )


def normalized_binary_entropy(
    probability: float,
) -> float:
    """
    Normalized binary entropy.

    0 -> completely certain
    1 -> maximum uncertainty
    """

    probability = float(
        np.clip(
            probability,
            1e-12,
            1.0 - 1e-12,
        )
    )

    entropy = -(
        probability * np.log2(probability)
        + (
            1.0 - probability
        )
        * np.log2(
            1.0 - probability
        )
    )

    return float(entropy)


def uncertainty_category(
    uncertainty: float,
) -> str:
    """
    Research-system uncertainty categories.
    """

    uncertainty = float(uncertainty)

    if uncertainty < 0.25:
        return "LOW"

    if uncertainty < 0.60:
        return "MODERATE"

    return "HIGH"


# ============================================================
# OOD DISTANCE
# ============================================================

def calculate_ood_distance(
    vector: np.ndarray,
    center: np.ndarray,
    mad: np.ndarray,
) -> float:
    """
    Robust standardized distance used by the
    registered OOD baseline.
    """

    vector = np.asarray(
        vector,
        dtype=float,
    )

    center = np.asarray(
        center,
        dtype=float,
    )

    mad = np.asarray(
        mad,
        dtype=float,
    )

    if (
        vector.shape
        != center.shape
        or center.shape
        != mad.shape
    ):
        raise ValueError(
            "OOD vector, center, and MAD "
            "dimensions do not match."
        )

    safe_mad = np.where(
        np.abs(mad) < 1e-12,
        1.0,
        mad,
    )

    standardized = np.abs(
        vector - center
    ) / safe_mad

    return float(
        np.mean(
            standardized
        )
    )


# ============================================================
# PREDICTION SERVICE
# ============================================================

class _AgeModelBundle:
    """One trained model + its preprocessing/OOD/SHAP artifacts, scoped
    to one age band. NEUROSCREEN AI has two of these (child, toddler)
    because the AQ-10-Child and Q-CHAT-10-Toddler instruments are
    different questionnaires for different developmental stages, not
    a single model spanning both -- see docs/architecture.md.
    """

    def __init__(
        self,
        registry_key: str,
        feature_columns: list[str],
        numeric_features: list[str],
        categorical_features: list[str],
        binary_columns: list[str],
        age_field_name: str,
        age_min_years: float,
        age_max_years: float,
        field_normalizers: Optional[Dict[str, Callable[[Any], Any]]] = None,
        excluded_features: Optional[list[str]] = None,
    ) -> None:

        self.registry_key = registry_key
        self.feature_columns = list(feature_columns)
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.binary_columns = list(binary_columns)
        self.age_field_name = age_field_name
        self.age_min_years = age_min_years
        self.age_max_years = age_max_years
        self.field_normalizers = field_normalizers or {}
        self.excluded_features = list(excluded_features or [])

        print(
            f"Initializing NEUROSCREEN AI "
            f"prediction service ({registry_key})..."
        )

        # ----------------------------------------------------
        # 1. Load registry
        # ----------------------------------------------------

        if not REGISTRY_PATH.exists():

            raise FileNotFoundError(
                "Model registry not found:\n"
                f"{REGISTRY_PATH}"
            )

        with open(
            REGISTRY_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            registry = json.load(file)

        # ----------------------------------------------------
        # 2. Validate active model
        #
        # Supports both the current multi-model registry
        # ("active_models": {"child": ..., "toddler": ...})
        # and the older single-model registry ("active_model": ...,
        # implicitly the child model) for backward compatibility.
        # ----------------------------------------------------

        active_models = registry.get("active_models")

        if active_models is not None:
            active_model = active_models.get(registry_key, {})
        elif registry_key == "child":
            active_model = registry.get("active_model", {})
        else:
            active_model = {}

        if not active_model:

            raise RuntimeError(
                f"No active '{registry_key}' model found in registry."
            )

        model_name = active_model.get(
            "name"
        )

        model_version = active_model.get(
            "version"
        )

        model_status = active_model.get(
            "status"
        )

        if not model_name:

            raise RuntimeError(
                "Active model does not contain "
                "a model name."
            )

        if not model_version:

            raise RuntimeError(
                "Active model does not contain "
                "a version."
            )

        # ----------------------------------------------------
        # 3. Resolve artifact path
        #
        # Current registry uses:
        #
        # "artifact": "models/artifacts/..."
        #
        # Backward compatibility:
        #
        # "artifact_path": "models/artifacts/..."
        # ----------------------------------------------------

        artifact_relative_path = (
            active_model.get("artifact")
            or active_model.get("artifact_path")
        )

        if not artifact_relative_path:

            raise RuntimeError(
                "Active model does not contain "
                "an artifact path.\n"
                f"Active model entry:\n"
                f"{active_model}"
            )

        artifact_path = (
            PROJECT_ROOT
            / artifact_relative_path
        ).resolve()

        if not artifact_path.exists():

            raise FileNotFoundError(
                "Model artifact not found.\n"
                f"Expected path:\n"
                f"{artifact_path}"
            )

        # ----------------------------------------------------
        # 4. Load artifact
        # ----------------------------------------------------

        print()
        print(
            "Loading registered model..."
        )

        print(
            f"Model name    : {model_name}"
        )

        print(
            f"Model version : {model_version}"
        )

        print(
            f"Model status  : {model_status}"
        )

        print(
            f"Artifact      : {artifact_path}"
        )

        artifact = joblib.load(
            artifact_path
        )

        if not isinstance(
            artifact,
            dict,
        ):

            raise RuntimeError(
                "Invalid model artifact format. "
                "Expected a dictionary."
            )

        # ----------------------------------------------------
        # 5. Extract model components
        # ----------------------------------------------------

        self.model = artifact.get(
            "model"
        )

        self.preprocessor = artifact.get(
            "preprocessor"
        )

        # ----------------------------------------------------
        # IMPORTANT FIX
        #
        # Current registered artifact contains:
        #
        #     feature_names
        #
        # Older artifact versions may contain:
        #
        #     feature_columns
        #
        # Prefer feature_names.
        # ----------------------------------------------------

        self.feature_columns = (
            artifact.get("feature_names")
            or artifact.get("feature_columns")
        )

        self.feature_names = (
            self.feature_columns
        )

        self.pipeline = artifact.get(
            "pipeline"
        )

        self.target_column = artifact.get(
            "target_column"
        )

        self.artifact_model_name = (
            artifact.get("model_name")
        )

        self.artifact_model_version = (
            artifact.get("model_version")
        )

        # ----------------------------------------------------
        # 6. Validate artifact
        # ----------------------------------------------------

        if self.model is None:

            raise RuntimeError(
                "Model artifact does not contain "
                "'model'."
            )

        if self.preprocessor is None:

            raise RuntimeError(
                "Model artifact does not contain "
                "'preprocessor'."
            )

        if not self.feature_columns:

            raise RuntimeError(
                "Model artifact does not contain "
                "'feature_names' or "
                "'feature_columns'."
            )

        # ----------------------------------------------------
        # 7. Validate feature schema
        # ----------------------------------------------------

        artifact_features = list(
            self.feature_columns
        )

        expected_features = list(
            feature_columns
        )

        if artifact_features != expected_features:

            raise RuntimeError(
                "Feature schema mismatch.\n\n"
                f"Expected:\n"
                f"{expected_features}\n\n"
                f"Artifact contains:\n"
                f"{artifact_features}"
            )

        # ----------------------------------------------------
        # 8. Validate artifact metadata
        # ----------------------------------------------------

        if (
            self.artifact_model_name
            and self.artifact_model_name
            != model_name
        ):

            raise RuntimeError(
                "Model name mismatch between "
                "registry and artifact.\n"
                f"Registry: {model_name}\n"
                f"Artifact: "
                f"{self.artifact_model_name}"
            )

        if (
            self.artifact_model_version
            and self.artifact_model_version
            != model_version
        ):

            raise RuntimeError(
                "Model version mismatch between "
                "registry and artifact.\n"
                f"Registry: {model_version}\n"
                f"Artifact: "
                f"{self.artifact_model_version}"
            )

        # ----------------------------------------------------
        # 9. OOD information
        # ----------------------------------------------------

        self.ood_center = artifact.get(
            "ood_center"
        )

        self.ood_mad = artifact.get(
            "ood_mad"
        )

        self.ood_threshold = artifact.get(
            "ood_threshold"
        )

        # ----------------------------------------------------
        # 10. SHAP background
        # ----------------------------------------------------

        self.shap_background = artifact.get(
            "shap_background"
        )

        self.transformed_feature_names = (
            artifact.get(
                "transformed_feature_names"
            )
        )

        # ----------------------------------------------------
        # 11. Additional metadata
        # ----------------------------------------------------

        self.transformed_feature_count = (
            artifact.get(
                "transformed_feature_count"
            )
        )

        self.training_samples = (
            artifact.get(
                "training_samples"
            )
        )

        self.holdout_samples = (
            artifact.get(
                "holdout_samples"
            )
        )

        self.cv_metrics = artifact.get(
            "cv_metrics"
        )

        self.holdout_metrics = artifact.get(
            "holdout_metrics"
        )

        # ----------------------------------------------------
        # 12. Save registry metadata
        # ----------------------------------------------------

        self.model_name = model_name

        self.model_version = model_version

        self.model_status = model_status

        self.artifact_path = artifact_path

        # ----------------------------------------------------
        # 13. Startup information
        # ----------------------------------------------------

        print()

        print(
            "Model loaded successfully."
        )

        print(
            f"Features      : "
            f"{len(self.feature_columns)}"
        )

        if (
            self.transformed_feature_count
            is not None
        ):

            print(
                f"Transformed   : "
                f"{self.transformed_feature_count}"
            )

        if self.ood_threshold is not None:

            print(
                f"OOD threshold : "
                f"{float(self.ood_threshold):.6f}"
            )

        else:

            print(
                "OOD detector  : Not available"
            )

        print()

        print(
            "NEUROSCREEN AI prediction "
            "service READY."
        )

        print()

    # ========================================================
    # INPUT PREPARATION
    # ========================================================

    def _prepare_input(
        self,
        patient_data: Dict[str, Any],
    ) -> pd.DataFrame:

        """
        Convert API/user input into the exact raw feature schema
        expected by this bundle's trained preprocessing pipeline.

        Generic across bundles: numeric features are coerced to float
        (binary_columns additionally validated as strictly 0/1),
        categorical features are stripped strings, and any bundle may
        supply per-field overrides (field_normalizers) for more
        forgiving input handling (e.g. "Male" -> "m").
        """

        data = dict(
            patient_data
        )

        row: Dict[str, Any] = {}

        for feature in self.feature_columns:
            row[feature] = data.get(feature)

        for feature in self.numeric_features:

            value = row[feature]

            if feature in self.binary_columns:

                if value is None:
                    row[feature] = np.nan
                    continue

                try:
                    value = int(float(value))
                    if value not in (0, 1):
                        raise ValueError(f"{feature} must be 0 or 1.")
                    row[feature] = value
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid value for {feature}: {row[feature]}"
                    ) from exc

            else:
                normalizer = self.field_normalizers.get(feature, normalize_numeric)
                row[feature] = normalizer(value)

        for feature in self.categorical_features:

            value = row[feature]
            normalizer = self.field_normalizers.get(feature)

            if normalizer is not None:
                row[feature] = normalizer(value)
            else:
                row[feature] = None if value is None else str(value).strip()

        df = pd.DataFrame(
            [row],
            columns=self.feature_columns,
        )

        return df

    # ========================================================
    # TRANSFORMATION
    # ========================================================

    def _transform(
        self,
        df: pd.DataFrame,
    ) -> Any:

        """
        Transform raw input using the exact
        fitted preprocessing object stored
        in the registered artifact.
        """

        return self.preprocessor.transform(
            df
        )

    # ========================================================
    # PROBABILITY
    # ========================================================

    def _get_probability(
        self,
        X: Any,
    ) -> float:

        """
        Return probability of positive class YES.
        """

        if not hasattr(
            self.model,
            "predict_proba",
        ):

            raise RuntimeError(
                "Loaded model does not support "
                "predict_proba()."
            )

        probabilities = (
            self.model.predict_proba(X)
        )

        classes = list(
            self.model.classes_
        )

        if 1 in classes:

            positive_index = classes.index(
                1
            )

        elif "YES" in classes:

            positive_index = classes.index(
                "YES"
            )

        elif "yes" in classes:

            positive_index = classes.index(
                "yes"
            )

        else:

            raise RuntimeError(
                "Unable to identify positive "
                "class in trained model."
            )

        return float(
            probabilities[
                0,
                positive_index,
            ]
        )

    # ========================================================
    # OOD
    # ========================================================

    def _is_age_out_of_training_range(
        self,
        age: Optional[float],
    ) -> bool:
        if age is None:
            return False
        try:
            age_value = float(age)
        except (TypeError, ValueError):
            return False
        return not (
            self.age_min_years <= age_value <= self.age_max_years
        )

    def _calculate_ood(
        self,
        X: Any,
        age: Optional[float] = None,
    ) -> Dict[str, Any]:

        """
        Calculate OOD information.

        This is a research-system distribution
        check and not a clinical determination.

        Combines two independent signals:
        - the Mahalanobis-style distance check against the transformed
          training feature distribution (below);
        - an explicit age-range check, because a single numeric feature
          like age gets diluted across 86 post-encoding dimensions and
          the distance check alone does not reliably flag an age far
          outside the training range (verified empirically: an age of
          0.25 years scored well under threshold).
        """

        age_out_of_range = self._is_age_out_of_training_range(age)

        if (
            self.ood_center is None
            or self.ood_mad is None
            or self.ood_threshold is None
        ):

            return {
                "status": (
                    "OUT_OF_DISTRIBUTION"
                    if age_out_of_range
                    else "OOD_NOT_AVAILABLE"
                ),
                "distance": None,
                "threshold": None,
                "is_ood": age_out_of_range,
                "age_out_of_range": age_out_of_range,
            }

        # ----------------------------------------------------
        # Convert sparse matrix to dense
        # ----------------------------------------------------

        if hasattr(
            X,
            "toarray",
        ):

            X_dense = X.toarray()

        else:

            X_dense = np.asarray(X)

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        center = np.asarray(
            self.ood_center
        )

        mad = np.asarray(
            self.ood_mad
        )

        if X_dense.shape[1] != len(
            center
        ):

            raise RuntimeError(
                "OOD baseline dimension does "
                "not match transformed "
                "feature dimension.\n"
                f"Input dimension: "
                f"{X_dense.shape[1]}\n"
                f"OOD dimension: "
                f"{len(center)}"
            )

        # ----------------------------------------------------
        # Calculate distance
        # ----------------------------------------------------

        distance = calculate_ood_distance(
            X_dense[0],
            center,
            mad,
        )

        threshold = float(
            self.ood_threshold
        )

        is_ood = (
            distance > threshold
        ) or age_out_of_range

        return {
            "status": (
                "OUT_OF_DISTRIBUTION"
                if is_ood
                else "IN_DISTRIBUTION"
            ),
            "distance": float(
                distance
            ),
            "threshold": threshold,
            "is_ood": bool(
                is_ood
            ),
            "age_out_of_range": age_out_of_range,
        }

    # ========================================================
    # DECISION ENGINE
    # ========================================================

    def _make_decision(
        self,
        probability: float,
        uncertainty: float,
        ood_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        """
        Research-system workflow decision.

        These rules are not medical triage
        rules and do not constitute diagnosis.
        """

        confidence = (
            calculate_confidence(
                probability
            )
        )

        category = (
            uncertainty_category(
                uncertainty
            )
        )

        # ----------------------------------------------------
        # OOD has highest workflow priority
        # ----------------------------------------------------

        if ood_result["is_ood"]:

            if ood_result.get("age_out_of_range"):
                band_label = {
                    "toddler": "the toddler model (Q-CHAT-10, ",
                    "child": "the child model (AQ-10, ",
                }.get(self.registry_key, "this model (")
                reason = (
                    f"The entered age is outside the training range of "
                    f"{band_label}{self.age_min_years:g}-{self.age_max_years:g} years); "
                    "there are no examples to learn from at this age, "
                    "so this prediction should not be trusted."
                )
            else:
                reason = (
                    "Input appears outside "
                    "the reference model "
                    "distribution."
                )

            return {
                "status": "OOD_REVIEW",
                "reason": reason,
                "uncertainty_category": category,
                "confidence": confidence,
            }

        # ----------------------------------------------------
        # High uncertainty
        # ----------------------------------------------------

        if category == "HIGH":

            return {
                "status": "HUMAN_REVIEW",
                "reason": (
                    "Model uncertainty is high."
                ),
                "uncertainty_category": category,
                "confidence": confidence,
            }

        # ----------------------------------------------------
        # Normal research workflow
        # ----------------------------------------------------

        return {
            "status": "SCREENING_RESULT",
            "reason": (
                "Prediction passed the "
                "configured uncertainty "
                "and distribution checks."
            ),
            "uncertainty_category": category,
            "confidence": confidence,
        }

    # ========================================================
    # SHAP
    # ========================================================

    def _get_shap_explanation(
        self,
        X: Any,
    ) -> Optional[Dict[str, Any]]:

        """
        Generate SHAP explanation.

        SHAP values represent model-output
        contributions. They are not causal
        or clinical explanations.
        """

        try:

            import shap

        except ImportError:

            return {
                "available": False,
                "reason": (
                    "SHAP package is not installed."
                ),
            }

        # ----------------------------------------------------
        # Background data
        # ----------------------------------------------------

        if self.shap_background is None:

            return {
                "available": False,
                "reason": (
                    "SHAP background data was "
                    "not stored in the model "
                    "artifact."
                ),
            }

        try:

            # ------------------------------------------------
            # Convert input to dense
            # ------------------------------------------------

            if hasattr(
                X,
                "toarray",
            ):

                X_dense = X.toarray()

            else:

                X_dense = np.asarray(X)

            background = np.asarray(
                self.shap_background
            )

            # ------------------------------------------------
            # SHAP masker
            # ------------------------------------------------

            masker = (
                shap.maskers.Independent(
                    background,
                    max_samples=len(
                        background
                    ),
                )
            )

            # ------------------------------------------------
            # Linear explainer
            # ------------------------------------------------

            explainer = (
                shap.LinearExplainer(
                    self.model,
                    masker,
                )
            )

            shap_values = explainer(
                X_dense
            )

            values = shap_values.values

            if values.ndim == 3:

                values = values[:, :, 0]

            values = np.asarray(
                values[0]
            )

            # ------------------------------------------------
            # Feature names
            # ------------------------------------------------

            if (
                self.transformed_feature_names
                is not None
            ):

                names = list(
                    self.transformed_feature_names
                )

            else:

                names = [
                    f"feature_{i}"
                    for i in range(
                        len(values)
                    )
                ]

            # ------------------------------------------------
            # Validate SHAP dimensions
            # ------------------------------------------------

            if len(names) != len(values):

                return {
                    "available": False,
                    "reason": (
                        "SHAP feature-name "
                        "dimension does not "
                        "match SHAP value dimension."
                    ),
                }

            # ------------------------------------------------
            # Rank contributions
            # ------------------------------------------------

            contributions = []

            for name, value in zip(
                names,
                values,
            ):

                if value > 0:

                    direction = "POSITIVE"

                elif value < 0:

                    direction = "NEGATIVE"

                else:

                    direction = "NEUTRAL"

                contributions.append(
                    {
                        "feature": name,
                        "shap_value": float(
                            value
                        ),
                        "direction": direction,
                    }
                )

            contributions.sort(
                key=lambda item: abs(
                    item["shap_value"]
                ),
                reverse=True,
            )

            return {
                "available": True,
                "output_scale": (
                    "model_output_margin"
                ),
                "note": (
                    "SHAP values represent "
                    "model-output contributions "
                    "and are not causal or "
                    "clinical explanations."
                ),
                "top_contributions": (
                    contributions[:10]
                ),
            }

        except Exception as exc:

            return {
                "available": False,
                "reason": (
                    "SHAP explanation failed."
                ),
                "error": str(exc),
            }

    # ========================================================
    # MAIN PREDICTION
    # ========================================================

    def predict(
        self,
        patient_data: Dict[str, Any],
        age_years: Optional[float] = None,
    ) -> Dict[str, Any]:

        """
        Run the complete research prediction pipeline.

        age_years overrides the age used for the training-range check
        when the caller already knows it in years (the public API
        always takes age in years; this bundle's own age_field_name
        may be a different unit, e.g. Age_Mons for the toddler model).
        Defaults to patient_data[self.age_field_name] when omitted.
        """

        if age_years is None:
            age_years = patient_data.get(self.age_field_name)


        # ----------------------------------------------------
        # 1. Prepare input
        # ----------------------------------------------------

        df = self._prepare_input(
            patient_data
        )

        # ----------------------------------------------------
        # 2. Transform
        # ----------------------------------------------------

        X = self._transform(
            df
        )

        # ----------------------------------------------------
        # 3. Positive probability
        # ----------------------------------------------------

        probability_yes = (
            self._get_probability(
                X
            )
        )

        probability_yes = float(
            np.clip(
                probability_yes,
                0.0,
                1.0,
            )
        )

        probability_no = float(
            1.0 - probability_yes
        )

        # ----------------------------------------------------
        # 4. Prediction
        # ----------------------------------------------------

        prediction = (
            "YES"
            if probability_yes >= 0.5
            else "NO"
        )

        # ----------------------------------------------------
        # 5. Confidence
        # ----------------------------------------------------

        confidence = (
            calculate_confidence(
                probability_yes
            )
        )

        # ----------------------------------------------------
        # 6. Uncertainty
        # ----------------------------------------------------

        uncertainty = (
            normalized_binary_entropy(
                probability_yes
            )
        )

        uncertainty_level = (
            uncertainty_category(
                uncertainty
            )
        )

        # ----------------------------------------------------
        # 7. OOD
        # ----------------------------------------------------

        ood_result = (
            self._calculate_ood(
                X,
                age=age_years,
            )
        )

        # ----------------------------------------------------
        # 8. Decision
        # ----------------------------------------------------

        decision = (
            self._make_decision(
                probability_yes,
                uncertainty,
                ood_result,
            )
        )

        # ----------------------------------------------------
        # 9. SHAP
        # ----------------------------------------------------

        shap_result = (
            self._get_shap_explanation(
                X
            )
        )

        # ----------------------------------------------------
        # 10. Final result
        # ----------------------------------------------------

        result = {

            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "status": self.model_status,
                "age_band": self.registry_key,
            },

            "prediction": {
                "label": prediction,
                "probability_yes": (
                    probability_yes
                ),
                "probability_no": (
                    probability_no
                ),
            },

            "uncertainty": {
                "confidence": (
                    confidence
                ),
                "uncertainty": (
                    uncertainty
                ),
                "category": (
                    uncertainty_level
                ),
            },

            "distribution": (
                ood_result
            ),

            "decision": (
                decision
            ),

            "explanation": (
                shap_result
            ),

            "metadata": {
                "feature_count": len(
                    self.feature_columns
                ),
                "feature_names": (
                    self.feature_columns
                ),
                "excluded_features": (
                    self.excluded_features
                ),
                "transformed_feature_count": (
                    self.transformed_feature_count
                ),
            },

            "disclaimer": (
                "This is a research/educational "
                "screening prototype and not a "
                "diagnostic medical system. "
                "Model probabilities are not "
                "clinical certainty."
            ),
        }

        return result


# ============================================================
# CHILD BUNDLE FIELD NORMALIZERS
# ============================================================

CHILD_FIELD_NORMALIZERS: Dict[str, Callable[[Any], Any]] = {
    "gender": normalize_gender,
    "jundice": normalize_binary_string,
    "austim": normalize_binary_string,
    "used_app_before": normalize_binary_string,
    "relation": normalize_relation,
}


# ============================================================
# TODDLER BUNDLE FIELD NORMALIZERS + INPUT TRANSLATION
# ============================================================

def _normalize_test_taker(value: Any) -> Optional[str]:
    """Same forgiving pattern as normalize_relation, for the toddler
    dataset's own "Who completed the test" categories."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    mapping = {
        "family member": "Family member",
        "parent": "Family member",
        "relative": "Family member",
        "health care professional": "Health care professional",
        "self": "Self",
    }
    return mapping.get(value.lower(), "Others")


TODDLER_FIELD_NORMALIZERS: Dict[str, Callable[[Any], Any]] = {
    "Sex": normalize_gender,
    "Jaundice": normalize_binary_string,
    "Family_mem_with_ASD": normalize_binary_string,
    "Who completed the test": _normalize_test_taker,
}


def _map_unified_input_to_toddler(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the app's unified request shape (AssessmentRequest --
    A1_Score.._A10_Score, age in years, gender, ethnicity, jundice,
    austim, contry_of_res, used_app_before, relation) into the toddler
    bundle's own native column names (A1..A10, Age_Mons, Sex,
    Ethnicity, Jaundice, Family_mem_with_ASD, "Who completed the test").

    Ethnicity is passed through as-is: the two source datasets use
    different category spellings/casing for ethnicity (e.g.
    "White-European" vs "White European"), so a value that doesn't
    exactly match one of the toddler dataset's known categories is
    encoded as "unknown" by the fitted OneHotEncoder(handle_unknown=
    "ignore") rather than raising -- a real precision limitation for
    that one field, not a crash.
    """

    mapped: Dict[str, Any] = {}

    for i in range(1, 11):
        mapped[f"A{i}"] = patient_data.get(f"A{i}_Score")

    age_years = patient_data.get("age")
    try:
        mapped["Age_Mons"] = round(float(age_years) * 12) if age_years is not None else None
    except (TypeError, ValueError):
        mapped["Age_Mons"] = None

    mapped["Sex"] = patient_data.get("gender")
    mapped["Ethnicity"] = patient_data.get("ethnicity")
    mapped["Jaundice"] = patient_data.get("jundice")
    mapped["Family_mem_with_ASD"] = patient_data.get("austim")
    mapped["Who completed the test"] = patient_data.get("relation")

    return mapped


# ============================================================
# PUBLIC ROUTER
# ============================================================

class PredictionService:
    """Public singleton: routes each request to the age-appropriate
    model bundle (toddler 1-3yr / child 4-11yr) and returns the same
    result shape either way. See _AgeModelBundle for the actual
    per-model prediction logic.
    """

    def __init__(self) -> None:
        child_score_columns = [f"A{i}_Score" for i in range(1, 11)]

        self.child = _AgeModelBundle(
            registry_key="child",
            feature_columns=APPROVED_FEATURES,
            numeric_features=[*child_score_columns, "age"],
            categorical_features=["gender", "ethnicity", "jundice", "austim", "contry_of_res", "used_app_before", "relation"],
            binary_columns=child_score_columns,
            age_field_name="age",
            age_min_years=TRAINING_AGE_MIN_YEARS,
            age_max_years=TRAINING_AGE_MAX_YEARS,
            field_normalizers=CHILD_FIELD_NORMALIZERS,
            excluded_features=EXCLUDED_FEATURES,
        )

        self.toddler = _AgeModelBundle(
            registry_key="toddler",
            feature_columns=TODDLER_FEATURE_NAMES,
            numeric_features=TODDLER_NUMERIC_FEATURES,
            categorical_features=TODDLER_CATEGORICAL_FEATURES,
            binary_columns=TODDLER_SCORE_COLUMNS,
            age_field_name="Age_Mons",
            age_min_years=TODDLER_AGE_MIN_YEARS,
            age_max_years=TODDLER_AGE_MAX_YEARS,
            field_normalizers=TODDLER_FIELD_NORMALIZERS,
            excluded_features=["Case_No", "Qchat-10-Score"],
        )

        # Kept for the one remaining external reader (history persistence
        # falls back to this if a response ever lacks its own "model" key).
        self.model_name = self.child.model_name
        self.model_version = self.child.model_version
        self.model_status = self.child.model_status

    def _select_bundle(self, age_years: Optional[float]) -> _AgeModelBundle:
        if age_years is None:
            return self.child

        if self.toddler.age_min_years <= age_years <= self.toddler.age_max_years:
            return self.toddler

        if self.child.age_min_years <= age_years <= self.child.age_max_years:
            return self.child

        # Outside both bands: still produce a prediction (age_out_of_range
        # will correctly be True regardless of which bundle answers) using
        # whichever band's range is numerically closer.
        distance_to_toddler = min(
            abs(age_years - self.toddler.age_min_years),
            abs(age_years - self.toddler.age_max_years),
        )
        distance_to_child = min(
            abs(age_years - self.child.age_min_years),
            abs(age_years - self.child.age_max_years),
        )
        return self.toddler if distance_to_toddler <= distance_to_child else self.child

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        raw_age = patient_data.get("age")
        try:
            age_years = float(raw_age) if raw_age is not None else None
        except (TypeError, ValueError):
            age_years = None

        bundle = self._select_bundle(age_years)

        if bundle is self.toddler:
            mapped = _map_unified_input_to_toddler(patient_data)
            return bundle.predict(mapped, age_years=age_years)

        return bundle.predict(patient_data, age_years=age_years)


# ============================================================
# DEMO INPUT
# ============================================================

DEMO_INPUT = {

    "A1_Score": 1,
    "A2_Score": 1,
    "A3_Score": 1,
    "A4_Score": 1,
    "A5_Score": 1,
    "A6_Score": 1,
    "A7_Score": 1,
    "A8_Score": 1,
    "A9_Score": 1,
    "A10_Score": 1,

    "age": 8,

    "gender": "m",

    "ethnicity": "White-European",

    "jundice": "no",

    "austim": "no",

    "contry_of_res": "United States",

    "used_app_before": "no",

    "relation": "Parent",
}


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "NEUROSCREEN AI - "
        "MODEL PREDICTION TEST"
    )

    print(
        "=" * 70
    )

    print()

    service = PredictionService()

    result = service.predict(
        DEMO_INPUT
    )

    print(
        "=" * 70
    )

    print(
        "PREDICTION RESULT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Prediction       : "
        f"{result['prediction']['label']}"
    )

    print(
        f"P(YES)           : "
        f"{result['prediction']['probability_yes']:.6f}"
    )

    print(
        f"P(NO)            : "
        f"{result['prediction']['probability_no']:.6f}"
    )

    print(
        f"Confidence       : "
        f"{result['uncertainty']['confidence']:.6f}"
    )

    print(
        f"Uncertainty      : "
        f"{result['uncertainty']['uncertainty']:.6f}"
    )

    print(
        f"Uncertainty level: "
        f"{result['uncertainty']['category']}"
    )

    print(
        f"OOD status       : "
        f"{result['distribution']['status']}"
    )

    if (
        result["distribution"]["distance"]
        is not None
    ):

        print(
            f"OOD distance     : "
            f"{result['distribution']['distance']:.6f}"
        )

        print(
            f"OOD threshold    : "
            f"{result['distribution']['threshold']:.6f}"
        )

    print(
        f"Decision status  : "
        f"{result['decision']['status']}"
    )

    print(
        f"Decision reason  : "
        f"{result['decision']['reason']}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explanation = (
        result["explanation"]
    )

    print()

    if explanation.get(
        "available"
    ):

        print(
            "=" * 70
        )

        print(
            "TOP SHAP CONTRIBUTIONS"
        )

        print(
            "=" * 70
        )

        for item in explanation[
            "top_contributions"
        ][:10]:

            print(
                f"{item['feature']:40s} "
                f"{item['shap_value']:+.6f} "
                f"{item['direction']}"
            )

    else:

        print(
            "SHAP explanation unavailable:"
        )

        print(
            explanation.get(
                "reason",
                "Unknown reason",
            )
        )

    print()

    print(
        "=" * 70
    )

    print(
        "TEST COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# SHARED PREDICTION SERVICE INSTANCE
# ============================================================

prediction_service = (
    PredictionService()
)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()