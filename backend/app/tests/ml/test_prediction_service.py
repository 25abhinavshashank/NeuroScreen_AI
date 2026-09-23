from __future__ import annotations

from backend.app.ml.prediction_service import DEMO_INPUT, prediction_service


def test_predict_returns_expected_top_level_shape():
    result = prediction_service.predict(DEMO_INPUT)

    for key in ("model", "prediction", "uncertainty", "distribution", "decision", "explanation", "metadata", "disclaimer"):
        assert key in result


def test_predict_probability_is_valid():
    result = prediction_service.predict(DEMO_INPUT)
    prediction = result["prediction"]

    assert prediction["label"] in ("YES", "NO")
    assert 0.0 <= prediction["probability_yes"] <= 1.0
    assert abs(prediction["probability_yes"] + prediction["probability_no"] - 1.0) < 1e-6


def test_predict_uncertainty_category_is_known_value():
    result = prediction_service.predict(DEMO_INPUT)
    assert result["uncertainty"]["category"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_distribution_has_ood_flag():
    result = prediction_service.predict(DEMO_INPUT)
    distribution = result["distribution"]
    assert "is_ood" in distribution
    assert isinstance(distribution["is_ood"], bool)


def test_predict_missing_scores_does_not_crash():
    incomplete_input = {**DEMO_INPUT}
    del incomplete_input["A1_Score"]
    result = prediction_service.predict(incomplete_input)
    assert result["prediction"]["label"] in ("YES", "NO")


def test_age_within_child_training_range_is_not_flagged():
    """Child model (Autism-Child-Data.arff) covers ages 4-11 only."""
    for age in (4, 8, 11):
        payload = {**DEMO_INPUT, "age": age}
        result = prediction_service.predict(payload)
        assert result["distribution"]["age_out_of_range"] is False
        assert result["distribution"]["is_ood"] is False
        assert result["model"]["age_band"] == "child"


def test_age_within_toddler_training_range_routes_to_toddler_model():
    """Toddler model (Toddler-Autism-Data.csv) covers ages 1-3 (12-36
    months). Routing must translate the unified request shape into the
    toddler bundle's own native column names (A1_Score -> A1, etc)."""
    for age in (1, 2, 3):
        payload = {**DEMO_INPUT, "age": age}
        result = prediction_service.predict(payload)
        assert result["distribution"]["age_out_of_range"] is False
        assert result["distribution"]["is_ood"] is False
        assert result["model"]["age_band"] == "toddler"
        assert result["model"]["name"] == "neuro_screen_logistic_regression_toddler"


def test_age_outside_both_training_ranges_is_flagged_and_routed_for_review():
    """A single numeric feature like age is diluted across dozens of
    post-encoding dimensions, so the generic Mahalanobis OOD distance
    alone does not reliably catch an out-of-range age (verified
    empirically before this explicit check was added: age=0.25 scored
    well under threshold). Covers below-toddler, the 3-4yr gap between
    the two models, and above-child."""
    for age in (0.25, 3.9, 11.1, 15):
        payload = {**DEMO_INPUT, "age": age}
        result = prediction_service.predict(payload)
        assert result["distribution"]["age_out_of_range"] is True
        assert result["distribution"]["is_ood"] is True
        assert result["decision"]["status"] == "OOD_REVIEW"
        assert "training range" in result["decision"]["reason"]
