from __future__ import annotations

from backend.app.rag.evaluation.groundedness import (
    CLAIM_SUPPORT_THRESHOLD,
    DEFAULT_SUPPORT_THRESHOLD,
    build_evidence_map,
    evaluate_claim,
    evaluate_groundedness,
    extract_citation_ids,
    lexical_support_score,
    sentence_split,
    support_score,
)


def test_public_api_still_exported():
    """citation_guard.py depends on these names and this exact signature."""
    assert DEFAULT_SUPPORT_THRESHOLD == 0.35
    assert CLAIM_SUPPORT_THRESHOLD == DEFAULT_SUPPORT_THRESHOLD
    assert support_score is lexical_support_score


def test_lexical_support_score_full_overlap():
    claim = "Developmental screening does not provide a diagnosis"
    evidence = "Developmental screening does not provide a diagnosis of any condition"
    assert lexical_support_score(claim, evidence) > 0.9


def test_lexical_support_score_no_overlap():
    assert lexical_support_score("cats are mammals", "the stock market fell today") == 0.0


def test_sentence_split_preserves_citations():
    text = "Screening does not diagnose. [E2] Early support helps. [E5]"
    sentences = sentence_split(text)
    assert len(sentences) == 2
    assert "[E2]" in sentences[0]
    assert "[E5]" in sentences[1]


def test_extract_citation_ids_dedupes_and_uppercases():
    ids = extract_citation_ids("See [e1] and [E2], also [e1] again.")
    assert ids == ["[E1]", "[E2]"]


def test_build_evidence_map_indexes_by_bracketed_and_bare_id():
    citations = [{"citation_id": "[E1]", "text": "some evidence text"}]
    evidence_map = build_evidence_map(citations)
    assert evidence_map["[E1]"] == "some evidence text"
    assert evidence_map["E1"] == "some evidence text"


def test_evaluate_claim_supported_above_threshold():
    evidence_map = {"[E1]": "Early identification allows earlier intervention for children."}
    result = evaluate_claim(
        claim="Early identification allows earlier intervention.",
        citation_ids=["[E1]"],
        evidence_map=evidence_map,
        threshold=DEFAULT_SUPPORT_THRESHOLD,
    )
    assert result.supported is True
    assert result.matched_evidence == ["[E1]"]


def test_evaluate_groundedness_every_sentence_counts_as_a_claim():
    """Uncited sentences must not be silently skipped from claim coverage."""

    citations = [{"citation_id": "[E1]", "text": "Screening tools do not provide a diagnosis."}]
    answer = (
        "Screening tools do not provide a diagnosis. [E1] "
        "This test guarantees an accurate result for every child."
    )

    result = evaluate_groundedness(answer=answer, citations=citations, evidence=citations)

    assert result["total_claims"] == 2
    assert result["supported_claims"] == 1
    assert result["unsupported_claims"] == 1
    assert result["citation_coverage"] == 0.5


def test_evaluate_groundedness_empty_answer_returns_zeroed_result():
    result = evaluate_groundedness(answer="", citations=[])
    assert result["total_claims"] == 0
    assert result["groundedness_score"] == 0.0


def test_evaluate_groundedness_rejects_bad_threshold():
    import pytest

    with pytest.raises(ValueError):
        evaluate_groundedness(answer="text", citations=[], threshold=1.5)

    with pytest.raises(TypeError):
        evaluate_groundedness(answer="text", citations=[], threshold=True)
