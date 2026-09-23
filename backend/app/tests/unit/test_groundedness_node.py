from __future__ import annotations

from backend.app.agents.nodes.groundedness import groundedness_node


def _citation(cid: str, text: str, source: str = "source.pdf") -> dict:
    return {"citation_id": cid, "text": text, "source": source}


def test_enriched_result_has_thresholds_and_cited_claims_on_success_path():
    citations = [_citation("[E1]", "A positive screening result should be followed by a thorough assessment.")]
    answer = "A positive screening result should be followed by a thorough assessment. [E1]"

    state = {
        "response": {"llm_explanation": answer},
        "citations": citations,
        "retrieved_evidence": citations,
    }

    out = groundedness_node(state)
    g = out["groundedness"]

    assert "thresholds" in g
    assert "cited_claims" in g
    assert g["passed"] is True
    assert out["groundedness_passed"] is True


def test_fallback_quotes_evidence_directly_and_passes():
    """The deterministic fallback must itself be groundable: it should
    quote retrieved evidence verbatim rather than use a generic template
    with no guaranteed lexical overlap with whatever was retrieved."""

    citations = [_citation("[E1]", "Only qualified clinicians can determine developmental outcomes.")]
    answer = (
        "Only qualified clinicians can determine developmental outcomes. [E1] "
        "This app is a certified medical diagnosis you can rely on."
    )

    state = {
        "response": {"llm_explanation": answer},
        "citations": citations,
        "retrieved_evidence": citations,
    }

    out = groundedness_node(state)

    assert out["groundedness_passed"] is True
    assert out["response"]["generator"] == "deterministic_grounded_fallback"
    assert "[E1]" in out["response"]["llm_explanation"]


def test_empty_result_has_consistent_shape():
    out = groundedness_node({"response": {}, "citations": [], "retrieved_evidence": []})
    g = out["groundedness"]
    assert g["cited_claims"] == 0
    assert "thresholds" in g
    assert out["groundedness_passed"] is False
