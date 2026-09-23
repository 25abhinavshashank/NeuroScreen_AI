from __future__ import annotations

from backend.app.agents.nodes.llm_response import _clean

EVIDENCE = [
    {"citation_id": "E1", "text": "foo"},
    {"citation_id": "E2", "text": "bar"},
    {"citation_id": "E3", "text": "baz"},
]


def test_clean_normalizes_bare_number_citations_to_e_prefix():
    """Regression test: gemma2:2b intermittently writes [1] instead of
    [E1] despite the evidence block and prompt both showing [E1] --
    confirmed by direct trace as the actual cause of a ~94% groundedness
    fallback rate, since every citation regex downstream requires the
    E prefix and silently treats [1] as no citation at all."""

    raw = "Screening tools should be followed by assessment. [1] Second claim here. [2]"
    cleaned = _clean(raw, evidence=EVIDENCE)

    assert "[E1]" in cleaned
    assert "[E2]" in cleaned
    assert "[1]" not in cleaned.replace("[E1]", "")


def test_clean_does_not_invent_citations_for_unknown_evidence():
    raw = "A claim citing evidence that doesn't exist. [99]"
    cleaned = _clean(raw, evidence=EVIDENCE)

    assert "[99]" in cleaned
    assert "[E99]" not in cleaned


def test_clean_leaves_correctly_formatted_citations_untouched():
    raw = "Already correct. [E2]"
    assert _clean(raw, evidence=EVIDENCE) == "Already correct. [E2]"


def test_clean_handles_no_evidence_without_crashing():
    raw = "Some text with [1] in it."
    assert _clean(raw, evidence=[]) == raw
    assert _clean(raw, evidence=None) == raw
