from __future__ import annotations

from backend.app.agents.nodes.classifier import classifier_node
from backend.app.agents.nodes.evidence import evidence_node
from backend.app.agents.nodes.final_audit import final_audit_node
from backend.app.agents.nodes.groundedness import groundedness_node
from backend.app.agents.nodes.response import response_node
from backend.app.agents.nodes.router import routing_node
from backend.app.agents.nodes.safety import safety_node
from backend.app.schemas.assessment import AssessmentResponse
from backend.app.tests.conftest import SAMPLE_ASSESSMENT_REQUEST


def _run_pipeline_without_rag_or_llm(patient_data: dict, llm_explanation: str, evidence: list[dict]) -> dict:
    """Exercises every node except retrieval/reranker/llm_response, which
    require a running Ollama server. This still covers the full safety,
    ML, evidence, groundedness, and audit chain that the API depends on."""

    state = {"patient_data": patient_data, "query": "Explain the screening result.", "errors": []}
    state.update(safety_node(state))
    state.update(classifier_node(state))
    state.update(routing_node(state))

    state["retrieved_evidence"] = evidence
    state["citations"] = []
    state.update(evidence_node(state))

    state["response"] = {"llm_explanation": llm_explanation, "generator": "test"}
    state.update(groundedness_node(state))
    state.update(final_audit_node(state))
    state.update(response_node(state))
    return state["response"]


def test_pipeline_produces_schema_valid_response_when_grounded():
    evidence = [
        {
            "chunk_id": "chunk_1",
            "source": "knowledge_base/guidelines/CDC_autism_screening.pdf",
            "file_name": "CDC_autism_screening.pdf",
            "text": "A positive screening result should be followed by a thorough assessment.",
            "similarity": 0.9,
        }
    ]
    response = _run_pipeline_without_rag_or_llm(
        SAMPLE_ASSESSMENT_REQUEST,
        "A positive screening result should be followed by a thorough assessment. [E1]",
        evidence,
    )

    # Must validate against the exact schema the API promises callers.
    validated = AssessmentResponse.model_validate(response)
    assert validated.route in ("STANDARD_REVIEW", "HUMAN_REVIEW", "OOD_REVIEW")
    assert validated.prediction.label in ("YES", "NO")


def test_pipeline_falls_back_safely_when_llm_output_is_unsafe():
    evidence = [
        {
            "chunk_id": "chunk_1",
            "source": "cdc.pdf",
            "file_name": "cdc.pdf",
            "text": "Developmental screening does not provide a medical diagnosis.",
            "similarity": 0.9,
        }
    ]
    unsafe_explanation = "The child definitely has autism based on this screening. [E1]"

    response = _run_pipeline_without_rag_or_llm(SAMPLE_ASSESSMENT_REQUEST, unsafe_explanation, evidence)

    validated = AssessmentResponse.model_validate(response)
    # groundedness_node substitutes an evidence-grounded fallback before
    # final_audit even runs (the unsafe text never reaches the user), so
    # final_audit's own checks correctly PASS against that safe text —
    # `fallback_used` is what actually tells the caller a fallback fired.
    assert validated.fallback_used is True
    assert "definitely has autism" not in validated.llm_explanation
    assert validated.audit.final_status == "PASSED"
    assert validated.groundedness_passed is True


def test_pipeline_handles_missing_patient_data():
    response = _run_pipeline_without_rag_or_llm({}, "", [])
    assert response["safety_status"] == "BLOCKED"
