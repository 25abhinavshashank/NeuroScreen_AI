from __future__ import annotations

from backend.app.agents.graph import neuroscreen_graph


import sys


def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 70)
    print("NEUROSCREEN AI - LANGGRAPH TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Sample assessment input
    # ------------------------------------------------------------------
    #
    # This is ONLY a software/integration test input.
    # It is not a real patient's assessment.
    #
    sample_patient_data = {
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

    # ------------------------------------------------------------------
    # Initial LangGraph state
    # ------------------------------------------------------------------

    initial_state = {
        "patient_data": sample_patient_data,
        "query": (
            "Explain the screening result and provide "
            "evidence-based context."
        ),
        "errors": [],
    }

    print()
    print("Starting LangGraph...")

    # ------------------------------------------------------------------
    # Execute graph
    # ------------------------------------------------------------------

    try:
        final_state = dict(initial_state)
        for output in neuroscreen_graph.stream(initial_state):
            for node_name, node_state in output.items():
                print(f"\n>>> NODE COMPLETED: {node_name}")
                if "response" in node_state:
                    print(f"    response keys: {list(node_state['response'].keys())}")
                    print(f"    response['llm_explanation']: {node_state['response'].get('llm_explanation')!r}")
                    print(f"    response['generator']: {node_state['response'].get('generator')!r}")
                if "groundedness" in node_state:
                    g = node_state["groundedness"]
                    if isinstance(g, dict):
                        print(f"    groundedness keys: {list(g.keys())}")
                        print(f"    groundedness passed: {g.get('passed')}, score: {g.get('groundedness_score')}")
                if "errors" in node_state:
                    print(f"    errors: {node_state['errors']}")
                final_state.update(node_state)
    except Exception as exc:
        print()
        print("=" * 70)
        print("GRAPH EXECUTION FAILED")
        print("=" * 70)

        print(f"\nError type: {type(exc).__name__}")
        print(f"Error     : {exc}")

        raise

    # ------------------------------------------------------------------
    # Extract state
    # ------------------------------------------------------------------

    safety_status = final_state.get(
        "safety_status",
        "UNKNOWN",
    )

    safety_notes = final_state.get(
        "safety_notes",
        [],
    )

    prediction = final_state.get(
        "prediction",
        {},
    )

    uncertainty = final_state.get(
        "uncertainty",
        {},
    )

    distribution = final_state.get(
        "distribution",
        {},
    )

    decision = final_state.get(
        "decision",
        {},
    )

    route = final_state.get(
        "route",
        "UNKNOWN",
    )

    route_reason = final_state.get(
        "route_reason",
        "",
    )

    evidence = final_state.get(
        "retrieved_evidence",
        [],
    )

    citations = final_state.get(
        "citations",
        [],
    )

    response = final_state.get(
        "response",
        {},
    )

    errors = final_state.get(
        "errors",
        [],
    )

    # ------------------------------------------------------------------
    # SAFETY
    # ------------------------------------------------------------------

    print_section("SAFETY")

    print("\nSafety:")
    print(safety_status)

    print("\nSafety Notes:")

    if safety_notes:
        for note in safety_notes:
            print(f"- {note}")
    else:
        print("- None")

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------

    print_section("MODEL PREDICTION")

    print("\nPrediction:")

    if prediction:
        print(prediction)
    else:
        print("No prediction returned.")

    # ------------------------------------------------------------------
    # UNCERTAINTY
    # ------------------------------------------------------------------

    print("\nUncertainty:")

    if uncertainty:
        print(uncertainty)
    else:
        print("No uncertainty information returned.")

    # ------------------------------------------------------------------
    # OOD / DISTRIBUTION
    # ------------------------------------------------------------------

    print("\nDistribution:")

    if distribution:
        print(distribution)
    else:
        print("No distribution information returned.")

    # ------------------------------------------------------------------
    # DECISION
    # ------------------------------------------------------------------

    print("\nDecision:")

    if decision:
        print(decision)
    else:
        print("No decision returned.")

    # ------------------------------------------------------------------
    # ROUTING
    # ------------------------------------------------------------------

    print("\nRoute:")
    print(route)

    print("\nRoute Reason:")
    print(route_reason)

    # ------------------------------------------------------------------
    # RETRIEVED EVIDENCE
    # ------------------------------------------------------------------

    print_section("RETRIEVED EVIDENCE")

    print(f"\nEvidence:")
    print(f"{len(evidence)} chunks retrieved")

    for index, item in enumerate(evidence, start=1):

        citation_id = item.get(
            "citation_id",
            f"[E{index}]",
        )

        rank = item.get(
            "rank",
            index,
        )

        source = item.get(
            "source",
            "",
        )

        file_name = item.get(
            "file_name",
            "",
        )

        chunk_id = item.get(
            "chunk_id",
            "",
        )

        similarity = item.get(
            "similarity",
            0.0,
        )

        text = item.get(
            "text",
            "",
        )

        print()
        print(f"Evidence {index}:")
        print(f"Citation ID : {citation_id}")
        print(f"Rank        : {rank}")
        print(f"Source      : {source}")
        print(f"File        : {file_name}")
        print(f"Chunk ID    : {chunk_id}")

        try:
            print(
                f"Similarity  : {float(similarity):.6f}"
            )
        except (TypeError, ValueError):
            print(f"Similarity  : {similarity}")

        print("Text        :")

        if text:
            print(text)
        else:
            print("[EMPTY]")

    # ------------------------------------------------------------------
    # CITATIONS
    # ------------------------------------------------------------------

    print_section("CITATIONS")

    if citations:

        for citation in citations:
            print(citation)

    else:
        print("No citations generated.")

    # ------------------------------------------------------------------
    # FINAL RESPONSE
    # ------------------------------------------------------------------

    print_section("FINAL RESPONSE")

    # Deterministic summary
    print("\nSummary:")

    summary = response.get(
        "summary",
        "",
    )

    if summary:
        print(summary)
    else:
        print("No summary returned.")

    # Route
    print("\nRoute:")

    print(
        response.get(
            "route",
            route,
        )
    )

    # Route reason
    print("\nRoute Reason:")

    print(
        response.get(
            "route_reason",
            route_reason,
        )
    )

    # Safety
    print("\nSafety Status:")

    print(
        response.get(
            "safety_status",
            safety_status,
        )
    )

    # ------------------------------------------------------------------
    # LLM EXPLANATION
    # ------------------------------------------------------------------

    print("\nLLM EXPLANATION:")

    llm_explanation = response.get(
        "llm_explanation",
        "",
    )

    if llm_explanation:

        print("-" * 70)
        print(llm_explanation)
        print("-" * 70)

    else:

        print(
            "[NO LLM EXPLANATION RETURNED]"
        )

    # ------------------------------------------------------------------
    # LLM GENERATOR
    # ------------------------------------------------------------------

    print("\nGenerator:")

    generator = response.get(
        "generator",
        "",
    )

    if generator:
        print(generator)
    else:
        print("[UNKNOWN]")

    # ------------------------------------------------------------------
    # FINAL AUDIT
    # ------------------------------------------------------------------

    print("\nFinal Audit:")

    audit = response.get(
        "audit",
        {},
    )

    if audit:

        print(audit)

    else:

        print(
            "[NO FINAL AUDIT RESULT RETURNED]"
        )

    # ------------------------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------------------------

    print("\nDisclaimer:")

    disclaimer = response.get(
        "disclaimer",
        "",
    )

    if disclaimer:
        print(disclaimer)
    else:
        print("[NO DISCLAIMER RETURNED]")

    # ------------------------------------------------------------------
    # ERRORS
    # ------------------------------------------------------------------

    print_section("ERRORS")

    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print("[]")

    # ------------------------------------------------------------------
    # FINAL STATUS
    # ------------------------------------------------------------------

    print_section(
        "LANGGRAPH TEST FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()