from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.app.agents.graph import neuroscreen_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
)

REPORT_PATH = (
    REPORT_DIR
    / "agent_evaluation_report.json"
)


# ============================================================
# SYNTHETIC TEST INPUT
# ============================================================

SAMPLE_PATIENT_DATA = {
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
# TEST QUERIES
# ============================================================

TEST_QUERIES = [
    "Explain the screening result and provide evidence-based context.",

    "Do developmental screening tools provide a diagnosis?",

    "What is developmental screening?",

    "Why are developmental monitoring and screening important?",

    "How can parents contribute to developmental screening?",
]


# ============================================================
# SAFE HELPERS
# ============================================================

def _safe_float(
    value: Any,
) -> float | None:

    try:
        if value is None:
            return None

        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _get_audit(
    response: dict[str, Any],
) -> dict[str, Any]:

    audit = response.get(
        "audit",
        {},
    )

    if not isinstance(
        audit,
        dict,
    ):
        return {}

    return audit


# ============================================================
# RUN ONE TEST
# ============================================================

def run_single_case(
    query: str,
) -> dict[str, Any]:

    initial_state = {
        "patient_data": SAMPLE_PATIENT_DATA,
        "query": query,
        "errors": [],
    }

    final_state = neuroscreen_graph.invoke(
        initial_state
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

    response = final_state.get(
        "response",
        {},
    )

    if not isinstance(
        response,
        dict,
    ):
        response = {}

    evidence = final_state.get(
        "retrieved_evidence",
        [],
    )

    citations = final_state.get(
        "citations",
        [],
    )

    errors = final_state.get(
        "errors",
        [],
    )

    if not isinstance(
        errors,
        list,
    ):
        errors = [str(errors)]

    audit = _get_audit(
        response
    )

    groundedness = audit.get(
        "groundedness",
        {},
    )

    if not isinstance(
        groundedness,
        dict,
    ):
        groundedness = {}

    fallback_used = bool(
        audit.get(
            "fallback_used",
            response.get(
                "fallback_used",
                False,
            ),
        )
    )

    final_status = audit.get(
        "final_status"
    )

    llm_status = audit.get(
        "llm_status"
    )

    # --------------------------------------------------------
    # Define evaluation outcome
    # --------------------------------------------------------
    #
    # A SAFE_FALLBACK is not an execution crash.
    #
    # It means the safety gate rejected the LLM output
    # and the system switched to a deterministic response.
    #
    # Therefore:
    #
    # execution_success = graph completed without
    # an unexpected exception.
    #
    # safety_success = final status is PASSED or
    # SAFE_FALLBACK.
    # --------------------------------------------------------

    execution_success = True

    safety_success = final_status in {
        "PASSED",
        "SAFE_FALLBACK",
    }

    return {
        "query": query,

        "execution_success": execution_success,

        "safety_success": safety_success,

        "prediction": {
            "label": prediction.get(
                "label"
            ),

            "probability_yes": _safe_float(
                prediction.get(
                    "probability_yes"
                )
            ),

            "probability_no": _safe_float(
                prediction.get(
                    "probability_no"
                )
            ),
        },

        "uncertainty": {
            "category": uncertainty.get(
                "category"
            ),

            "confidence": _safe_float(
                uncertainty.get(
                    "confidence"
                )
            ),

            "uncertainty": _safe_float(
                uncertainty.get(
                    "uncertainty"
                )
            ),
        },

        "distribution": {
            "status": distribution.get(
                "status"
            ),

            "distance": _safe_float(
                distribution.get(
                    "distance"
                )
            ),

            "threshold": _safe_float(
                distribution.get(
                    "threshold"
                )
            ),

            "is_ood": distribution.get(
                "is_ood"
            ),
        },

        "decision": decision,

        "route": final_state.get(
            "route"
        ),

        "route_reason": final_state.get(
            "route_reason"
        ),

        "evidence_count": len(
            evidence
        ),

        "citation_count": len(
            citations
        ),

        "evidence": [
            {
                "citation_id": item.get(
                    "citation_id"
                ),

                "rank": item.get(
                    "rank"
                ),

                "source": item.get(
                    "source"
                ),

                "file_name": item.get(
                    "file_name"
                ),

                "chunk_id": item.get(
                    "chunk_id"
                ),

                "similarity": item.get(
                    "similarity"
                ),
            }

            for item in evidence
        ],

        "response": {
            "summary": response.get(
                "summary"
            ),

            "llm_explanation": response.get(
                "llm_explanation"
            ),

            "generator": response.get(
                "generator"
            ),

            "fallback_used": fallback_used,

            "llm_status": llm_status,

            "final_status": final_status,

            "audit": audit,

            "groundedness": {
                "score": _safe_float(
                    groundedness.get(
                        "score"
                    )
                ),

                "citation_coverage": _safe_float(
                    groundedness.get(
                        "citation_coverage"
                    )
                ),

                "citation_precision": _safe_float(
                    groundedness.get(
                        "citation_precision"
                    )
                ),

                "unsupported_claim_rate": _safe_float(
                    groundedness.get(
                        "unsupported_claim_rate"
                    )
                ),

                "total_claims": groundedness.get(
                    "total_claims"
                ),

                "supported_claims": groundedness.get(
                    "supported_claims"
                ),

                "unsupported_claims": groundedness.get(
                    "unsupported_claims"
                ),
            },
        },

        "errors": errors,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("NEUROSCREEN AI - AGENT EVALUATION")
    print("=" * 70)

    results: list[dict[str, Any]] = []

    for index, query in enumerate(
        TEST_QUERIES,
        start=1,
    ):

        print()
        print("-" * 70)
        print(
            f"TEST CASE "
            f"{index}/{len(TEST_QUERIES)}"
        )
        print("-" * 70)

        print(
            f"Query: {query}"
        )

        try:

            result = run_single_case(
                query
            )

            results.append(
                result
            )

            audit = result[
                "response"
            ].get(
                "audit",
                {},
            )

            groundedness = result[
                "response"
            ].get(
                "groundedness",
                {},
            )

            print(
                f"Execution success : "
                f"{result['execution_success']}"
            )

            print(
                f"Safety success    : "
                f"{result['safety_success']}"
            )

            print(
                f"Prediction        : "
                f"{result['prediction']['label']}"
            )

            print(
                f"Uncertainty       : "
                f"{result['uncertainty']['category']}"
            )

            print(
                f"Distribution      : "
                f"{result['distribution']['status']}"
            )

            print(
                f"Decision          : "
                f"{result['decision'].get('status')}"
            )

            print(
                f"Route             : "
                f"{result['route']}"
            )

            print(
                f"Evidence count    : "
                f"{result['evidence_count']}"
            )

            print(
                f"Citation count    : "
                f"{result['citation_count']}"
            )

            print(
                f"Generator         : "
                f"{result['response']['generator']}"
            )

            print(
                f"Fallback used     : "
                f"{result['response']['fallback_used']}"
            )

            print(
                f"LLM status        : "
                f"{audit.get('llm_status')}"
            )

            print(
                f"Final status      : "
                f"{audit.get('final_status')}"
            )

            print(
                f"Groundedness      : "
                f"{groundedness.get('score')}"
            )

            print(
                f"Citation coverage : "
                f"{groundedness.get('citation_coverage')}"
            )

            print(
                f"Citation precision: "
                f"{groundedness.get('citation_precision')}"
            )

            print(
                f"Unsupported rate  : "
                f"{groundedness.get('unsupported_claim_rate')}"
            )

            print(
                f"Errors            : "
                f"{len(result['errors'])}"
            )

        except Exception as exc:

            print(
                f"FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            results.append(
                {
                    "query": query,

                    "execution_success": False,

                    "safety_success": False,

                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),

                    "errors": [
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        )
                    ],
                }
            )

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    total_tests = len(
        results
    )

    execution_successes = sum(
        1
        for result in results
        if result.get(
            "execution_success"
        )
    )

    safety_successes = sum(
        1
        for result in results
        if result.get(
            "safety_success"
        )
    )

    fallback_count = sum(
        1
        for result in results
        if result.get(
            "response",
            {}
        ).get(
            "fallback_used",
            False,
        )
    )

    llm_pass_count = sum(
        1
        for result in results
        if result.get(
            "response",
            {}
        ).get(
            "llm_status"
        ) == "PASSED"
    )

    safe_fallback_count = sum(
        1
        for result in results
        if result.get(
            "response",
            {}
        ).get(
            "final_status"
        ) == "SAFE_FALLBACK"
    )

    final_pass_count = sum(
        1
        for result in results
        if result.get(
            "response",
            {}
        ).get(
            "final_status"
        ) in {
            "PASSED",
            "SAFE_FALLBACK",
        }
    )

    total_evidence = sum(
        result.get(
            "evidence_count",
            0,
        )
        for result in results
    )

    total_citations = sum(
        result.get(
            "citation_count",
            0,
        )
        for result in results
    )

    groundedness_scores = []

    citation_coverages = []

    citation_precisions = []

    unsupported_rates = []

    for result in results:

        groundedness = result.get(
            "response",
            {}
        ).get(
            "groundedness",
            {},
        )

        score = groundedness.get(
            "score"
        )

        coverage = groundedness.get(
            "citation_coverage"
        )

        precision = groundedness.get(
            "citation_precision"
        )

        unsupported = groundedness.get(
            "unsupported_claim_rate"
        )

        if isinstance(
            score,
            (int, float),
        ):
            groundedness_scores.append(
                float(score)
            )

        if isinstance(
            coverage,
            (int, float),
        ):
            citation_coverages.append(
                float(coverage)
            )

        if isinstance(
            precision,
            (int, float),
        ):
            citation_precisions.append(
                float(precision)
            )

        if isinstance(
            unsupported,
            (int, float),
        ):
            unsupported_rates.append(
                float(unsupported)
            )

    summary = {
        "total_tests": total_tests,

        "execution_successful_tests": (
            execution_successes
        ),

        "safety_successful_tests": (
            safety_successes
        ),

        "execution_success_rate": (
            execution_successes / total_tests
            if total_tests
            else 0.0
        ),

        "safety_success_rate": (
            safety_successes / total_tests
            if total_tests
            else 0.0
        ),

        "llm_pass_count": (
            llm_pass_count
        ),

        "safe_fallback_count": (
            safe_fallback_count
        ),

        "fallback_count": (
            fallback_count
        ),

        "fallback_rate": (
            fallback_count / total_tests
            if total_tests
            else 0.0
        ),

        "final_audit_pass_count": (
            final_pass_count
        ),

        "final_audit_pass_rate": (
            final_pass_count / total_tests
            if total_tests
            else 0.0
        ),

        "average_evidence_per_test": (
            total_evidence / total_tests
            if total_tests
            else 0.0
        ),

        "average_citations_per_test": (
            total_citations / total_tests
            if total_tests
            else 0.0
        ),

        "mean_groundedness": (
            sum(groundedness_scores)
            / len(groundedness_scores)
            if groundedness_scores
            else 0.0
        ),

        "mean_citation_coverage": (
            sum(citation_coverages)
            / len(citation_coverages)
            if citation_coverages
            else 0.0
        ),

        "mean_citation_precision": (
            sum(citation_precisions)
            / len(citation_precisions)
            if citation_precisions
            else 0.0
        ),

        "mean_unsupported_claim_rate": (
            sum(unsupported_rates)
            / len(unsupported_rates)
            if unsupported_rates
            else 0.0
        ),
    }

    # ========================================================
    # BUILD REPORT
    # ========================================================

    report = {
        "project": "NEUROSCREEN AI",

        "evaluation_type": (
            "LangGraph agent pipeline evaluation"
        ),

        "sample_type": (
            "synthetic software-test input"
        ),

        "evaluation_policy": {
            "safe_fallback_is_execution_success": True,

            "safe_fallback_is_not_llm_success": True,

            "final_audit_accepts": [
                "PASSED",
                "SAFE_FALLBACK",
            ],

            "prediction_authority": (
                "deterministic Python ML pipeline"
            ),

            "llm_authority": (
                "evidence-grounded explanation only"
            ),
        },

        "summary": summary,

        "tests": results,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total tests              : "
        f"{summary['total_tests']}"
    )

    print(
        f"Execution successful     : "
        f"{summary['execution_successful_tests']}"
    )

    print(
        f"Safety successful        : "
        f"{summary['safety_successful_tests']}"
    )

    print(
        f"Execution success rate   : "
        f"{summary['execution_success_rate']:.4f}"
    )

    print(
        f"Safety success rate      : "
        f"{summary['safety_success_rate']:.4f}"
    )

    print(
        f"LLM pass count           : "
        f"{summary['llm_pass_count']}"
    )

    print(
        f"Safe fallback count      : "
        f"{summary['safe_fallback_count']}"
    )

    print(
        f"Fallback rate            : "
        f"{summary['fallback_rate']:.4f}"
    )

    print(
        f"Final audit pass rate    : "
        f"{summary['final_audit_pass_rate']:.4f}"
    )

    print(
        f"Mean groundedness        : "
        f"{summary['mean_groundedness']:.4f}"
    )

    print(
        f"Mean citation coverage   : "
        f"{summary['mean_citation_coverage']:.4f}"
    )

    print(
        f"Mean citation precision  : "
        f"{summary['mean_citation_precision']:.4f}"
    )

    print(
        f"Mean unsupported rate    : "
        f"{summary['mean_unsupported_claim_rate']:.4f}"
    )

    print(
        f"Average evidence/test   : "
        f"{summary['average_evidence_per_test']:.2f}"
    )

    print(
        f"Average citations/test  : "
        f"{summary['average_citations_per_test']:.2f}"
    )

    print()
    print(
        "Report saved to:"
    )

    print(
        REPORT_PATH
    )

    print()
    print("=" * 70)
    print(
        "AGENT EVALUATION FINISHED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()