
from __future__ import annotations

from backend.app.agents.nodes.router import routing_node


def test_standard_review():
    state = {
        "decision": {
            "status": "SCREENING_RESULT"
        },
        "uncertainty": {
            "category": "LOW",
            "confidence": 0.95,
        },
        "distribution": {
            "status": "IN_DISTRIBUTION",
            "is_ood": False,
        },
    }

    result = routing_node(state)

    print("STANDARD REVIEW")
    print(result)
    print()


def test_human_review():
    state = {
        "decision": {
            "status": "HUMAN_REVIEW"
        },
        "uncertainty": {
            "category": "HIGH",
            "confidence": 0.10,
        },
        "distribution": {
            "status": "IN_DISTRIBUTION",
            "is_ood": False,
        },
    }

    result = routing_node(state)

    print("HUMAN REVIEW")
    print(result)
    print()


def test_ood_review():
    state = {
        "decision": {
            "status": "OOD_REVIEW"
        },
        "uncertainty": {
            "category": "LOW",
            "confidence": 0.90,
        },
        "distribution": {
            "status": "OOD",
            "is_ood": True,
        },
    }

    result = routing_node(state)

    print("OOD REVIEW")
    print(result)
    print()


def main():
    print("=" * 70)
    print("NEUROSCREEN AI - ROUTING TEST")
    print("=" * 70)
    print()

    test_standard_review()
    test_human_review()
    test_ood_review()

    print("=" * 70)
    print("ROUTING TEST FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()