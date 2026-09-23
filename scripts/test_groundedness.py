from backend.app.rag.evaluation.groundedness import evaluate_groundedness


citations = [
    {
        "citation_id": "[E1]",
        "text": "Parents are reliable sources of information about their children's development.",
    },
    {
        "citation_id": "[E2]",
        "text": "A positive screening result should be followed by a thorough assessment. Screening tools do not provide diagnoses.",
    },
    {
        "citation_id": "[E5]",
        "text": "Developmental monitoring and screening work together to help identify developmental concerns.",
    },
]


# This is intentionally incorrect.
# E1 does not support the claim about developmental screening.
wrong = (
    "Developmental screening can help identify developmental concerns. [E1] "
    "A positive screening result should be followed by further assessment. [E2]"
)


# This is the correct citation mapping.
# E5 supports the first claim and E2 supports the second claim.
correct = (
    "Developmental screening can help identify developmental concerns. [E5] "
    "A positive screening result should be followed by further assessment. [E2]"
)


for name, answer in (
    ("WRONG CITATIONS", wrong),
    ("CORRECT CITATIONS", correct),
):
    print(f"\n{name}")

    result = evaluate_groundedness(
        answer,
        citations,
    )

    print(result)