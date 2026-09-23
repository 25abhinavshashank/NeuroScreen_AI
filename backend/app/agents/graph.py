from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.app.agents.state import NeuroScreenState
from backend.app.agents.nodes.safety import safety_node
from backend.app.agents.nodes.classifier import classifier_node
from backend.app.agents.nodes.router import routing_node, route_after_classifier
from backend.app.agents.nodes.standard_review import standard_review_node
from backend.app.agents.nodes.human_review import human_review_node
from backend.app.agents.nodes.ood_review import ood_review_node
from backend.app.agents.nodes.retrieval import retrieval_node
from backend.app.agents.nodes.reranker import reranker_node
from backend.app.agents.nodes.evidence import evidence_node
from backend.app.agents.nodes.llm_response import llm_response_node
from backend.app.agents.nodes.citation_guard import citation_guard_node
from backend.app.agents.nodes.groundedness import groundedness_node
from backend.app.agents.nodes.final_audit import final_audit_node
from backend.app.agents.nodes.response import response_node


def build_graph():
    workflow = StateGraph(NeuroScreenState)
    workflow.add_node("safety", safety_node)
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("router", routing_node)
    workflow.add_node("standard_review", standard_review_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("ood_review", ood_review_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("evidence", evidence_node)
    workflow.add_node("llm_response", llm_response_node)
    workflow.add_node("citation_guard", citation_guard_node)
    workflow.add_node("groundedness", groundedness_node)
    workflow.add_node("final_audit", final_audit_node)
    workflow.add_node("response", response_node)

    workflow.add_edge(START, "safety")
    workflow.add_edge("safety", "classifier")
    workflow.add_edge("classifier", "router")
    workflow.add_conditional_edges("router", route_after_classifier, {
        "standard_review": "standard_review",
        "human_review": "human_review",
        "ood_review": "ood_review",
        "error": END,
    })
    for node in ("standard_review", "human_review", "ood_review"):
        workflow.add_edge(node, "retrieval")
    workflow.add_edge("retrieval", "reranker")
    workflow.add_edge("reranker", "evidence")
    workflow.add_edge("evidence", "llm_response")
    workflow.add_edge("llm_response", "citation_guard")
    workflow.add_edge("citation_guard", "groundedness")
    workflow.add_edge("groundedness", "final_audit")
    workflow.add_edge("final_audit", "response")
    workflow.add_edge("response", END)
    return workflow.compile()


neuroscreen_graph = build_graph()
