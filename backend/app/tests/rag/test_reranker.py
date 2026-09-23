from __future__ import annotations

from backend.app.rag.reranking.bm25 import BM25Index, tokenize
from backend.app.rag.reranking.fusion import rerank_candidates
from backend.app.rag.evaluation.retrieval import evaluate_single_query, get_source, ndcg_at_k


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Screening, does NOT diagnose!") == ["screening", "does", "not", "diagnose"]


def test_bm25_scores_favor_keyword_matches(tmp_path, monkeypatch):
    metadata = [
        {"chunk_id": "chunk_0", "text": "Developmental screening tools do not provide a diagnosis."},
        {"chunk_id": "chunk_1", "text": "The weather today is sunny with a light breeze."},
    ]
    metadata_path = tmp_path / "metadata.json"
    import json

    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    index = BM25Index()
    monkeypatch.setattr("backend.app.rag.reranking.bm25.METADATA_PATH", metadata_path)
    index.load()

    scores = index.score_by_chunk_id("developmental screening diagnosis")
    assert scores["chunk_0"] > scores["chunk_1"]


def test_rerank_candidates_dedupes_by_chunk_id():
    candidates = [
        {"chunk_id": "chunk_0", "text": "duplicate text here", "similarity": 0.9},
        {"chunk_id": "chunk_0", "text": "duplicate text here", "similarity": 0.5},
        {"chunk_id": "chunk_1", "text": "other content entirely", "similarity": 0.3},
    ]
    ranked = rerank_candidates("duplicate text", candidates, top_n=5)
    chunk_ids = [c["chunk_id"] for c in ranked]
    assert chunk_ids.count("chunk_0") == 1


def test_rerank_candidates_assigns_rank_and_citation_id():
    candidates = [
        {"chunk_id": f"chunk_{i}", "text": f"content number {i}", "similarity": 0.1 * i}
        for i in range(3)
    ]
    ranked = rerank_candidates("content", candidates, top_n=2)
    assert len(ranked) == 2
    assert ranked[0]["rank"] == 1
    assert ranked[0]["citation_id"] == "[E1]"
    assert ranked[1]["rank"] == 2


def test_get_source_normalizes_windows_path_to_basename():
    item = {"source": "knowledge_base\\guidelines\\CDC_autism_screening.pdf"}
    assert get_source(item) == "CDC_autism_screening.pdf"


def test_get_source_prefers_file_name_field():
    item = {"source": "some/other/path.pdf", "file_name": "CDC_autism_screening.pdf"}
    assert get_source(item) == "CDC_autism_screening.pdf"


def test_evaluate_single_query_source_hits_match_after_basename_fix():
    retrieved = [
        {"chunk_id": "chunk_28", "source": "knowledge_base\\guidelines\\CDC_clinical_screening.pdf"},
    ]
    result = evaluate_single_query(
        retrieved,
        relevant_chunks=["chunk_28"],
        relevant_sources=["CDC_clinical_screening.pdf"],
        k=5,
    )
    assert result["source_hits"] == ["CDC_clinical_screening.pdf"]
    assert result["recall_at_k"] == 1.0


def test_ndcg_perfect_ranking_is_one():
    retrieved = [{"chunk_id": "chunk_1"}, {"chunk_id": "chunk_2"}]
    assert ndcg_at_k(retrieved, {"chunk_1"}) == 1.0


def test_ndcg_no_relevant_chunks_is_zero():
    assert ndcg_at_k([{"chunk_id": "chunk_1"}], set()) == 0.0
