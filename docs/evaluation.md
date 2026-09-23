# Evaluation

Evaluation (offline, against held-out/labeled data) is separate from monitoring (live traffic, no ground truth) — see `docs/architecture.md`.

## ML

```bash
python -m scripts.train        # retrain, writes to models/artifacts + models/metadata
python -m scripts.calibrate
python -m scripts.evaluate      # holdout metrics, confidence intervals, calibration
python -m scripts.audit_dataset # label-reconstruction / leakage audit — read this before trusting accuracy numbers
```

Outputs land in `models/metadata/*.csv`. **Read `scripts/audit_dataset.py`'s output before quoting any accuracy figure** — the dataset has a known label-reconstruction issue (the target is tightly determined by the questionnaire score), so near-perfect holdout metrics are expected and do not indicate clinical validity.

## RAG retrieval + reranking

```bash
python -m scripts.build_rag_index   # (re)build the FAISS index from knowledge_base/
python -m scripts.evaluate_rag      # Recall@K, Precision@K, MRR, NDCG — raw FAISS vs. reranked
```

`scripts/evaluate_rag.py` reports both the raw FAISS ranking and the hybrid BM25+dense reranked ranking side by side, against `data/processed/rag/evaluation_dataset.json`, so the reranker's actual effect is visible rather than assumed.

## Groundedness / citations

```bash
python -m scripts.test_groundedness
python -m scripts.test_citations
```

These exercise `backend/app/rag/evaluation/groundedness.py` directly against fixed inputs — useful for checking a change to `GROUNDEDNESS_THRESHOLD`/`MAX_UNSUPPORTED_CLAIM_RATE` (in `backend/app/agents/nodes/groundedness.py`) doesn't silently change pass/fail behavior.

## Full agent pipeline

```bash
python -m scripts.test_graph      # requires Ollama running (nomic-embed-text + gemma2:2b pulled)
python -m scripts.evaluate_agent
python -m scripts.red_team        # adversarial prompts against the safety/audit layer
```

## Automated tests

```bash
python -m pytest                  # unit/ml/rag/api/integration — no Ollama required
```

The integration tests exercise every node except `retrieval`/`reranker`/`llm_response` directly with fabricated evidence, so the full safety→groundedness→audit chain is covered without a live Ollama dependency in CI.
