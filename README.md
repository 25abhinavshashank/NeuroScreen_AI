# NEUROSCREEN AI

A research/educational prototype for AI-assisted autism screening support. It is **not a diagnostic system** — outputs are model-estimated probabilities and evidence-grounded explanations meant to support, not replace, a qualified professional's judgment.

## What this is

A LangGraph-orchestrated pipeline that takes AQ-10-Child questionnaire responses, runs them through a logistic regression screening model with uncertainty/OOD detection, routes the result (standard / human review / OOD review), retrieves supporting evidence from a knowledge base of CDC/WHO/NIMH guidance via a hybrid FAISS + BM25 reranked RAG pipeline, generates an LLM explanation, and runs that explanation through deterministic citation and groundedness checks before a final safety audit — falling back to evidence-quoted or fully deterministic text if any check fails. Every run is persisted for history and live-traffic monitoring, and surfaced through a React dashboard.

```
Questionnaire → Safety gate → ML classifier (+ uncertainty/OOD) → Router
                                                                      │
                              ┌───────────────────────────────────────┘
                              ▼
                    Retrieval → Rerank (BM25 + dense RRF) → Evidence
                              │
                              ▼
                    LLM explanation → Citation guard → Groundedness
                              │
                              ▼
                    Final audit (deterministic) → Response → persisted + served
```

## Repository layout

```
backend/app/
  agents/          LangGraph workflow (graph.py, state.py, nodes/)
  api/routes/       FastAPI endpoints
  core/            settings, logging, security middleware
  db/              SQLAlchemy models + repositories (screening history)
  ml/              prediction service, training, calibration, uncertainty, SHAP
  monitoring/      live-traffic performance/fairness/drift/audit
  rag/             ingestion, chunking, embeddings, retrieval, reranking, evaluation
  schemas/         Pydantic request/response contracts
  tests/           pytest suite (unit/ml/rag/api/integration)
frontend/          React + Vite + Tailwind + shadcn-style UI
models/            trained artifacts + registry (single source of truth)
data/              raw dataset, processed RAG index
knowledge_base/    source PDFs for RAG
scripts/           training, evaluation, and manual integration-test scripts
```

## Running it locally

### Backend

Requires Python 3.11 and (for RAG retrieval + LLM explanation) a local [Ollama](https://ollama.com) server with `nomic-embed-text` and `gemma2:2b` pulled. Without Ollama, the ML prediction/routing/safety/audit chain still works — RAG retrieval and LLM generation degrade to their deterministic fallbacks rather than crashing.

```bash
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env          # edit as needed

python -m alembic upgrade head
uvicorn backend.app.main:app --reload
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Opens on http://localhost:5173, proxying `/api` and `/health` to the backend on port 8000 (see `frontend/vite.config.ts`).

### Tests

```bash
python -m pytest              # backend
cd frontend && npm run build  # frontend typecheck + build
```

### Docker

```bash
docker compose up --build
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull gemma2:2b
```

Frontend on http://localhost:8080, backend on http://localhost:8000.

## Important limitations (read before presenting results as accurate)

- **Not a diagnosis.** Every response carries this disclaimer and it must not be dropped from any UI or report built on top of this API.
- **Dataset label-reconstruction issue.** The training target is tightly determined by the questionnaire score itself (see `scripts/audit_dataset.py`), so near-perfect model metrics reflect the dataset's construction, not clinical validity. Report probabilities as "model-estimated," never as a percentage chance of a diagnosis.
- **Groundedness metrics are a lexical-overlap heuristic**, not a semantic entailment or factual-correctness check. See `backend/app/rag/evaluation/groundedness.py`.
- **Fairness monitoring is descriptive, not a validated audit** — group sizes in live traffic are typically far too small for statistical conclusions.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — component and data-flow detail
- [`docs/api.md`](docs/api.md) — endpoint reference
- [`docs/evaluation.md`](docs/evaluation.md) — how to (re)run ML/RAG/agent evaluation
- [`docs/deployment.md`](docs/deployment.md) — Docker/CI notes
- [`CHANGES.md`](CHANGES.md) — history of major fixes and design decisions
