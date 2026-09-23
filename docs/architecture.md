# Architecture

## Request flow

`POST /api/v1/assessment` builds an initial `NeuroScreenState` (see `backend/app/agents/state.py`) from the questionnaire payload and invokes the compiled LangGraph (`backend/app/agents/graph.py`):

1. **safety** — gates on presence of patient data; sets `safety_status`.
2. **classifier** — calls the singleton `PredictionService` (`backend/app/ml/prediction_service.py`) for prediction, uncertainty, OOD distance, decision, and SHAP explanation. Pure ML, no LangChain/Ollama dependency.
3. **router** — maps decision + uncertainty + OOD into `STANDARD_REVIEW` / `HUMAN_REVIEW` / `OOD_REVIEW`.
4. **retrieval** — embeds the query via Ollama (`nomic-embed-text`) and searches the FAISS index (`backend/app/rag/retrieval/vector_store.py`), over-fetching a 20-document candidate pool.
5. **reranker** — fuses dense similarity and BM25 lexical scores via Reciprocal Rank Fusion (`backend/app/rag/reranking/fusion.py`) down to the top 5.
6. **evidence** — normalizes retrieved chunks into a consistent `text`/`citation_id` shape.
7. **llm_response** — calls Ollama (`gemma2:2b`) to generate an explanation citing `[E1]`-style markers.
8. **citation_guard** — validates/repairs citation markers against actual evidence.
9. **groundedness** — the canonical evaluator (`backend/app/rag/evaluation/groundedness.py`) scores every sentence (cited or not) for lexical support; if the LLM text fails, a deterministic fallback quoting evidence directly is tried and re-evaluated.
10. **final_audit** — deterministic regex checks for diagnostic/probability/reliability/causal claims etc.; on failure, substitutes a fully deterministic safe response.
11. **response** — assembles the final payload; persisted to `screening_records` by the API route.

Every node returns a partial state update; LangGraph merges these into a single `NeuroScreenState` dict threaded through the whole run.

## Why two "groundedness" concerns exist

`backend/app/rag/evaluation/groundedness.py` is the *evaluator*: pure functions (`evaluate_groundedness`, `lexical_support_score`, `sentence_split`, ...) with no LangGraph dependency, reused by `citation_guard.py` and the offline evaluation scripts. `backend/app/agents/nodes/groundedness.py` is the *node*: it calls the evaluator, layers on the workflow's pass/fail policy (`GROUNDEDNESS_THRESHOLD`, `MAX_UNSUPPORTED_CLAIM_RATE`), and owns the fallback-and-re-evaluate loop. Keep this split — the evaluator's public function names and `evaluate_groundedness` signature are depended on elsewhere and should not change without updating both call sites.

## Persistence

`backend/app/db/models.py` defines one table, `screening_records`, storing both queryable summary columns (label, route, groundedness score, ...) and the full JSON response. Alembic (`backend/alembic/`) owns schema migrations — `backend/app/db/database.py`'s `init_db()` exists only as a fallback for ad-hoc scripts, not for the API path.

## Monitoring vs. evaluation

`backend/app/monitoring/*` computes live-traffic signals from `screening_records` (label mix, groundedness pass rate, OOD-rate drift vs. history, descriptive subgroup breakdowns). This is distinct from `scripts/evaluate.py` / `scripts/evaluate_rag.py`, which score the model/retriever against held-out labeled data. Monitoring has no ground truth to compare against; evaluation does.

## Frontend

React + Vite + TypeScript, Tailwind with shadcn-style primitives in `frontend/src/components/ui/`. `frontend/src/types/api.ts` mirrors the backend Pydantic schemas by hand — update both sides together when a schema changes. `frontend/src/services/api.ts` is the only place that knows the backend's URL paths.
