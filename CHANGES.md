
# Full ZIP inspection findings

## Critical findings

1. `backend/app/agents/nodes/groundedness.py` and `backend/app/rag/evaluation/groundedness.py` implemented two different groundedness systems.
2. The LangGraph graph used the agents version while `scripts/test_groundedness.py` used the RAG version.
3. The assessment API called `prediction_service.predict()` directly, bypassing LangGraph, RAG, citation validation, and final audit.
4. The active model registry in the uploaded project used absolute Windows paths, which is not portable.
5. The uploaded artifact lacked `ood_center`, `ood_mad`, `ood_threshold`, and `shap_background`, so the prediction service reported OOD and SHAP as unavailable.
6. Several ML modules and API/monitoring/database files are empty skeletons. They should not be advertised as implemented production features.
7. The RAG retrieval report has Recall@5 = 1.0 but Precision@5 = 0.2 on its four-query evaluation, meaning the retriever often puts only one relevant chunk in the top five.
8. The existing agent evaluation report is stale relative to the current graph/schema and should be regenerated rather than trusted as the current system result.
9. `requirements.txt` in the uploaded ZIP is a raw environment freeze rather than a maintainable project dependency file and is encoded as UTF-16.
10. The project currently has a strong dataset-label reconstruction issue: the target is tightly determined by the questionnaire score. Near-perfect metrics therefore cannot be interpreted as clinical accuracy.

## What v2 changes

- centralizes paths and feature schema;
- creates one canonical groundedness evaluator;
- adds deterministic citation repair;
- routes API assessments through LangGraph;
- makes the model registry portable;
- trains a deployment artifact with OOD and SHAP background;
- records dataset fingerprint and label-reconstruction audit;
- keeps the existing safe fallback behavior;
- adds a clean dependency file and project health check.

## What v3 changes

Backend stabilization:
- fixed a UTF-16-encoded `requirements.txt` and added the missing `langchain-text-splitters`/`python-docx` deps it never declared;
- rebuilt `.venv` (the committed one pointed at a Python 3.12 install from a different machine/user);
- deleted `backend/app/ml/registry/` — a dead, unused duplicate of the model registry pinned to the stale `v1.0.0` artifact; `models/registry/registry.json` is now the single source of truth;
- built out real Pydantic request/response schemas (`backend/app/schemas/`) for every endpoint, validated against live model/pipeline output rather than guessed;
- wired `/health`, `/api/v1/models`, `/api/v1/predict` (new ML-only fast path), `/api/v1/rag/retrieve`, `/api/v1/assessment` (now versioned, was `/assessment`), with structured error handling and CORS restricted to configured origins via `Settings`.

RAG quality:
- fixed a real bug in `get_source()` (`backend/app/rag/evaluation/retrieval.py`): it compared a full Windows path against a bare filename, so `source_hits`/Precision@K were silently near-zero;
- added NDCG@K to retrieval evaluation;
- replaced the reranker's plain FAISS-similarity sort with a hybrid BM25 (`backend/app/rag/reranking/bm25.py`) + dense Reciprocal Rank Fusion (`backend/app/rag/reranking/fusion.py`), shared between the LangGraph node and `scripts/evaluate_rag.py` so both score evidence identically; retrieval now over-fetches 20 candidates before reranking to 5.

Groundedness hardening:
- fixed an inconsistency where `evaluate_groundedness()`'s output shape differed between its empty-result and normal-result paths (missing `thresholds`/`cited_claims`) — now enriched consistently in `_evaluate()`;
- fixed the deterministic fallback (`_find_best_fallback`) generating generic template text with no guaranteed lexical overlap with the evidence it cited, meaning it could fail its own groundedness check — it now quotes the evidence chunk's own opening sentence, grounded by construction;
- fixed a real bug found via testing: `final_audit_node`'s PASSED branch unconditionally set `response["fallback_used"] = False`, clobbering a `True` that `groundedness_node` had already set when it substituted a grounded fallback — silently hiding from callers (including the frontend's fallback warning banner) that a fallback had fired.

New: database, monitoring, frontend, tests, Docker/CI (previously all empty skeletons or missing entirely):
- SQLAlchemy `ScreeningRecord` model + Alembic migrations + repository layer; every assessment is now persisted (best-effort — a DB failure doesn't turn a valid result into a 500) and exposed via `/api/v1/history`;
- live-traffic monitoring (`backend/app/monitoring/`) — performance, descriptive fairness breakdown, OOD/prediction drift vs. history, and an audit trail, all built on recorded screenings rather than re-deriving what `scripts/evaluate.py` already covers;
- a full React + Vite + TypeScript + Tailwind frontend (screening form, results dashboard with evidence/groundedness/audit tabs, history, monitoring) — verified end-to-end in-browser against the real backend, including the safety-fallback chain firing correctly with no Ollama present;
- a real pytest suite (44 tests: unit, ML, RAG/reranking, API, integration) replacing the empty `__init__.py` stubs — no Ollama required, since the integration tests exercise every node except retrieval/reranker/llm_response directly;
- `Dockerfile` (backend), `frontend/Dockerfile` + nginx, `docker-compose.yml` (backend + frontend + optional Ollama), and `.github/workflows/ci.yml` (pytest + frontend build + docker build);
- `README.md`, `docs/architecture.md`, `docs/api.md`, `docs/evaluation.md`, `docs/deployment.md`.

Known gaps / next steps:
- SHAP-based feature-importance UI panel not built (backend already returns `top_contributions`; only the frontend display is missing);
- no auth/rate-limiting layer (not previously in scope; add before any non-local deployment);
- `frontend/src/components/ui/` shadcn primitives were hand-written rather than generated via the shadcn CLI (network/registry access wasn't available in the build environment) — functionally equivalent, but worth diffing against upstream shadcn if you later run the CLI.

## What v4 changes

Second age band -- toddler model (1-3yr):
- trained a second model on a Kaggle Q-CHAT-10-Toddler dataset (1054 rows, ages 12-36 months), provided directly rather than sourced from UCI -- confirmed the same label-reconstruction issue applies (Class is a deterministic function of score>3, 100% match, audited);
- `PredictionService` is now a router over two `_AgeModelBundle`s (child 4-11yr, toddler 1-3yr), translating the unified request shape into each bundle's native column names; routes by age, and an age outside *both* bands is honestly flagged out-of-range (routed for review) rather than scored as if valid -- verified this matters: the generic Mahalanobis OOD distance alone does *not* reliably catch an out-of-range age (age=0.25 scored well under threshold in direct testing) since age is 1 of dozens of post-encoding features and gets diluted;
- model registry moved from a single `active_model` to `active_models: {child, toddler}`, with backward-compatible reads; `/health` and `/api/v1/models` updated to report both.

Groundedness / explanation quality -- found and fixed two real bugs via direct tracing of live behavior, not hypothetical review:
- **the actual cause of a ~94% fallback rate**: gemma2:2b intermittently writes citations as `[1]` instead of `[E1]`; every citation regex downstream requires the `E` prefix, so that sentence silently became "uncited" and failed groundedness. Fixed with a normalization pass in `llm_response.py` (`[1]` -> `[E1]`, only when evidence `E1` actually exists) plus a strengthened prompt; verified the LLM's own explanation now passes consistently in real trials, down from ~94% fallback;
- the deterministic fallback (`_find_best_fallback`) now assembles up to 3 sentences from distinct evidence items instead of 1, and rejects PDF-extraction junk (section headings run into the next sentence via a bare newline, scraped-page timestamps, mojibake, mid-clause fragments from chunk-boundary cuts) that was previously accepted as a "sentence" because it happened to be long enough;
- `final_audit_node`'s PASSED branch was unconditionally setting `fallback_used = False`, clobbering a `True` that `groundedness_node` had already set -- hid fallback usage from callers and the frontend's warning banner even when a fallback had actually fired; now only defaults when unset.

Video observation (descriptive only, not a prediction):
- investigated directly (not from memory) whether a pretrained "video -> autism" classifier exists anywhere publicly: it does not. OpenFace ships no disease classifier of any kind. The closest lead (the "Hugging Rain Man" AU dataset) is a facial-expression-atypicality *research* dataset for children already diagnosed, not a diagnostic classifier, and its face images are privacy-restricted;
- built `backend/app/video/analyzer.py` (MediaPipe FaceLandmarker Tasks API -- the legacy `mediapipe.solutions` API used in most tutorials is gone as of mediapipe 1.0) extracting face-detection ratio, gaze-toward-camera proxy, head-pose variability (via the task's own facial transformation matrix, not manual solvePnP), and blink rate (EAR) from an uploaded video or webcam recording;
- deliberately NOT fused into the screening prediction -- exposed as a separate `/api/v1/video/analyze` endpoint and UI panel with its own disclaimer, since there is no validated dataset linking these signals to ASD outcomes;
- verified end-to-end with a real face image (100% detection) vs. random noise (0% detection) to confirm detection genuinely works, then again through the live browser upload flow against the real backend.

Frontend wording fixes (both from direct user reports, verified against live behavior before fixing):
- OOD/decision reason text said "Child's age is outside **this model's** training range (1-3 years)" even when the *toddler* band was used, reading as if the child-band model applied -- now names the actual band explicitly ("the toddler model (Q-CHAT-10, 1-3 years)");
- inline `[E1]`-style citation markers stripped from the displayed explanation text (still fully tracked -- Groundedness tab has the claim-by-claim breakdown, Evidence tab has the sources); fallback banner text now differs by `generator` (grounded-fallback vs. no-evidence-safe-fallback) instead of always claiming "evidence-quoted".

Also: the AQ-10 item wording shown on the form is the Q-CHAT-10-Toddler instrument's real, documented wording (confirmed by re-reading the toddler dataset's description docx with table extraction, which the original child-dataset docx does not have at all -- zero tables, checked directly) -- accurate for the 1-3yr band, labeled as a proxy (not a verbatim match) for the 4-11yr band since that instrument's exact wording isn't available anywhere in this project.
