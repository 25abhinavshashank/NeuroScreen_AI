# API reference

Full interactive docs at `/docs` (Swagger UI) when the backend is running. This is a summary.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + active model/RAG index status |
| POST | `/api/v1/assessment` | Full pipeline: ML + RAG + LLM + safety checks. Persists to history. |
| POST | `/api/v1/predict` | ML-only: prediction/uncertainty/OOD/decision/SHAP, no RAG or LLM. Fast, works without Ollama. |
| POST | `/api/v1/rag/retrieve` | Evidence retrieval only, for debugging/evidence-explorer use. |
| GET | `/api/v1/models` | Active model registry entry. |
| GET | `/api/v1/history` | Paginated screening history (`limit`, `offset`, `route`). |
| GET | `/api/v1/history/{id}` | Full detail for one screening, including the complete response payload. |
| GET | `/api/v1/monitoring/performance` | Live-traffic label/route/decision distribution, groundedness pass rate. |
| GET | `/api/v1/monitoring/fairness` | Descriptive subgroup breakdown (gender/ethnicity/jaundice/family history). |
| GET | `/api/v1/monitoring/drift` | OOD-rate and prediction drift, recent window vs. full history. |
| GET | `/api/v1/monitoring/audit` | Flattened audit trail across recorded screenings. |
| POST | `/api/v1/video/analyze` | Descriptive face/gaze/head-pose metrics from an uploaded video (file or webcam recording). NOT an autism prediction, not combined with the screening result -- see "Video observation" below. |

## Request/response contracts

All request/response models live in `backend/app/schemas/`. The important ones:

- `AssessmentRequest` (`schemas/assessment.py`) — the 18 AQ-10-Child fields. `A1_Score`–`A10_Score` are `0`/`1`; categorical fields accept the raw dataset string values (`"m"`/`"f"`, `"yes"`/`"no"`) and are left unvalidated beyond that, since the ML preprocessing pipeline — not the API layer — is the source of truth for acceptable categories.
- `AssessmentResponse` — nests `Prediction`, `Uncertainty`, `Distribution`, `Decision` (`schemas/prediction.py`), `EvidenceItem`/`Citation` (`schemas/rag.py`), `GroundednessResult`/`AuditResult` (`schemas/groundedness.py`).
- Error responses: validation errors return `{"error": "validation_error", "detail": [...], "status_code": 422}`; unhandled exceptions return `{"error": "internal_error", "detail": "...", "status_code": 500}`. `HTTPException`s raised intentionally by route handlers use FastAPI's default `{"detail": "..."}` shape.

## CORS

Restricted to `NEUROSCREEN_ALLOWED_ORIGINS` (comma-separated, see `.env.example`) — defaults to the Vite dev server origin only. Update this for any other deployed frontend origin.

## Video observation

`backend/app/video/analyzer.py` uses MediaPipe's FaceLandmarker (Tasks API, model auto-downloaded to `backend/app/video/models/face_landmarker.task` on first use) to extract descriptive signals from a video: face-detection ratio, approximate gaze-toward-camera ratio, head-pose variability, and blink rate. It does **not** predict autism. This was investigated directly: no publicly available pretrained model classifies autism from video, and no labeled dataset for training one is available to this project (see `CHANGES.md`). The output is deliberately kept separate from `/api/v1/assessment`'s screening result rather than fused into it.
