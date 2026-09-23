# Deployment

Two paths, depending on what you need:

- **Path A — Local Docker Compose**: single machine, fastest to get running, no git/cloud account needed. Good for a demo or single-instance internal use.
- **Path B — Cloud deployment**: a real host reachable from the internet, behind HTTPS, with a durable database. Needs a few decisions this repo doesn't make for you (see Step 0).

Read Step 0 before either path — it covers two things that are easy to miss and will otherwise cause a broken build.

## Step 0: Two things to decide first

**1. There is no git repository yet.** `git status` in the project root fails with "not a git repository." If you're doing Path A only, you don't need one. For Path B (or just to have version control at all), initialize one first:

```bash
git init
git add .
git commit -m "Initial commit"
```

**2. The trained model files are intentionally excluded from git** (`.gitignore`: `models/artifacts/*.joblib`, `models/artifacts/*.pt`) — they're binaries, some hundreds of MB, and don't belong in a normal git history. This has a real consequence: **a fresh `git clone` will not have working models.** `models/registry/registry.json` and the metadata JSON *are* tracked and will reference artifact filenames that don't exist on disk — the backend raises `FileNotFoundError`/`RuntimeError` at startup or first prediction without them (`backend/app/ml/prediction_service.py`).

You have three options for getting the real artifacts onto whatever host builds/runs the image:

- **(a) Build where the artifacts already exist.** For Path A (`docker compose up --build` on this machine), this is already true — `docker build`'s context is your local disk, not git, so it doesn't matter that these files are gitignored. Nothing to do.
- **(b) Retrain as a build/deploy step.** `python scripts/train.py`, `scripts/train_toddler.py` are fast (small tabular data). `scripts/train_video_asd.py` is not (see `CHANGES.md` — real run took hours and needed a memory fix; only do this if you actually want the video-ASD model, which has ~59% test accuracy and is experimental by design).
- **(c) Store artifacts externally** (git-lfs, S3/GCS/Azure Blob, a private release) and fetch them in the Dockerfile or an entrypoint script before the app starts. This is the right call for a real CI/CD pipeline building from a fresh clone.

If you skip the video-ASD model specifically, the app degrades gracefully — `/api/v1/video/predict-asd` returns `503` with a clear message (`backend/app/api/routes/video.py`) instead of crashing. The three `.joblib` files (the actual screening models — child, toddler, and an older v1) are small (under 500KB total) and are the one thing you should not skip; the app's core feature depends on them.

---

## Path A: Local Docker Compose

```bash
docker compose up --build
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama pull gemma2:2b
```

What this starts (`docker-compose.yml`):

- **backend** — FastAPI + Uvicorn on `:8000`. Runs `alembic upgrade head` on container start, before serving. Persists its SQLite DB to the `neuroscreen-data` volume.
- **frontend** — production build served by nginx on `:8080`, proxying `/api` and `/health` to `backend:8000` (`frontend/nginx.conf`).
- **ollama** — local embedding/LLM server on `:11434`. The backend degrades gracefully without it (RAG retrieval and LLM explanation fall back to deterministic behavior), but you need it for the system to behave as designed — pull the two models above once the container is up, or nothing will actually generate explanations.

Verify:

```bash
curl http://localhost:8000/health
```

Open `http://localhost:8080` for the app.

---

## Path B: Cloud deployment

This assumes a host that can run Docker containers (a VPS, a container platform like Fly.io/Render/Railway, or your own Kubernetes) and public DNS you control. Adjust specifics to your platform; the steps and their order don't change.

### 1. Resolve Step 0 above
Push to a git remote (GitHub, etc.) after deciding how the model artifacts will reach the build.

### 2. Provision a real database
SQLite (the default) is fine for Path A but not for anything with more than one instance or that needs to survive container recreation without relying on a volume. Provision Postgres and set:

```
NEUROSCREEN_DATABASE_URL=postgresql://user:pass@host:5432/neuroscreen
```

The app uses SQLAlchemy, so this is a connection-string change plus `alembic upgrade head` — no code change. The backend container's `CMD` already runs migrations on start (`Dockerfile`).

### 3. Set required environment variables
All runtime config is env-var driven, prefix `NEUROSCREEN_` (`backend/app/core/config.py:Settings`), with safe local-dev defaults in `.env.example`. **Never commit a real `.env`** — it's git-ignored already. Before anything reachable outside localhost:

- `NEUROSCREEN_ALLOWED_ORIGINS` — set to your real frontend origin(s). The default only allows the local Vite dev server; leaving it as-is will silently break the deployed frontend with CORS errors.
- `NEUROSCREEN_DATABASE_URL` — your Postgres URL from step 2.
- `NEUROSCREEN_ENV=production`
- `NEUROSCREEN_OLLAMA_HOST` — point at wherever you're running Ollama (see step 5), or an alternative LLM endpoint if you've swapped providers.

### 4. Build and push images
```bash
docker build -t your-registry/neuroscreen-backend:latest .
docker build -t your-registry/neuroscreen-frontend:latest ./frontend
docker push your-registry/neuroscreen-backend:latest
docker push your-registry/neuroscreen-frontend:latest
```

If deploying to a GPU host and you want the video-ASD model to actually use it: `docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .` (default is CPU-only wheels — works everywhere, just slower for video feature extraction; this is what CI builds and what Path A uses).

### 5. Run Ollama somewhere real
Ollama is a stateful service that needs real CPU/RAM (the `gemma2:2b` + `nomic-embed-text` models need a few GB of RAM resident) — it does not fit a serverless/functions model well. Either run the `ollama/ollama` image on the same host/cluster as the backend (as Path A does), or point `NEUROSCREEN_OLLAMA_HOST` at a separately-hosted Ollama instance. Pull the two required models once it's reachable, same as Path A.

### 6. Put the frontend and backend behind TLS
`frontend/Dockerfile` serves a static build via nginx on port 80 — put a TLS-terminating reverse proxy or your platform's load balancer in front of both services (frontend origin and backend origin), and make sure `NEUROSCREEN_ALLOWED_ORIGINS` matches the real public frontend URL exactly (scheme + host, no trailing slash).

### 7. Review security headers for your actual domain
`backend/app/core/security.py`'s `SecurityHeadersMiddleware` covers cheap defaults (MIME sniffing, clickjacking, referrer leakage) but ships with no CSP/HSTS policy — tune those for your real domain before going live.

### 8. Smoke test
```bash
curl https://your-api-domain/health
```
Then open the frontend, run a screening end-to-end (all 3 wizard steps), and check `/monitoring` shows the resulting entry — this exercises the DB, the model registry, and (if configured) Ollama in one pass.

---

## CI

`.github/workflows/ci.yml` runs on push/PR to `main`:

1. **backend** — installs `requirements.txt`, runs `pytest` (no Ollama needed — see `docs/evaluation.md`).
2. **frontend** — `npm ci`, then `npm run build` (TypeScript check + production build).
3. **docker-build** — builds both images, gated on the above two passing. Building from a fresh CI checkout means Step 0's model-artifact question applies here too — as-is, this job builds an image without the `.joblib`/`.pt` files unless you've addressed that (option b or c above).

## Known gaps / things to revisit

- No automated deploy step (CI builds images but doesn't push/deploy them anywhere) — wire that up once you've picked a host.
- No git-lfs or artifact-fetch step configured yet for the model binaries (Step 0, option c) — needed for a real CI/CD pipeline from a fresh clone.
- No CSP/HSTS policy tuned yet (Step 7).
