# Langfuse — self-hosted observability (MLOps)

Self-hosted **Langfuse v3** stack that traces every question asked through the
projects. Each trace is **named after the project** that answered (the backend
label), so you can filter/compare traces per project in the Langfuse UI.

## Stack
`docker-compose.yml` (official Langfuse v3) brings up: `langfuse-web` (:3000) ·
`langfuse-worker` · `postgres` · `clickhouse` · `redis` · `minio`.

Secrets and **auto-provisioned** org/project/user/API-keys live in `.env`
(gitignored). On first boot Langfuse creates the project **"Egyptian Civil Code
LLM"** with the keys already wired into the repo-root `.env`, so no manual setup.

## Run
```bash
./scripts/run_langfuse.sh up      # start (first run pulls ~2-3 GB of images)
./scripts/run_langfuse.sh ps      # status
./scripts/run_langfuse.sh logs    # follow logs
./scripts/run_langfuse.sh down    # stop
```
- **UI:** http://localhost:3000
- **Login:** email/password from `deploy/langfuse/.env` (`LANGFUSE_INIT_USER_*`).

## How the app connects
The project-root `.env` holds:
```
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-…   # matches LANGFUSE_INIT_PROJECT_PUBLIC_KEY
LANGFUSE_SECRET_KEY=sk-lf-…   # matches LANGFUSE_INIT_PROJECT_SECRET_KEY
```
The tracing helper `src/legal_explainer/observability/langfuse_tracing.py` reads
these and logs one trace per question. It **no-ops safely** if the stack is down,
so the apps never break when Langfuse isn't running.

## What gets traced
From the Unified UI, every submitted question creates a trace:
- **name** = the project/backend (e.g. `Neo4j Graph RAG`, `Finetuned · Qwen2.5-3B Knowledge adapter`)
- **input** = the question · **output** = the answer
- **metadata** = backend id, kind (chat/rag/agent), status line, #retrieved articles

> Prereq: Docker + the compose plugin. On WSL2, if hosts fail to resolve, set a
> public DNS: `echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf`.
