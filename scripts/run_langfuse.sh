#!/usr/bin/env bash
# Manage the self-hosted Langfuse v3 stack (observability for all projects).
#
#   ./scripts/run_langfuse.sh up      # start (detached); first run pulls ~GBs
#   ./scripts/run_langfuse.sh down    # stop
#   ./scripts/run_langfuse.sh logs    # follow logs
#   ./scripts/run_langfuse.sh ps      # status
#
# UI:   http://localhost:3000   (login: see deploy/langfuse/.env → LANGFUSE_INIT_USER_*)
# Keys: auto-provisioned on first boot; already wired into the project .env.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/deploy/langfuse/.env"
FILE="$ROOT/deploy/langfuse/docker-compose.yml"
COMPOSE=(docker compose --env-file "$ENVF" -f "$FILE")

case "${1:-up}" in
  up)
    echo "[langfuse] starting stack (web :3000)…"
    "${COMPOSE[@]}" up -d
    echo "[langfuse] waiting for http://localhost:3000 …"
    for i in $(seq 1 90); do
      if curl -s -o /dev/null http://localhost:3000/api/public/health 2>/dev/null; then
        echo "[langfuse] ✓ up → http://localhost:3000"; exit 0
      fi
      sleep 2
    done
    echo "[langfuse] still starting — check: ./scripts/run_langfuse.sh logs"
    ;;
  down) "${COMPOSE[@]}" down ;;
  logs) "${COMPOSE[@]}" logs -f --tail=120 ;;
  ps)   "${COMPOSE[@]}" ps ;;
  *) echo "usage: $0 [up|down|logs|ps]"; exit 1 ;;
esac
