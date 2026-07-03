#!/usr/bin/env bash
# Container entrypoint: apply database migrations, then start the API server.
set -euo pipefail

echo "Running database migrations (alembic upgrade head)..."
alembic upgrade head

echo "Starting Gunicorn (Uvicorn workers)..."
exec gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    --workers "${WEB_CONCURRENCY:-4}" \
    --bind "0.0.0.0:${PORT:-8000}" \
    --access-logfile - \
    --error-logfile -
