#!/usr/bin/env bash
# Start FitFlow. Builds the frontend if it hasn't been built, then serves
# everything — UI and API — from one Flask process on one port.
set -euo pipefail
cd "$(dirname "$0")"

PY=venv/bin/python3
[ -x "$PY" ] || PY=python3

if [ ! -f frontend/dist/index.html ]; then
  echo "==> frontend/dist missing — building it (one time, ~30s)"
  ( cd frontend && npm install && npm run build )
fi

echo "==> starting FitFlow on http://127.0.0.1:${PORT:-5050}"
exec "$PY" app/server.py
