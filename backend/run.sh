#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

cd "$HERE"
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
