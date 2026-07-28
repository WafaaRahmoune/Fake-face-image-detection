#!/usr/bin/env bash
# Lance FaceGuard. Voir le README pour l'installation.
set -e
cd "$(dirname "$0")"

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python app/server.py
else
  exec python3 app/server.py
fi
