#!/usr/bin/env bash
# One-command setup + run for macOS / Linux.
#
#   ./run.sh          set up (first time only) and open the web app
#   ./run.sh test     run all tests
#   ./run.sh train    retrain the model and regenerate artifacts + docs
#
# Needs Python 3.11 or 3.12 (or the `uv` tool, which downloads Python itself).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
  echo "==> Creating virtual environment (.venv)"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
  else
    BASE=$(command -v python3.11 || command -v python3.12 || true)
    if [ -z "$BASE" ]; then
      echo "Python 3.11 or 3.12 was not found."
      echo "Install it from https://www.python.org/downloads/ (or install uv:"
      echo "  curl -LsSf https://astral.sh/uv/install.sh | sh ) and run ./run.sh again."
      exit 1
    fi
    "$BASE" -m venv .venv
  fi
fi

if ! "$PY" -c "import streamlit, sklearn, plotly" >/dev/null 2>&1; then
  echo "==> Installing packages (first run only, ~1-2 minutes)"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r requirements.txt
  else
    "$PY" -m pip install --upgrade pip -q
    "$PY" -m pip install -r requirements.txt
  fi
fi

if [ ! -f artifacts/kmeans_model.joblib ] || [ "$1" = "train" ]; then
  echo "==> Training the model"
  "$PY" -m src.train
  [ "$1" = "train" ] && exit 0
fi

if [ "$1" = "test" ]; then
  exec "$PY" -m pytest -v
fi

echo "==> Starting the app at http://localhost:8501  (Ctrl+C to stop)"
exec "$PY" -m streamlit run app/app.py
