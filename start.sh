#!/bin/bash
# Run FastAPI in the background, Streamlit in the foreground.
# HF Spaces requires a foreground process bound to $PORT (defaults to 7860).
set -e

# FastAPI on the internal port. The dashboard reaches it via localhost.
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Tear FastAPI down with the container.
trap "kill $FASTAPI_PID" EXIT

# Give FastAPI a head start so the dashboard's first /health probe succeeds.
sleep 3

streamlit run dashboard/app.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
