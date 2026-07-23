#!/usr/bin/env bash
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"
.venv/bin/python -m uvicorn web.server:app --port 8756 &
SERVER_PID=$!
sleep 2
open "http://localhost:8756"
echo "Transcrb is running. Close this window to stop."
wait $SERVER_PID
