#!/usr/bin/env bash
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"
# Bound to this machine explicitly. Recordings and transcripts of private
# meetings are served over this port with no authentication, so it must never be
# reachable from the network the laptop happens to be on.
.venv/bin/python -m uvicorn web.server:app --host 127.0.0.1 --port 8756 &
SERVER_PID=$!
sleep 2
open "http://localhost:8756"
echo "Transcrb is running. Close this window to stop."
wait $SERVER_PID
