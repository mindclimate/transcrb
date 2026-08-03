#!/bin/bash
# Start the Transcrb API for launchd.
#
# THIS WRAPPER EXISTS FOR ONE MEASURED REASON. Under launchd the PATH is
# /usr/bin:/bin:/usr/sbin:/sbin and no shell profile is sourced. ffmpeg lives
# in Homebrew's bin, which is on neither, and engine/audio.py does
# shutil.which("ffmpeg") and raises. A plist that runs uvicorn directly
# therefore installs a service whose /api/health returns ok and whose every
# transcription fails with "ffmpeg not found". Measured on this machine,
# 2026-08-03.
#
# brew's prefix is resolved rather than hardcoded so this does not silently do
# nothing on an Intel Mac, where Homebrew lives under /usr/local. Same
# reasoning, and the same three-branch shape, as wrktbl-live/run.sh.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    PATH="$(brew --prefix)/bin:$PATH"
  elif [ -x /opt/homebrew/bin/brew ]; then
    PATH="$(/opt/homebrew/bin/brew --prefix)/bin:$PATH"
  elif [ -x /usr/local/bin/brew ]; then
    PATH="$(/usr/local/bin/brew --prefix)/bin:$PATH"
  fi
  export PATH
fi

command -v ffmpeg >/dev/null 2>&1 || {
  echo "FATAL: ffmpeg not found on PATH. Every transcription would fail." >&2
  exit 1
}

# Loopback only, and it must stay that way: recordings and transcripts of
# private meetings are served over this port with no authentication at all.
exec .venv/bin/python -m uvicorn web.server:app --host 127.0.0.1 --port 8756
