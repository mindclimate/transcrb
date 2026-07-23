#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:$PATH"

echo "== Transcrb first-time setup =="

# 1. Homebrew deps (macOS)
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Installing ffmpeg..."; brew install ffmpeg
fi
if ! brew list blackhole-2ch >/dev/null 2>&1; then
  echo "Installing BlackHole (virtual audio for system-audio capture)..."
  brew install blackhole-2ch || echo "BlackHole install skipped — system audio will be unavailable."
fi

# 2. Python 3.11 venv + deps
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if [ -d ".venv" ]; then
  echo "Reusing the existing .venv (delete it if you want a clean rebuild)."
else
  uv venv --python 3.11 .venv
fi
uv pip install --python .venv -e ".[dev]"

# 3. Fetch the ivrit.ai Hebrew model, already in CTranslate2 format.
#    ivrit.ai publishes pre-converted CT2 weights, so no local conversion is needed.
#    Alternatives: ivrit-ai/whisper-large-v3-turbo-ct2 (faster, slightly less accurate).
if [ ! -d "models/ivrit-whisper-ct2" ]; then
  echo "Downloading the Hebrew model (a few GB, one time)..."
  .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ivrit-ai/whisper-large-v3-ct2",
    local_dir="models/ivrit-whisper-ct2",
)
print("Hebrew model ready.")
PY
fi

# 4. Warm the English model cache (faster-whisper auto-downloads large-v3)
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

echo "== Setup complete. Double-click 'Start Transcrb' to run. =="
echo "Reminder: set HF_TOKEN (HuggingFace) for speaker labels — see docs/SETUP.md."
