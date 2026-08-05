#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."          # this script lives in scripts/; work from the root
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

# Give a fresh install a config file of its own to edit.
if [ ! -f "config/config.toml" ]; then
  cp config/config.example.toml config/config.toml
  echo "Created config/config.toml — add your HuggingFace token there for speaker labels."
fi

# Reuse the token from config/config.toml for model downloads (higher rate
# limits, faster transfers). Downloads still work without it — models are public.
if [ -z "${HF_TOKEN:-}" ] && [ -f "config/config.toml" ]; then
  HF_TOKEN=$(.venv/bin/python -c "
import tomllib
with open('config/config.toml','rb') as f:
    print(tomllib.load(f).get('hf_token',''))
" 2>/dev/null || true)
  [ -n "$HF_TOKEN" ] && export HF_TOKEN && \
    echo "Using the HF token from config/config.toml for downloads."
fi

# 3. Fetch the ivrit.ai Hebrew model, already in CTranslate2 format.
#    ivrit.ai publishes pre-converted CT2 weights, so no local conversion is needed.
#    The turbo variant is the same fine-tune with 4 decoder layers instead of 32:
#    measured 4.0x faster on this hardware, and half the size (1.6GB vs 2.9GB),
#    which matters on a 16GB machine that was already swapping.
if [ ! -d "models/ivrit-whisper-turbo-ct2" ]; then
  echo "Downloading the Hebrew model (~1.6GB, one time)..."
  .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ivrit-ai/whisper-large-v3-turbo-ct2",
    local_dir="models/ivrit-whisper-turbo-ct2",
)
print("Hebrew model ready.")
PY
fi

# 4. Warm the model caches (faster-whisper auto-downloads these).
#    `base` is the language-ID model: it scans a recording for language switches
#    so a mixed Hebrew/English meeting is not transcribed as one language.
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# 5. Audio routing for recording both sides of a call. On macOS 14.2+ this only
#    verifies that a system-audio tap can be created — nothing to configure. On
#    older macOS it creates the Multi-Output and Aggregate devices in CoreAudio
#    that would otherwise have to be built by hand in Audio MIDI Setup.
if [ "$(uname)" = "Darwin" ]; then
  .venv/bin/python -m engine.macos_audio || \
    echo "! Audio setup did not complete — see docs/SETUP.md for the manual steps."

  # 6. The native recorder. Without it recording falls back to ffmpeg, whose
  #    avfoundation input holds one pending audio buffer and blocks the capture
  #    callback until it is read — about 10ms of tolerance for the whole
  #    capture. Measured under the macOS background throttle, ffmpeg lost 88.7%
  #    of a 30-second capture and the native recorder lost 0.1%.
  echo "Building the native recorder..."
  bash scripts/build-native.sh || \
    echo "! Recording will use ffmpeg and may lose audio on a busy Mac."
fi

echo "== Setup complete. Double-click 'Start Transcrb' to run. =="
echo "Reminder: set HF_TOKEN (HuggingFace) for speaker labels — see docs/SETUP.md."
