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
uv venv --python 3.11 .venv
uv pip install --python .venv -e ".[dev]"

# 3. Convert the ivrit.ai Hebrew model to CTranslate2 (int8)
#    NOTE: confirm the current best model id at https://huggingface.co/ivrit-ai
if [ ! -d "models/ivrit-whisper-ct2" ]; then
  echo "Converting the Hebrew model (this downloads a few GB)..."
  .venv/bin/ct2-transformers-converter \
    --model ivrit-ai/whisper-large-v3 \
    --output_dir models/ivrit-whisper-ct2 \
    --quantization int8 \
    --copy_files tokenizer.json preprocessor_config.json
fi

# 4. Warm the English model cache (faster-whisper auto-downloads large-v3)
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

echo "== Setup complete. Double-click 'Start Transcrb' to run. =="
echo "Reminder: set HF_TOKEN (HuggingFace) for speaker labels — see docs/SETUP.md."
