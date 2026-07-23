@echo off
cd /d "%~dp0"
where uv >nul 2>nul || powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv --python 3.11 .venv
uv pip install --python .venv -e ".[dev]"
echo Install ffmpeg (winget install Gyan.FFmpeg) if not present.
echo Converting Hebrew model...
.venv\Scripts\ct2-transformers-converter --model ivrit-ai/whisper-large-v3 --output_dir models\ivrit-whisper-ct2 --quantization int8 --copy_files tokenizer.json preprocessor_config.json
.venv\Scripts\python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
echo Setup complete.
pause
